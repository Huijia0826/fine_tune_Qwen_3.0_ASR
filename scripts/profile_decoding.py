"""Where does the time go in one dictation, and why is greedy faster than beam 5? (E7 follow-up, 2026-11-02)

    python scripts/profile_decoding.py --config configs/experiments/latency/hf_bf16_beam5.yaml [--limit N]

Transformers backend only (faster-whisper's loop is C++ and cannot be split from Python). Two parts:
1. Stage breakdown per short clip (<= 30 s, users' val), for beam 5 and greedy: feature extraction, language ID
   (encoder + 1 decoder step, as HFWhisperTranscriber._detect_languages), encoder alone, generate (encoder +
   decoder loop; decoder loop = generate − encoder), post-processing (detokenise + t2s). Decoder steps = length of
   the returned sequence (for beam search the loop can run a few steps longer, so per-step time is approximate).
   No loop guard here (plain beam / greedy) so every stage is a single pass.
2. Micro-benchmark of ONE decoder step with a KV cache, batch 1 / 2 / 5 / 10: if batch 5 costs about the same as
   batch 1, the step is bound by reading the weights, not by arithmetic. Effective bandwidth = decoder weight
   bytes / step time.
Output: reports/runs/<run-id>/profile.json and a printed summary.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from whisperflow.data.audio import load_16k_mono
from whisperflow.data.common import load_yaml
from whisperflow.data.schema import load_manifest
from whisperflow.eval.latency import USERS_VAL
from whisperflow.eval.transcribe import HFWhisperTranscriber
from whisperflow.text.normalize import _t2s


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elif torch.backends.mps.is_available():
        torch.mps.synchronize()


def timed(fn):
    sync()
    t0 = time.perf_counter()
    out = fn()
    sync()
    return out, time.perf_counter() - t0


def stage_breakdown(tr: HFWhisperTranscriber, audio: np.ndarray, num_beams: int) -> dict:
    m, proc, dev, dt = tr.model, tr.processor, tr.device, tr.dtype
    feats, t_feat = timed(lambda: proc.feature_extractor([audio], sampling_rate=16000, return_tensors="pt",
                                                         return_attention_mask=True))
    x = feats.input_features.to(dev, dtype=dt)
    gc = m.generation_config
    allowed = ["zh", "en"]
    cand = torch.tensor([gc.lang_to_id[f"<|{lang}|>"] for lang in allowed], device=dev)
    start = torch.full((1, 1), gc.decoder_start_token_id, device=dev)
    with torch.inference_mode():
        logits, t_lid = timed(lambda: m(input_features=x, decoder_input_ids=start, use_cache=False).logits[:, -1])
        lang = allowed[int(logits[:, cand].argmax(-1))]
        _, t_enc = timed(lambda: m.model.encoder(x))
        out, t_gen = timed(lambda: m.generate(x, attention_mask=feats.attention_mask.to(dev), language=lang,
                                              task="transcribe", num_beams=num_beams, do_sample=False,
                                              max_new_tokens=200, return_timestamps=False))
    seq = out["sequences"] if isinstance(out, dict) else out
    n_prompt = 4  # <|startoftranscript|><|lang|><|transcribe|><|notimestamps|>
    steps = int(seq.shape[1]) - n_prompt
    _, t_post = timed(lambda: _t2s().convert(proc.batch_decode(seq, skip_special_tokens=True)[0]))
    t_dec = t_gen - t_enc
    return {"feat": t_feat, "lid": t_lid, "encoder": t_enc, "decoder_loop": t_dec, "post": t_post,
            "total": t_feat + t_lid + t_gen + t_post, "steps": steps, "per_step": t_dec / max(steps, 1)}


def decoder_step_bench(tr: HFWhisperTranscriber, batch_sizes=(1, 2, 5, 10), prefix: int = 40,
                       iters: int = 30) -> dict:
    """Time one cached decoder step (+ output projection) at several batch sizes, same cache length."""
    m, dev, dt = tr.model, tr.device, tr.dtype
    dec = m.model.decoder
    d_model = m.config.d_model
    res = {}
    for b in batch_sizes:
        enc = torch.randn(b, 1500, d_model, device=dev, dtype=dt)
        ids = torch.randint(0, 50000, (b, prefix), device=dev)
        with torch.inference_mode():
            pre = dec(input_ids=ids, encoder_hidden_states=enc, use_cache=True)
            cache = pre.past_key_values
            nxt = torch.randint(0, 50000, (b, 1), device=dev)
            times = []
            for i in range(iters + 3):  # first 3 = warm-up
                def step(nxt=nxt, enc=enc, cache=cache):
                    return m.proj_out(dec(input_ids=nxt, encoder_hidden_states=enc, past_key_values=cache,
                                          use_cache=True).last_hidden_state)
                _, t = timed(step)
                if i >= 3:
                    times.append(t)
        res[b] = float(np.median(times))
    return res


def per_step_weight_bytes(model) -> int:
    """Weight bytes one cached decoder step must read. Excluded: cross-attention k/v projections (applied to the
    encoder output once, then cached) and the position table (one row per step). The token embedding is counted
    once, because proj_out is tied to it and reads the full matrix for the logits."""
    total = 0
    for name, p in model.model.decoder.named_parameters():
        if "encoder_attn.k_proj" in name or "encoder_attn.v_proj" in name or "embed_positions" in name:
            continue
        total += p.numel() * p.element_size()
    if model.proj_out.weight.data_ptr() != model.model.decoder.embed_tokens.weight.data_ptr():
        total += model.proj_out.weight.numel() * model.proj_out.weight.element_size()
    return total


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--run-id", default="profile_decoding")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-cudnn-sdp", action="store_true",
                    help="disable the cuDNN SDPA backend (it costs ~1.6 ms CPU per attention call in a bare step)")
    args = ap.parse_args()
    if args.no_cudnn_sdp:
        torch.backends.cuda.enable_cudnn_sdp(False)
    cfg = load_yaml(args.config)
    tr = HFWhisperTranscriber.from_config(cfg["model"], cfg["decoding"], cfg["runtime"])
    root = Path(cfg["data"]["data_root"])
    recs = sorted((r for mf in USERS_VAL for r in load_manifest(mf) if r.split == "val"),
                  key=lambda r: r.utterance_id)
    audios = [a for a in (load_16k_mono(root / r.audio_path) for r in recs) if len(a) <= 30 * 16000]
    if args.limit:
        audios = audios[:args.limit]
    for a in audios[:3]:  # warm-up
        stage_breakdown(tr, a, 5)
        stage_breakdown(tr, a, 1)

    rows = {5: [], 1: []}
    for i, a in enumerate(audios):
        for nb in (5, 1):
            rows[nb].append({"dur": len(a) / 16000, **stage_breakdown(tr, a, nb)})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(audios)}", flush=True)

    keys = ["feat", "lid", "encoder", "decoder_loop", "post", "total"]
    summary = {}
    for nb, rs in rows.items():
        summary[f"beam{nb}"] = {
            "n": len(rs),
            "median_s": {k: round(float(np.median([r[k] for r in rs])), 4) for k in keys},
            "share_of_total_pct": {k: round(100 * sum(r[k] for r in rs) / sum(r["total"] for r in rs), 1)
                                   for k in keys[:-1]},
            "median_steps": float(np.median([r["steps"] for r in rs])),
            "median_per_step_ms": round(1000 * float(np.median([r["per_step"] for r in rs])), 2)}
    step = decoder_step_bench(tr)
    dec_bytes = per_step_weight_bytes(tr.model)
    summary["decoder_step_ms_by_batch"] = {b: round(1000 * t, 2) for b, t in step.items()}
    summary["decoder_weight_gb"] = round(dec_bytes / 1e9, 3)
    summary["effective_bandwidth_tb_s_batch1"] = round(dec_bytes / step[1] / 1e12, 2)
    summary["device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else tr.device
    summary["dtype"] = str(tr.dtype)
    summary["cudnn_sdp_enabled"] = torch.backends.cuda.cudnn_sdp_enabled()

    out = Path("reports/runs") / args.run_id
    out.mkdir(parents=True, exist_ok=True)
    per_clip = {f"beam{k}": v for k, v in rows.items()}
    (out / "profile.json").write_text(json.dumps({"summary": summary, "rows": per_clip},
                                                 indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
