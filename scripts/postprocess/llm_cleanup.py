"""Post-process ASR outputs with an open Qwen3 instruct model and the production cleanup prompts (2026-11-10).

Owner: our transcripts are often right, but disfluencies are not removed as cleanly as production's final output.
This runs the production prompts unchanged (reports/postprocess_prompts/: transcription-fully-cleanup-chinese v8 for
text with Chinese characters, transcription-fully-cleanup v32 otherwise) with the transcript as the user message
"<TRANSCRIPT>…</TRANSCRIPT>" (the production format, reports/postprocess_data_audit/report.md §3), greedy decoding,
and writes a new run directory with the same predictions.jsonl layout (hypothesis = cleaned text, hypothesis_asr =
the input) so every scorer works on it unchanged.

    .venv/bin/python scripts/postprocess/llm_cleanup.py bv1_qwen3asr_J3_step_600_ctx --model Qwen/Qwen3-4B-Instruct-2507
"""

import argparse
import json
import re
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROMPTS = Path("reports/postprocess_prompts")
ZH = (PROMPTS / "transcription-fully-cleanup-chinese__v8.txt").read_text(encoding="utf-8")
EN = (PROMPTS / "transcription-fully-cleanup__v32.txt").read_text(encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+", help="run ids under reports/runs/")
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--tag", default="pp_q3_4b")
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(args.model)
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda:0").eval()
    for run in args.runs:
        src = Path("reports/runs") / run
        rows = [json.loads(line) for line in open(src / "predictions.jsonl", encoding="utf-8")]
        out_dir = Path("reports/runs") / f"{run}_{args.tag}"
        out_dir.mkdir(parents=True, exist_ok=True)
        t0, cleaned = time.time(), []
        for k in range(0, len(rows), args.batch):
            batch = rows[k:k + args.batch]
            texts = []
            for r in batch:
                system = ZH if re.search(r"[一-鿿]", r["hypothesis"]) else EN
                msgs = [{"role": "system", "content": system},
                        {"role": "user", "content": f"<TRANSCRIPT>{r['hypothesis']}</TRANSCRIPT>"}]
                texts.append(tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))
            enc = tok(texts, return_tensors="pt", padding=True).to(model.device)
            longest = max(len(tok(r["hypothesis"]).input_ids) for r in batch)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=int(longest * 1.5) + 32, do_sample=False)
            for r, g in zip(batch, gen, strict=True):
                out = tok.decode(g[enc.input_ids.shape[1]:], skip_special_tokens=True).strip()
                out = re.sub(r"^<TRANSCRIPT>|</TRANSCRIPT>$", "", out).strip()
                cleaned.append({**r, "hypothesis_asr": r["hypothesis"], "hypothesis": out or r["hypothesis"]})
            print(f"  {run}: {min(k + args.batch, len(rows))}/{len(rows)}", flush=True)
        with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
            for r in cleaned:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        meta = json.loads((src / "run_meta.json").read_text()) if (src / "run_meta.json").exists() else {}
        meta.update({"run_id": out_dir.name, "postprocess": {"model": args.model, "prompts": [
            "transcription-fully-cleanup-chinese__v8", "transcription-fully-cleanup__v32"], "source_run": run,
            "seconds": round(time.time() - t0, 1)}})
        (out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{out_dir}: {len(cleaned)} rows, {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
