"""LoRA fine-tuning of Whisper with a plain PyTorch loop (no HF Trainer, so every step is visible).

Usage:
    wf-train --config configs/train_lora.yaml --run-id smoke --max-steps 20          # smoke test
    wf-train --config configs/train_lora.yaml --run-id overfit --overfit 16 --max-steps 150
    wf-train --config configs/train_lora.yaml --run-id lora_v1                       # full run

Writes runs/<run_id>/: config.yaml, log.jsonl (one line per logged step / eval), adapter/ (LoRA weights),
best_adapter/ (lowest val loss). Practice target: the loop in `train()` (docs/learning/training_walkthrough.md).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from whisperflow.training.data import WhisperCollator, WhisperDataset, load_examples, usable_for_loss


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def pick_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(cfg: dict):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    name, rev = cfg["base_model"]["name_or_path"], cfg["base_model"]["revision"]
    processor = WhisperProcessor.from_pretrained(name, revision=rev)
    model = WhisperForConditionalGeneration.from_pretrained(name, revision=rev)
    model.config.use_cache = False  # the KV cache is only useful for generation
    if cfg["training"].get("gradient_checkpointing"):
        # Recompute activations during backward instead of storing them: much less memory, ~30% slower.
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    lcfg = cfg["lora"]
    model = get_peft_model(model, LoraConfig(
        r=lcfg["r"], lora_alpha=lcfg["alpha"], lora_dropout=lcfg["dropout"],
        target_modules=lcfg["target_modules"], bias="none"))
    return model, processor


def count_params(model) -> tuple[int, int]:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, sum(p.numel() for p in model.parameters())


@torch.no_grad()
def evaluate(model, loader, device: str, max_batches: int | None = None, autocast=contextlib.nullcontext) -> float:
    """Mean token-level cross-entropy on the validation loader (cheap proxy; MER comes from wf-score)."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        n = int((batch["labels"] != -100).sum())
        with autocast():
            total_loss += model(**batch).loss.item() * n  # .loss is the mean over non-ignored tokens
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


def train(cfg: dict, run_dir: Path, *, max_steps: int | None, overfit: int | None, resume: bool = False) -> dict:
    t, d = cfg["training"], cfg["data"]
    set_seed(t["seed"])
    device = pick_device()
    model, processor = build_model(cfg)
    model.to(device)
    trainable, total = count_params(model)
    print(f"device={device} trainable params {trainable:,} / {total:,} ({trainable / total:.2%})")

    train_recs = load_examples(d["manifests"], d["train_split"])
    val_recs = load_examples(d["manifests"], d["eval_split"])
    train_recs, n_drop_train = usable_for_loss(train_recs, processor.tokenizer, d["max_label_tokens"])
    val_recs, n_drop_val = usable_for_loss(val_recs, processor.tokenizer, d["max_label_tokens"])
    print(f"dropped (audio > 30 s or label > {d['max_label_tokens']} tokens): "
          f"train {n_drop_train}, val {n_drop_val}")
    if overfit:  # sanity check: a tiny set the model must be able to memorise
        train_recs = random.Random(t["seed"]).sample(train_recs, overfit)
        val_recs = train_recs
    ts = d.get("timestamp_ratio", 0.0)
    mk = lambda recs: WhisperDataset(recs, processor, d["data_root"], d["max_label_tokens"], ts)  # noqa: E731
    collate = WhisperCollator()
    g = torch.Generator().manual_seed(t["seed"])
    train_loader = torch.utils.data.DataLoader(mk(train_recs), batch_size=t["per_device_batch_size"],
                                               shuffle=True, generator=g, collate_fn=collate,
                                               num_workers=t.get("num_workers", 0))
    val_loader = torch.utils.data.DataLoader(mk(val_recs), batch_size=t["per_device_batch_size"],
                                             shuffle=False, collate_fn=collate,
                                             num_workers=t.get("num_workers", 0))

    # bf16 autocast on CUDA (Ampere+): half the activation memory, faster matmuls, no loss scaling needed.
    # MPS / CPU keep fp32 (bf16 stability on MPS not measured).
    use_bf16 = t.get("precision") == "bf16" and device == "cuda" and torch.cuda.is_bf16_supported()
    autocast = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if use_bf16 else contextlib.nullcontext
    print(f"precision: {'bf16 autocast' if use_bf16 else 'fp32'}")
    accum = t["gradient_accumulation_steps"]
    steps_per_epoch = math.ceil(len(train_loader) / accum)
    total_steps = max_steps or steps_per_epoch * t["num_epochs"]
    from transformers import get_linear_schedule_with_warmup

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=t["learning_rate"], weight_decay=t.get("weight_decay", 0.0))
    scheduler = get_linear_schedule_with_warmup(optimizer, int(t["warmup_ratio"] * total_steps), total_steps)
    print(f"train clips {len(train_recs)}, val clips {len(val_recs)}, batch {t['per_device_batch_size']}x{accum}, "
          f"{steps_per_epoch} optimizer steps/epoch, total {total_steps}, timestamp_ratio {ts}")

    log = open(run_dir / "log.jsonl", "a", encoding="utf-8")
    eval_every, log_every = t["eval_every_steps"], t.get("log_every_steps", 10)
    eval_batches = t.get("eval_max_batches")
    patience, min_delta = t.get("early_stopping_patience"), t.get("early_stopping_min_delta", 0.0)
    state_path = run_dir / "training_state.pt"
    best, bad_evals, step = float("inf"), 0, 0
    if resume and state_path.exists():
        # Continue an interrupted run: LoRA weights + optimizer + LR schedule + counters. The data order is
        # reshuffled from step onwards (exact batch position inside the epoch is not restored).
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file

        set_peft_model_state_dict(model, load_file(run_dir / "adapter" / "adapter_model.safetensors"))
        state = torch.load(state_path, map_location=device, weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        best, bad_evals, step = state["best"], state["bad_evals"], state["step"]
        g.manual_seed(t["seed"] + step)
        print(f"resumed from step {step} (best val loss {best:.4f})")
    else:
        # step-0 val loss: the untouched base model, so later numbers have a reference point
        rec = {"step": 0, "val_loss": round(evaluate(model, val_loader, device, eval_batches, autocast), 4)}
        print(json.dumps(rec))
        log.write(json.dumps(rec) + "\n")
        log.flush()
    start_step, micro, running, since_log, t0 = step, 0, 0.0, 0, time.time()
    stop = False
    model.train()
    while step < total_steps and not stop:
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            with autocast():
                loss = model(**batch).loss / accum   # scale so accumulated grads = mean over the big batch
            loss.backward()
            running += loss.item()
            micro += 1
            if micro % accum:
                continue                        # keep accumulating gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), t["max_grad_norm"])
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            since_log += 1
            if step % log_every == 0 or step == 1:
                rec = {"step": step, "train_loss": round(running / since_log, 4),
                       "lr": scheduler.get_last_lr()[0], "grad_norm": round(float(grad_norm), 3),
                       "sec_per_step": round((time.time() - t0) / max(step - start_step, 1), 2)}
                if device == "mps":
                    rec["mps_mem_gb"] = round(torch.mps.driver_allocated_memory() / 1e9, 2)
                elif device == "cuda":
                    rec["cuda_peak_mem_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
                print(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                running, since_log = 0.0, 0
            if step % eval_every == 0 or step == total_steps:
                val_loss = evaluate(model, val_loader, device, eval_batches, autocast)
                rec = {"step": step, "val_loss": round(val_loss, 4)}
                print(json.dumps(rec))
                log.write(json.dumps(rec) + "\n")
                log.flush()
                model.save_pretrained(run_dir / "adapter")
                if t.get("keep_all_checkpoints"):  # select later by decoded val MER, not only by loss
                    model.save_pretrained(run_dir / "checkpoints" / f"step_{step}")
                if val_loss < best - min_delta:
                    best, bad_evals = val_loss, 0
                    model.save_pretrained(run_dir / "best_adapter")
                else:
                    bad_evals += 1
                torch.save({"step": step, "best": best, "bad_evals": bad_evals,
                            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict()}, state_path)
                if patience and bad_evals >= patience:  # early stopping: val loss stalled for `patience` evals
                    rec = {"step": step, "early_stop": True, "evals_without_improvement": bad_evals}
                    print(json.dumps(rec))
                    log.write(json.dumps(rec) + "\n")
                    stop = True
            if step >= total_steps or stop:
                break
    log.close()
    return {"steps": step, "best_val_loss": best, "early_stopped": stop,
            "sec_per_step": (time.time() - t0) / max(step - start_step, 1),
            "trainable_params": trainable}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/train_lora.yaml")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--max-steps", type=int, help="stop after N optimizer steps (smoke / overfit tests)")
    ap.add_argument("--overfit", type=int, help="train AND evaluate on N random train clips")
    ap.add_argument("--resume", action="store_true", help="continue runs/<run_id> from its last saved state")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    run_dir = Path("runs") / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "cli": vars(args)}, allow_unicode=True),
                                         encoding="utf-8")
    summary = train(cfg, run_dir, max_steps=args.max_steps, overfit=args.overfit, resume=args.resume)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
