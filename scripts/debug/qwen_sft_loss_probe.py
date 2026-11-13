"""Per-token loss of a Qwen3-ASR model on a few SFT samples, built by the official collator (2026-11-09).
Why: the smoke run's first logged loss was 9.86 (grad norm 418) although the base model transcribes these clips well.
    .venv-qwen/bin/python scripts/debug/qwen_sft_loss_probe.py <model_dir> runs/qwen_sft/I2/train.jsonl [n]
"""
import json
import sys

import torch

sys.path.insert(0, "scripts")
from qwen3_asr_sft import (DataCollatorForQwen3ASRFinetuning, make_preprocess_fn_prefix_only,  # noqa: E402
                           patch_outer_forward)
from qwen_asr import Qwen3ASRModel  # noqa: E402

model_dir, data, n = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 4
w = Qwen3ASRModel.from_pretrained(model_dir, dtype=torch.bfloat16, device_map="cuda:0")
model, proc = w.model, w.processor
patch_outer_forward(model)
model.eval()
rows = [json.loads(line) for line in open(data, encoding="utf-8")][:n]
pre = make_preprocess_fn_prefix_only(proc)
feats = [pre(r) for r in rows]
coll = DataCollatorForQwen3ASRFinetuning(processor=proc)
print("padding_side", proc.tokenizer.padding_side)
for f in feats:  # one at a time: no padding effects
    batch = {k: (v.to("cuda:0", dtype=torch.bfloat16) if v.is_floating_point() else v.to("cuda:0"))
             for k, v in coll([f]).items()}
    with torch.no_grad():
        out = model(**batch)
    logits = out.logits[0, :-1].float()
    labels = batch["labels"][0, 1:]
    mask = labels != -100
    tok_loss = torch.nn.functional.cross_entropy(logits[mask], labels[mask], reduction="none")
    ids = labels[mask].tolist()
    pred = logits[mask].argmax(-1).tolist()
    print(f"\n== {f['target'][:60]}  model loss {out.loss.item():.3f}  mean token loss {tok_loss.mean().item():.3f}"
          f"  n_label_tokens {mask.sum().item()} / seq {labels.numel()}")
    for t, p, lo in zip(ids, pred, tok_loss.tolist()):
        if lo > 1.0:
            print(f"   high {lo:6.2f}  label {proc.tokenizer.decode([t])!r}  pred {proc.tokenizer.decode([p])!r}")
    print("   label text:", proc.tokenizer.decode(ids)[:120])
