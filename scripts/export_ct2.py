"""Merge a LoRA adapter into Whisper (fp32) and convert it to CTranslate2 for faster-whisper (E7).

    python scripts/export_ct2.py --base openai/whisper-large-v3 --revision 06f233fe… \
        --adapter runs/lora_large_v3_C/checkpoints/step_600 --out runs/ct2/C600_fp16

Weights are stored as float16; int8_float16 etc. are chosen at load time (compute_type), so one export serves
every quantization variant. Only weights move to CTranslate2: our Python decoding (allowed_languages, loop guard)
is re-implemented in src/whisperflow/eval/ct2_backend.py.
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from transformers import WhisperForConditionalGeneration, WhisperProcessor

ap = argparse.ArgumentParser()
ap.add_argument("--base", required=True)
ap.add_argument("--revision")
ap.add_argument("--adapter")
ap.add_argument("--out", required=True)
ap.add_argument("--quantization", default="float16")
args = ap.parse_args()

out = Path(args.out)
merged = out.with_name(out.name + "_hf_merged")
model = WhisperForConditionalGeneration.from_pretrained(args.base, revision=args.revision, dtype=torch.float32)
if args.adapter:
    from peft import PeftModel

    model = PeftModel.from_pretrained(model, args.adapter).merge_and_unload()
model.save_pretrained(merged)
WhisperProcessor.from_pretrained(args.base, revision=args.revision).save_pretrained(merged)
# transformers 5 writes processor_config.json; the converter and faster-whisper read preprocessor_config.json
# (feature_size 128 for large-v3 — without it faster-whisper falls back to 80 mel bins). Copy the original.
shutil.copy(hf_hub_download(args.base, "preprocessor_config.json", revision=args.revision),
            merged / "preprocessor_config.json")
print(f"merged -> {merged}", flush=True)

conv = Path(sys.executable).with_name("ct2-transformers-converter")
subprocess.run([str(conv), "--model", str(merged), "--output_dir", str(out), "--quantization", args.quantization,
                "--copy_files", "tokenizer.json", "preprocessor_config.json", "--force"], check=True)
shutil.rmtree(merged)  # fp32 merged copy (~6 GB) is only an intermediate
print(f"ct2 -> {out}")
