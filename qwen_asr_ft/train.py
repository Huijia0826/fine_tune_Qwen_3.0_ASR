# Adapted from QwenLM/Qwen3-ASR finetuning/qwen3_asr_sft.py (fetched 2026-11-09).
# Copyright 2026 The Alibaba Qwen team. Licensed under the Apache License, Version 2.0
# (http://www.apache.org/licenses/LICENSE-2.0). Changes: settings come from the framework config instead of
# argparse; seed, save_only_model, max_steps and eval_steps exposed; eval disabled when there is no val file;
# base-model download / revision handling. The training maths (prompt masking, loss, optimiser) is unchanged.
"""Full-parameter SFT of Qwen3-ASR.

Each training row is {"audio", "prompt", "text"}. The chat template puts `prompt` in the system turn and the audio in
the user turn; the target text follows the assistant prefix. Labels of the prefix are set to -100, so the loss is
computed on the output text only — the model conditions on the context but is never trained to reproduce it.
"""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")
HF_FILES = ["config.json", "generation_config.json", "preprocessor_config.json", "processor_config.json",
            "tokenizer_config.json", "tokenizer.json", "special_tokens_map.json", "chat_template.json",
            "merges.txt", "vocab.json"]


def checkpoints(output_dir: str | Path) -> list[Path]:
    """checkpoint-<step> folders, sorted by step."""
    d = Path(output_dir)
    if not d.is_dir():
        return []
    found = [(int(m.group(1)), p) for p in d.iterdir() if p.is_dir() and (m := _CKPT_RE.match(p.name))]
    return [p for _, p in sorted(found)]


def resolve_base(model_cfg: dict) -> str:
    """Local folder, or a Hugging Face repo id (downloaded once, pinned to `revision` if given)."""
    base = str(model_cfg["base"])
    if Path(base).expanduser().is_dir():
        return str(Path(base).expanduser())
    from huggingface_hub import snapshot_download

    return snapshot_download(base, revision=model_cfg.get("revision"))


def build_prefix_messages(prompt: str, audio_array):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def make_preprocess_fn(processor):
    def _preprocess(ex: dict[str, Any]) -> dict[str, Any]:
        prompt = ex.get("prompt", "") or ""
        prefix_text = processor.apply_chat_template(
            [build_prefix_messages(prompt, None)], add_generation_prompt=True, tokenize=False)[0]
        return {"prompt": prompt, "audio": ex["audio"], "target": ex["text"], "prefix_text": prefix_text}

    return _preprocess


def load_audio(path: str, sr: int = 16000):
    import librosa

    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


@dataclass
class DataCollatorForQwen3ASRFinetuning:
    """Encodes prefix + target + EOS, and the prefix alone; the prefix length masks the labels (-100)."""

    processor: Any
    sampling_rate: int = 16000

    def __call__(self, features: list[dict[str, Any]]):
        prefix_texts = [f["prefix_text"] for f in features]
        targets = [f["target"] for f in features]
        eos = self.processor.tokenizer.eos_token or ""
        full_texts = [p + t + eos for p, t in zip(prefix_texts, targets)]
        audios = [load_audio(f["audio"], sr=self.sampling_rate) for f in features]

        full_inputs = self.processor(text=full_texts, audio=audios, return_tensors="pt", padding=True,
                                     truncation=False)
        prefix_inputs = self.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True,
                                       truncation=False)
        prefix_lens = prefix_inputs["attention_mask"].sum(dim=1).tolist()
        labels = full_inputs["input_ids"].clone()
        for i, pl in enumerate(prefix_lens):
            labels[i, :pl] = -100          # no loss on system prompt + audio placeholder
        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100  # no loss on padding
        full_inputs["labels"] = labels
        return full_inputs


def patch_outer_forward(model) -> None:
    """The trainable forward lives in model.thinker; route the wrapper's forward to it."""
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker"):
        raise RuntimeError("model has no `.thinker`: incompatible qwen_asr version (expected qwen-asr==0.0.6)")

    def forward(self, input_ids=None, attention_mask=None, input_features=None, feature_attention_mask=None,
                labels=None, **kwargs):
        return self.thinker.forward(input_ids=input_ids, attention_mask=attention_mask,
                                    input_features=input_features, feature_attention_mask=feature_attention_mask,
                                    labels=labels, **kwargs)

    cls.forward = forward
    cls._forward_patched = True


def train(cfg: dict) -> Path:
    import torch
    from datasets import load_dataset
    from qwen_asr import Qwen3ASRModel
    from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments

    tcfg = cfg["train"]
    work = Path(cfg["work_dir"])
    train_file, val_file = work / "sft" / "train.jsonl", work / "sft" / "val.jsonl"
    if not train_file.exists():
        raise FileNotFoundError(f"{train_file} not found: run `qasr prepare` first")
    out_dir = work / "train"
    base = resolve_base(cfg["model"])

    class CastFloatInputsTrainer(Trainer):
        def _prepare_inputs(self, inputs):
            inputs = super()._prepare_inputs(inputs)
            dtype = getattr(self.model, "dtype", None)
            if dtype is not None:
                for k, v in list(inputs.items()):
                    if torch.is_tensor(v) and v.is_floating_point():
                        inputs[k] = v.to(dtype=dtype)
            return inputs

    class MakeEveryCheckpointInferable(TrainerCallback):
        """Copy tokenizer / processor files into every checkpoint so each one can be decoded directly."""

        def on_save(self, args, state, control, **kwargs):
            if args.process_index == 0:
                ckpt = Path(args.output_dir) / f"checkpoint-{state.global_step}"
                ckpt.mkdir(parents=True, exist_ok=True)
                for fn in HF_FILES:
                    if (Path(base) / fn).exists():
                        shutil.copy2(Path(base) / fn, ckpt / fn)
            return control

    use_bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
    wrapper = Qwen3ASRModel.from_pretrained(base, dtype=torch.bfloat16 if use_bf16 else torch.float16,
                                            device_map=None)
    model, processor = wrapper.model, wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)

    has_val = val_file.exists() and val_file.stat().st_size > 0
    raw = load_dataset("json", data_files={"train": str(train_file), **({"validation": str(val_file)} if has_val else {})})
    ds = raw.map(make_preprocess_fn(processor), num_proc=1)
    keep = {"prompt", "audio", "target", "prefix_text"}
    for split in ds.keys():
        drop = [c for c in ds[split].column_names if c not in keep]
        if drop:
            ds[split] = ds[split].remove_columns(drop)

    nw = int(tcfg["num_workers"])
    args = TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=int(tcfg["batch_size"]),
        gradient_accumulation_steps=int(tcfg["grad_acc"]),
        learning_rate=float(tcfg["lr"]),
        num_train_epochs=float(tcfg["epochs"]),
        max_steps=int(tcfg["max_steps"]),
        logging_steps=int(tcfg["log_steps"]),
        lr_scheduler_type=tcfg["lr_scheduler_type"],
        warmup_ratio=float(tcfg["warmup_ratio"]),
        dataloader_num_workers=nw,
        dataloader_pin_memory=True,
        dataloader_persistent_workers=nw > 0,
        dataloader_prefetch_factor=2 if nw > 0 else None,
        save_strategy="steps",
        save_steps=int(tcfg["save_steps"]),
        save_total_limit=int(tcfg["save_total_limit"]),
        save_safetensors=True,
        save_only_model=bool(tcfg["save_only_model"]),
        eval_strategy="steps" if has_val else "no",
        eval_steps=int(tcfg["eval_steps"]) or int(tcfg["save_steps"]),
        seed=int(tcfg["seed"]),
        bf16=use_bf16,
        fp16=not use_bf16,
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        report_to=tcfg.get("report_to", "none"),
    )
    trainer = CastFloatInputsTrainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds.get("validation"),
        data_collator=DataCollatorForQwen3ASRFinetuning(processor=processor, sampling_rate=int(tcfg["sr"])),
        tokenizer=processor.tokenizer, callbacks=[MakeEveryCheckpointInferable()])
    resume = None
    if tcfg.get("resume"):
        ck = checkpoints(out_dir)
        resume = str(ck[-1]) if ck else None
    trainer.train(resume_from_checkpoint=resume)
    (out_dir / "done").touch()
    return out_dir
