"""qwen_asr_ft — fine-tune and evaluate Qwen3-ASR on your own speech data.

Modules: config (YAML + overrides), data (manifests -> SFT jsonl), context (prompt building + augmentation),
train (full-parameter SFT), infer (batch transcription), evaluate / metrics / text (MER scoring), cli (`qasr`).
"""

__version__ = "0.1.0"
