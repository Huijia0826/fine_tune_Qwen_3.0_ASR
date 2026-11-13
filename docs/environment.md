# Environment report (measured 2026-10-27)

| item | value |
|---|---|
| Machine | MacBook Pro, Apple M5, 10 CPU cores, 10-core GPU (Metal 4) |
| Memory | 24 GB unified (shared CPU/GPU) |
| Disk | 711 GB free of 926 GB |
| OS | macOS, Darwin 25.4.0 arm64 |
| CUDA / nvidia-smi | not present |
| System Python | 3.9.6 (unused) |
| Project Python | 3.12.15 (uv-managed), `.venv` 1.2 GB with `ml` extra |
| Tools | uv 0.13.6 (installed via Homebrew this session), ffmpeg 9.0.2, git; no sox/conda |
| torch | 2.14.1, MPS available; 20× fp32 2048² matmul = 0.39 s (sanity check, not a benchmark) |
| transformers / peft | 5.21.0 / 0.21.2 |
| Network | huggingface.co **timeout**; hf-mirror.com OK; PyPI OK |
| HF cache before work | ~128 MB: partial mlx-community whisper-large-v3(-turbo), faster-whisper-large-v3-turbo, OPPOer/HearInContext dataset — pre-existing, not used |
| Downloaded this session | whisper-small tokenizer/config files only (4.2 MB, commit 973afd2), via hf-mirror.com |
| Repo state | was not a git repo; `git init` done, nothing committed |

Implications
- Transformers 5.x: `from_pretrained(dtype=...)` (not `torch_dtype`); Whisper `generate()`
  returns only generated tokens (forced language/task prefix is stripped) — tests account for this.
- MPS lacks some ops / bitwise determinism guarantees; record versions per run; CPU fallback
  available via `runtime.device: cpu`.
- Training speed on M5 is **unmeasured**. Phase 2 starts with a timed smoke run.
