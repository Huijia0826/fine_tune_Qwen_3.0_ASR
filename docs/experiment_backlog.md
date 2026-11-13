# Experiment backlog

Ideas to test **after** the plain LoRA baseline (`runs/lora_v1`). Change one thing at a time, compare on val with
otherwise identical settings, and only then on benchmark v1.

## E1 — Uncertainty-weighted SMA loss (owner idea, 2026-10-29)

**Source:** Zhuang et al., "Towards Efficiently Whisper Fine-tuning with Monotonic Alignments", Interspeech 2025,
pp. 3598–3602, doi:10.21437/Interspeech.2025-179 (https://www.isca-archive.org/interspeech_2025/zhuang25_interspeech.pdf).

**Method:** an extra training loss on the decoder cross-attention. The model architecture does not change and
inference costs nothing extra.
- For output token i and cross-attention head k: expected input position π_i = Σ_j α_ij · j (attention-weighted frame index).
- Δπ_i = π_i − π_{i−1}; L_SMA = ‖ |Δπ − 1| − (Δπ − 1) ‖₁, which is 0 when every token moves ≥ 1 frame forward.
- Uncertainty weighting over all M×N heads: L_W = d/(M·N) · Σ_k [ L_SMA_k / (2σ_k²) + ln(1 + σ_k²) ], with σ_k learnable (init 1.0).
- L_total = L_CE + λ·L_W. Paper: λ = 0.3, d = 2e-3, lr 1e-5, 3 epochs, greedy decoding.

**Reported (Whisper-large-v2, full fine-tune):** AISHELL-1 CER 3.08 → 2.41 (plain SMA 2.93); AISHELL-2 3.66 → 3.39;
LibriSpeech test-other 4.04 → 3.89. It hurt ja→en and zh→en *translation* (non-monotonic); our task is transcription.

**Open questions before implementing:**
- Our setup differs: whisper-small + LoRA, about 9 h, zh-en code-switch. Does the gain transfer? Only a run will tell.
- Exclude the 30 s zero-padding frames from π? Exclude the prefix tokens (<|zh|>, <|transcribe|>, <|notimestamps|>) and padded label positions?
- σ_k are extra trainable parameters outside LoRA: add them to the optimizer and save them (they're discarded after training).
- Needs attention weights (`output_attentions=True`, eager attention): measure the extra memory on M5 with the smoke test.
- Single paper, no confidence intervals: require a val gain beyond run-to-run noise before adopting.

**Learning value:** good exercise (π / Δπ / weighted loss in ~30 lines; makes cross-attention concrete).

## E2 — Checkpoint averaging

Average the LoRA adapters of the last few saved checkpoints (e.g. ApproBiVT, arXiv 2308.02870). Cheap; adapters are saved every 200 steps.

## E3 — Segment > 30 s benchmark_chinese clips (option B of 2026-10-29)

Would add about 0.5 h of in-domain user audio to training. Needs timestamp alignment of the reference text.

## E4 — LR schedule matched to the chosen length
lora_v2's best checkpoint (step 1000) was taken while the linear LR was still ~half of peak (the schedule targeted
1872 steps). Retrain with total steps ≈ 1000 so the LR anneals to 0 at the end. About 3 min on the cloud GPU.

## E5 — Temperature fallback for short clips too
Fallback currently runs only in long-form decoding. lora_v2 step 1200 looped on 1–2 s ASCEND clips ("对对对…"),
decoded in one pass with no check. Add a compression-ratio check and retry to the short path (or decode short clips
through the same fallback path).

## E6 — Remove immediate repetitions from ASCEND labels
ASCEND labels have 1.19% immediately repeated tokens (vs 0.22% in benchmark v1). The model's tendency to loop grows
with training. Collapse disfluent repeats ("我我我", "对对对") in training labels and compare.

## E7 — Faster large-v3 inference (owner, 2026-10-31)
Base model is large-v3 (RTF 0.091 on RTX PRO 6000, fp32). Options to measure on the same clips, with MER unchanged:
fp16/bf16, CTranslate2 / faster-whisper (int8 / int8_float16), batching, speculative decoding with turbo or
distil-large-v3 as the draft model, smaller beam. Report stop-to-text p50/p95 per plan.md §2.
