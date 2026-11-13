# Research notes: why lora_v2 / lora_v3 did not move benchmark v1 (2026-10-31)

Context: lora_v2 / lora_v3 gave ASCEND test 29.0% → 12.6%, but benchmark v1 tied the baseline (11.36% → 11.38% / 11.64%).
Sources below were read through search summaries and abstracts, not full papers. Check numbers in the originals
before citing them.

## What the literature says, mapped to our runs

| Finding | Source | Our runs |
|---|---|---|
| Gains in one domain rarely transfer. Whisper fine-tuned on LibriSpeech degrades faster elsewhere | [Whisper paper](https://cdn.openai.com/papers/whisper.pdf) | 66% of train hours are ASCEND (8.16 / 12.37 h): ASCEND test improves, users' benchmark doesn't |
| Single-source fine-tuning hurts OOD (5.8% in-domain vs 11.1% other channel); more diverse data fixes it | [RO-N3WS](https://arxiv.org/pdf/2603.02368) | users'-style zh/zh-en is only ~1.6 h |
| Pure-ASCEND LoRA on whisper-small: 48.7% → 15.9% (in-domain) | [whisper-cs-lora](https://github.com/elrond298/whisper-cs-lora) | same pattern as our ASCEND gain |
| zh-en code-switch adaptation saturates after ~1–10 h of in-domain data | [Adapting Whisper for Code-Switch](https://arxiv.org/pdf/2311.17382) | more ASCEND-like data won't help; target-domain data might |
| Small-data fine-tuning causes forgetting (Yoruba: English WER 21.8% → 147%); mixing general data back fixes it | [Medium write-up](https://medium.com/@ccibeekeoc42/advancing-multilingual-speech-recognition-fine-tuning-whisper-for-enhanced-low-resource-34529b525f90), [Averaging domain experts](https://arxiv.org/pdf/2305.09681) | English drifts 9.87 → 10.31 → 10.63% (n.s.) |
| Adapters / LoRA forget less than full fine-tuning | [Whisper fine-tuning strategies, EURASIP 2024](https://link.springer.com/article/10.1186/s13636-024-00349-3) | we already use LoRA |
| small/medium + ~60 h target data beat zero-shot large-v2; with < ~20 h, large models overfit fast | [Greek Podcast Corpus](https://arxiv.org/pdf/2406.15284), [ICON 2024](https://aclanthology.org/2024.icon-1.47.pdf) | zero-shot large-v3 may already beat fine-tuned small; fine-tuning large-v3 on ~12 h needs LoRA + early stopping |
| Fine-tuned models hallucinate on very short clips | [Code-switching under the lens](https://arxiv.org/pdf/2509.24310) | lora_v2 step 1200 / lora_v3 step ≥ 1400: "对对对…" loops on 1–2 s ASCEND clips |

Other reasons specific to us: a good starting point (base 11.36%, so little headroom), mixed reference styles in
benchmark v1 (mostly cleaned, 33/271 English refs verbatim), and a zh-en test slice too small (63 clips, 34 speakers)
to detect < ~1.5-point changes.

## Directions, in order

1. **Target-domain data** (users' dictation, hard scenarios): plan in docs/data_collection_plan.md.
2. **Bigger base model**: zero-shot medium / large-v3 / large-v3-turbo on val (running 2026-10-31).
3. **Data mix**: down-weight ASCEND or over-sample users' data.
4. **Anti-forgetting**: scale the LoRA delta (interpolate base ↔ fine-tuned, near-free to test on val), replay data.
5. **Bigger locked zh-en test set** (≥ 300 clips, ≥ 100 speakers).
