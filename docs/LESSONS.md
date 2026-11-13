# What we learned (VoiceCursor dictation, Oct–Nov 2026)

Reference results from our own runs, to set expectations. Data: Chinese / English / code-switched dictation;
Qwen3-ASR-1.7B fine-tuned with this framework's recipe on one H100 (≈ 20 min per run). MER, lower is better.
The datasets themselves are internal and not part of this repository.

| run | training data | users' val | dictionary test (hotword recall) | dictation benchmark (412 clips) |
| --- | --- | ---: | ---: | ---: |
| Qwen3-ASR-1.7B zero-shot | – | 9.09% | 11.38% (75%) | 9.72% |
| J1 | 7.7 h real dictation, verbatim labels, real logged context | 7.47% | 11.23% (64%) | 9.62% |
| J2 | J1 + context augmentation | 6.85% | 13.07% (72%) | 9.65% |
| J3 | J2 (lighter vocabulary augmentation) + 1.2 h long real + 5.8 h synthetic TTS | 6.89% | 10.79% (80%) | 9.54% |
| J4 | J3 + acronym / long-English synthetic + concatenated long clips | **6.76%** | **10.12% (81%)** | **9.33%** |

All fine-tuned rows decoded with context. On the benchmark, J3 is clearly better than zero-shot on clips ≤ 30 s
(7.98% vs 9.94%) but not on clips > 30 s (10.67% vs 9.56%), which are mostly long English monologues.

## Lessons

1. **Labels matter more than tricks.** When self-corrections were labelled as the corrected sentence, the model
   learned to rewrite what people said. Keep the ASR target verbatim (remove only fillers like 嗯 / 呃 / um / uh) and
   do clean-up in a separate post-processing step.
2. **Context needs augmentation.** Trained on real logged context only, the model learned to ignore the vocabulary
   (it was rarely relevant) and sometimes copied the text before the cursor into its output. Mixing in
   distractor-only vocabularies, look-alike terms and mismatched cursor text fixed copying; too many English
   distractors made English output unstable, so the lighter mix (`augment.vocab` defaults) works better.
3. **Synthetic TTS data helps where it is targeted.** ~2,000 TTS clips written around names and terms gave the best
   hotword recall of all systems. It did nothing for problems it did not target. QC every TTS clip by decoding it
   with the zero-shot model; drop clips where the TTS clearly misread the text.
4. **More hours ≠ better.** Gains came from covering a failure mode (names, short sentences), not from volume.
   Long English monologues stayed weak even after adding synthetic and concatenated long clips.
5. **Decode long audio in ~25 s chunks.** One-pass decoding of multi-minute clips drifted; chunking at quiet points
   fixed it.
6. **Measure with intervals.** One seed per run means 1–2 point differences are often noise; use `qasr compare`.
   Check whether your references were derived from one system's output (they will favour that system).
