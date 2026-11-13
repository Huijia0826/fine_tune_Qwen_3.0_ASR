# Hands-on exercises

Core code blocks of the pipeline that are worth writing yourself. Each has a spec, the file to edit,
and tests that define "done". Run one exercise's tests with `uv run pytest -q <test file>`.
Concepts and debugging lessons from the exercises: `notes.md` (学习笔记).
Training code walkthrough for line-by-line re-implementation: `training_walkthrough.md`.

| # | Block | Status |
|---|---|---|
| 1 | ASCEND adapter: parquet → WAV + ingest TSV | ✅ done 2026-10-29 (5/5 tests) |
| 2 | Filler removal: `strip_fillers` (regex, whitespace) | written by Claude 2026-10-29 · re-implement later without looking |
| 3 | Audio standardization: resample to 16 kHz mono, sha256 (`src/whisperflow/data/audio.py`) | review existing code |
| 4 | Leakage check: identical audio / shared speakers across splits | when ingesting benchmark_chinese |
| 5 | Speaker-grouped split (union-find, `src/whisperflow/data/split.py`) | when building val |
| 6 | MER: edit-distance alignment + normalization (`src/whisperflow/eval/metrics.py`) | review anytime |
| 7 | ⭐ Labels + collator: `make_labels`, `WhisperCollator`, `-100` padding | written by Claude 2026-10-29 · rewrite from `training_walkthrough.md` §2–3 |
| 8 | ⭐ LoRA setup + training loop (grad accumulation, clipping, scheduler) | written by Claude 2026-10-29 · rewrite from `training_walkthrough.md` §4–5 |
| 9 | Weighted SMA loss (π, Δπ, uncertainty weights) on cross-attention | backlog: `docs/experiment_backlog.md` E1 |
| 10 | ⭐ Cleaning diagnostic: `classify_span` (filler / repeat / grammar / other) | ✅ done 2026-10-31 (32/32 tests) · steps 1–2 by owner, step 3 `is_same_word` by Claude · re-implement step 3 later |

---

## Exercise 1 — ASCEND adapter

**File:** `src/whisperflow/data/adapters/ascend.py` · **Tests:** `tests/test_ascend_adapter.py`

**Goal:** translate ASCEND's format into the generic `wf-ingest` input (a TSV + a folder of audio).
Hashing, IDs, 16 kHz checks happen in `wf-ingest`, not here. An adapter only translates the format.

**What one real ASCEND row looks like:**
```
id                  '00000'            ← restarts at 00000 in every split → not unique
path                '/storage/hf-datasets-cache/.../ses4_spk20_L1593_13.1900_0.8200.wav'  ← publisher's machine
audio               {'bytes': b'RIFF....WAVE...', 'path': 'ses4_spk20_L1593_13.1900_0.8200.wav'}
                       ↑ a complete 16 kHz WAV file   ↑ file name, unique across all 12,314 rows
transcription       '嗯'
duration            0.82
language            'zh' | 'en' | 'mixed'
original_speaker_id 20
session_id          4                  ← 1..4 for EVERY speaker → not a global conversation id
topic               'technology'
```

**Requirements:**
- skip rows whose transcription contains `[UNK]` (decision 2026-10-29: annotator could not hear that part)
- write the audio file unchanged
- TSV columns: `audio_path` (relative to `audio_out`), `transcript`, `speaker_id`, `session_id`, `source_utterance_id`
- `session_id` must not merge different speakers (later, the split step groups utterances by session)
- `source_utterance_id` unique across train/validation/test
- return `{"n_total", "n_skipped_unk", "n_written"}`

**Concepts:** reading parquet; audio stored as bytes; why IDs must be unique; why a wrong
session/speaker id silently breaks the train/test split later.

**Review notes (first draft, 2026-10-29):**
- `'[UNK]' in df['text'].values` checks whether any *whole row* equals `"[UNK]"`, for the whole column. You need a substring check on *this* row's string.
- `skip` from `unittest` is a test decorator; inside a loop use `continue`.
- the column is `transcription`, not `text`. Print `df.columns` / one row first.
- `audio` is a dict; the audio itself is in `["bytes"]` and is already a WAV file. Write the bytes as-is: no decode/encode, no quality loss, faster.
- `f"{i}.wav"`: `i` restarts per parquet file → shard 2 overwrites shard 1.
- one JSON per clip vs. the single TSV that `wf-ingest` reads; speaker_id is missing, but splitting and leakage checks depend on it.

**Result (2026-10-29):** all 5 tests pass. Bugs fixed on the way, in order: missing imports →
non-existent `utils` module → `.wav` outside the string → missing `audio_out.mkdir` → no `return` →
TSV written outside the `with` block → session_id trap → id-not-unique trap. Debugging lessons: `notes.md` §3–9.

---

## Exercise 2 — Filler removal

**File:** `src/whisperflow/text/fillers.py` · **Tests:** `tests/test_fillers.py` (14 cases)

**Goal:** `strip_fillers(text) -> str` removes 呃 and 嗯 from a transcript and tidies whitespace.
Applied to ASCEND training labels only; the audio stays unchanged. Clips that become empty are dropped by the caller.

**Why (decision 2026-10-29):** ASCEND writes hesitations verbatim (呃 128×, 嗯 81× in val); our references almost
never do (呃 0×, 嗯 1×). Train on them and the model learns to insert 呃/嗯 → insertion errors on our benchmark.

**Keep (traps the tests check):** 额 in 额度 · sentence-final 啊 · 哦 · English um/uh (benchmark v1 keeps them) ·
no spaces added between Chinese and English.

**Concepts:** regular expressions (character class `[...]`, `+` for runs, `\s`), `re.sub`, `str.strip`;
rules derived from the target data instead of from intuition.

**Hint:** two steps. (1) delete the filler characters; (2) fix the whitespace that is left over.
Try each step in `uv run python` on `"我觉得 呃 还可以"` and look at the result after each step.

---

## Exercise 10 — Cleaning diagnostic: `classify_span`

**File:** `src/whisperflow/eval/cleaning.py` · **Tests:** `tests/test_cleaning.py` (run one step: `-k step1`)

**Why:** the owner wants cleaned output (decision 2026-10-31), but we only have cleaned references, no verbatim ones.
Aligning a system's output to the cleaned reference shows where it did *not* clean. This exercise sorts those places
into categories, so we can measure what base / lora_v2 / VoiceCursor each learned to clean.

**Given (already written):** `extract_spans(ref, hyp)` aligns and groups each run of non-matching ops into a `Span`:

```
ref: 之类的我们      hyp: 然后之类的对我们
Span(kind="ins", ref=[], hyp=["然","后"], before=[],                after=["之","类","的","对","我","们"])
Span(kind="ins", ref=[], hyp=["对"],      before=["然","后","之","类","的"], after=["我","们"])
```

Chinese tokens are single characters, English tokens are lower-case words. `join_tokens(["然","后"])` → `"然后"`.

**Write `classify_span(span) -> str`**, rules in this order:

1. **filler**: `kind == "ins"` and the span text is made *entirely* of `FILLERS_ZH`/`FILLERS_EN` entries
   (`对对`, `就是那个`, `so um` are fillers; `是`, `就是狗` are not).
2. **repeat**: `kind == "ins"` and `span.hyp` equals the same number of tokens right before it or right after it.
3. **grammar**: `kind == "sub"`, exactly one word on each side, both English, and one is the other + `s`/`es`/`ed`/`ing`
   (either direction), or + `d` when the shorter ends in `e` (use → used), or the pair is in `IRREGULAR`.
4. **other**: everything else, including every `"del"`.

**Concepts:** alignment ops (eq/ins/del/sub); why we measure *what is left* instead of needing a verbatim reference;
rule order matters (`对` after `对` is both filler and repeat → filler); a calibration system (VoiceCursor) tells us
whether the rules measure the right thing.

**Progress log:**
- 2026-10-31: first draft had the right structure; repeat rule correct. Step 1: 9/13 (checked token by token, so
  multi-token fillers 然后 / 就是 / you know failed) → fixed with `is_all_fillers` (notes §16). Step 3: comparing
  `ref.lower() == hyp.lower()` never fires (tokens are already lower-case and equal tokens are `eq`, not `sub`) → use
  the suffix / `IRREGULAR` rule. Walkthroughs: notes §14 (reading pytest output), §15 (how `span.hyp` is built), §16.
- 2026-10-31: step 1 passes with `is_all_fillers` (written by owner from notes §16). Step 3 `is_same_word` written by
  Claude at owner's request (notes §17), 32/32 tests. Results: reports/cleaning_diagnostic.md.

## Exercise 11 — Latency summary: `summarize_latency` (E7, written by Claude at owner's request, 2026-11-02)

**File:** `src/whisperflow/eval/latency.py`. **Test:** `pytest tests/test_latency.py` (3 tests).

**Input:** `[{"duration_s": 6.0, "latency_s": 0.41}, ...]` (one row per clip, batch 1).
**Output:** `{"<5 s": {n, p50_s, p95_s, mean_s, rtf}, "5-15 s": …, "15-30 s": …, ">30 s": …, "all": …}`.

**Concepts to review:**
- Why p50/p95 and not the mean: one 382 s clip can dominate the mean; users feel the typical (p50) and the bad
  tail (p95) wait.
- RTF = total latency / total audio, not the mean of per-clip ratios (short clips have fixed overhead and would
  inflate a mean of ratios).
- p95 with n = 8 is basically the maximum: always read percentiles together with n.
- The timing loop in `main`: warm-up clips are untimed (CUDA init, kernel autotuning); `torch.cuda.synchronize()`
  before and after, because GPU calls return before the kernels finish; audio file loading is outside the timer.
- Bucket edges: exactly 30 s is still short-form (one encoder window), so `15-30 s` includes 30.0.

**Practice idea:** re-write `summarize_latency` from the docstring without looking, then run the tests.

## Exercises 12–15 — this week's plan (docs/plan_week_2026-11-09.md), proposed 2026-11-09, not started

Owner writes these first; Claude reviews and adds tests. Each one is small and sits on the critical path.

- **12 `build_sft_example(row) -> dict`** (plan P2). Input: one production pair (`transcription`, `processed`, mode,
  tone_id, app_type, cleanup_format, language). Output: `{"messages": [system with compact control fields, user =
  transcript, assistant = target]}`. Concepts: why control fields replace a 4.4k-token prompt (prefill latency), how
  the student learns rules from data instead of instructions, train / serve format must be identical.
- **13 `apply_casual_period_rule(text, tone) -> str`** (plan P4). Casual: drop only the sentence-final period
  (。 or .); keep mid-sentence ones, keep ？ ！ and ellipses; formal / default: unchanged. Tests: "好的。" → "好的",
  "第一句。第二句。" → "第一句。第二句", "OK?" unchanged, "3.5" unchanged. Concept: deterministic rules for targets the
  teacher does not follow.
- **14 vLLM latency client** (plan A1). Send N audio clips to an OpenAI-compatible endpoint at a given concurrency,
  record per-request latency, reuse `summarize_latency` (exercise 11). Concepts: batch-1 latency vs throughput under
  concurrency, warm-up, p95 / p99 with small n.
- **15 gateway pipeline handler** (plan C2). `POST /v1/audio/transcriptions` → ASR call → post-processing call →
  `{raw, processed, timings}`; pass mode / tone / app through; on post-processing failure return the raw text.
  Concepts: two-stage latency budget, graceful fallback, logging for error analysis.

## Exercise 16 — inclusion probabilities for a capped stratified sample (lid probe, 2026-11-09), not started

Reference: `scripts/lid_probe/sample.py` (`draw`, the Monte-Carlo loop in `main`) and `prevalence` in
`scripts/lid_probe/analyze.py`.

- **16 `inclusion_probs(cells, alloc, cap, runs) -> dict[(cell, user), float]`**. Re-run a "random order, at most
  `cap` clips per user" sampler `runs` times and return pi for one request = E[clips of user u in cell] / N_{u,cell}.
  Then write `hajek(y, w)` = sum(w*y) / sum(w) and a user-clustered bootstrap CI. Test: with cap = infinity the pi of
  every request must equal n_cell / N_cell; a heavy user (1,000 requests, cap 3) must get pi ≈ 3 / 1,000.
  Concepts: why "≤ 3 per user" makes the sample represent users rather than requests, how weights 1/pi turn it back
  into a request-level estimate, why the CI must resample users, not clips.
