# Post-processing model: distil the production Gemini cleanup into a small, fast model (plan, 2026-11-04)

Scope: phase 1 = dictation only; edit mode is phase 2 (owner, 2026-11-05).
Goal (owner): replace the third-party post-processing call (production: gemini-3.5-flash-lite + a format classifier,
via API) with our own fine-tuned small LLM, served on our GPUs with an optimised engine, so that ASR + post-processing
stays at ≈ 0.7 s end to end on average (Wispr Flow budget: ASR < 200 ms, LLM < 200 ms, network < 200 ms at p99).
Division of labour (decisions 2026-11-04): ASR transcribes faithfully (letters as "V-E-R-C-E-L", self-corrections
verbatim); post-processing removes fillers, applies self-corrections, assembles spelled words, formats.

## What distillation means here (learning notes)
- **Teacher** = the current production post-processor (Gemini + our prompts). **Student** = a small open model
  (candidate: Qwen3 1.7B / 4B — strong Chinese, permissive licence) fine-tuned to produce what the teacher produces.
- **Sequence-level distillation**: we do not need the teacher's probabilities, only its outputs. Collect (input →
  teacher output) pairs, filter them, and fine-tune the student on them with ordinary supervised fine-tuning (SFT).
  This is the standard way to compress an API model's behaviour into a model we can host.
- **Why it can work for this task**: cleanup is narrow (mostly copy the input, delete fillers, fix punctuation, apply a
  correction). A 2–4B model can learn it; it does not need the teacher's general knowledge.
- **Beyond the teacher**: after SFT, use human corrections (owner reviews, production edits / correction_learning) to
  fix cases where the teacher is wrong (DPO or a second SFT round on corrected outputs). The student can then beat the
  teacher on our own failure cases.

## Data (already available)
- PostHog `asr_metrics`: `transcription` (ASR raw) and `processed` (teacher output) for every request, plus mode,
  languages, app, prompt name/label — ≈ 200k dictate requests per month (scanned 2026-10-05 → 11-04: 258,651).
- Langfuse `gen_ai.postprocess` generations: the exact teacher INPUT (system prompt, dictionary, screen context,
  prompt version) — needed so the student sees the same information as the teacher.
- Quality filters we already have (trace-lookup online.py flags): drop pp_expand / chatty / lang_mismatch / pp_dropped
  (teacher answered or translated instead of cleaning), error_*; keep pp_shrink only after a check (often correct
  filler removal). De-duplicate, cap per user, balance languages / apps / prompt types.
- Held-out evaluation: a locked set of (raw → human-approved clean text) for zh / zh-en / en, including the hard
  cases from the trial: self-correction, spelled letters → word, entities, long dictation.

## Steps
1. **Data pull + audit**: 30 days of (Langfuse input, teacher output) pairs; flag statistics; per-prompt counts.
2. **Eval set**: ~500 pairs reviewed by the owner (clean text as it should be), split by phenomenon.
3. **Student SFT**: Qwen3 1.7B and 4B (LoRA or full), same prompt format as production; metrics: edit distance to the
   human reference, phenomenon pass rates (filler removed, correction applied, spelling assembled, nothing invented),
   hallucination / answering rate.
4. **Serving**: vLLM or TensorRT-LLM, FP8 / INT8, **speculative decoding with the input as the draft** (output mostly
   copies the input, so several tokens are accepted per step), on the same GPU as the ASR. Measure p50 / p95 / p99 per
   input length and under concurrency.
5. **End-to-end prototype**: ASR (faster-whisper or TensorRT Whisper, possibly a LoRA on large-v3-turbo) + student,
   one GPU, measured from "stop speaking" to final text.

## Open questions / risks
- **Terms of use / consent — decided (owner, 2026-11-05):** Gemini outputs may be used for training, and users'
  production text is authorised for training (docs/decisions.md).
- Personalisation (tone, per-user style) and context (dictionary, screen text) must be part of the student's input from
  the start, or the student will not be able to match the teacher on those requests.

## Production prompts (Langfuse, pulled 2026-11-05; copies in reports/postprocess_prompts/)
- Base cleanup prompts (production label): transcription-fully-cleanup v32 (≈ 5.5k tokens), -chinese v8 (≈ 1.5k),
  email_list_cleanup(_chinese) (≈ 1.7–2k), transcription-gpt4o-cleanup v24, transcription-input-box-cleanup v4,
  transcription-light-cleanup v1, dictate-formatter-zh v6; blocks for email / list format, vocabulary, mapping,
  application / recent-conversation context.
- **Tone** = three presets, each a short block appended to the base prompt: tone-casual, tone-formal, tone-very-casual
  (≈ 250 tokens each), plus a **free-text user preference** passed as JSON (POSTPROCESS_TONE_INSTRUCTION_DATA, governed by
  [block]cleanup-tone-instruction: style only, never content).
- Consequence for the student: the teacher reads up to ≈ 6k prompt tokens per request. The student should get compact
  control fields instead (mode, format, tone preset or the user's free-text preference, app, language, vocabulary,
  short screen context) and learn the rules from data — this is what keeps prefill small and latency low.

## Model choice (2026-11-05)
| family | what it can do here | latency (one request) | verdict |
|---|---|---|---|
| Whisper (ASR) trained to emit clean text | fillers / punctuation in one pass, no extra model | 0 extra | no: cannot take tone / format / context per request, mixes ASR and style targets (against the layering decision) |
| BERT-style encoder tagger (e.g. XLM-R / Chinese RoBERTa, GECToR-style keep / delete / insert-punct / small replace tags) | filler and repetition removal, punctuation, deleting the overridden span of a self-correction, deleting spelled letters next to the word, newlines | ≈ 5–20 ms, even on CPU | candidate for the plain-cleanup segment only; cannot rewrite tone, assemble an unseen word from letters, or format e-mails freely |
| seq2seq (mT5 / ByT5 small–base) | general rewriting | ≈ 50–150 ms | weaker than current small LLMs on Chinese + code-switch; no advantage |
| **Decoder LLM: Qwen3 0.6B / 1.7B / 4B** (Apache-2.0, strong zh + en) | everything the teacher does, with control fields | 1.7B–4B ≈ 0.1–0.25 s on H100 with vLLM / TensorRT-LLM + speculative decoding (input as draft) | **main candidate** |
| Llama 3.2 1B / 3B, Gemma 3 1B / 4B, Phi-4-mini | same | similar | fallbacks; weaker Chinese (Llama, Phi) or licence terms (Gemma) |
Plan: SFT Qwen3-1.7B and -4B (0.6B as a speed floor) on the largest segment first; in parallel, a quick tagger baseline
on the plain-cleanup segment to measure how much of the traffic a ≈ 10 ms model already handles (possible hybrid:
tagger for plain cleanup, LLM for format / tone / edit).

## Data audit results (2026-11-05, reports/postprocess_data_audit/report.md)
- Only 62.5% of dictate requests reach the post-processing LLM; 28.3% are cleaned by the ASR itself through its prompt
  (transcription-for-multimodality); 9.2% cancel / no output. Edit mode (phase 2) goes audio + selection straight to
  gemini-3.5-flash, so a text-only student cannot replace it.
- Teacher input = fixed system prompt (fully-cleanup v32 ≈ 4.4k tokens) + <TRANSCRIPT>; no dictionary / screen context /
  preset tone reaches it (custom tone adds ≈ 1.4k characters, also in PostHog tone_prompt). PostHog transcription →
  processed equals the real teacher input → output (16/16), so the scans are usable training pairs as they are.
- Post-processed traffic: en 63%, pt (mostly pt-BR) 32%, zh / zh-en 4.3%; list / email prompt 3.1%; custom tone 1.8%.
  en + pt-BR with the standard prompt = 89%. Android 85%.
- Teacher operations: 66.5% identical output, 10% punctuation / case only, ≈ 7% duplicate / spoken-punctuation removal,
  7% small word edits, 3.6% fillers, 1.4% self-correction, < 1% list formatting. Bad flags < 0.4% overall, concentrated in
  custom tone (gpt-5.4-nano: 17% expansions / answering).
- No real production user-correction signal (correction_learning_* is test / replay); proxy = users editing their own
  dictation within 30 min: 2,810 / 30 days, ≈ 370 explicit term / value fixes → evals and hard cases, not training.
- Usable after filters: 46.7k pairs / 29 days (35k with changes); ≈ 79k / month at the current growth.
- Open: prompt version per request before ≈ 10-28 is unknown; where preset tones take effect is unknown (they are in
  neither the post-processing nor the ASR prompt — check with the backend).
