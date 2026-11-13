# Project WhisperFlow — Chinese-English Code-switch ASR, Fine-tuning & Dictation

> **Working contract for Claude Code.** This document is the project specification, execution plan, and acceptance checklist. Begin with **Phase 0 and Phase 1 only**. Inspect the environment and repo first; do not assume GPU, datasets, credentials, or files exist. Do not trigger paid compute, download large checkpoints/datasets, access private audio, or run training until the owner explicitly approves the proposed cost/data scope. Do not claim measured improvements without reproducible runs.

## 1. Objective

Build an independently trained (fine-tuned) and deployed speech-to-text dictation system optimized for **Mandarin Chinese, Mandarin-English code-switching, technical terms/hotwords, spontaneous corrections, punctuation, and context-sensitive formatting**. Use a pre-trained multilingual Whisper checkpoint and **LoRA**, not training from random initialization. Ultimately compare against Wispr Flow on a controlled, consented, same-audio evaluation and report where this system wins, ties, or loses. **Do not promise a win**: the challenger is an experimental system, and Wispr Flow is a full product, not simply an ASR checkpoint.

Initial priority: **training data and measurement**, not UI. Build a trustworthy dataset, freeze a test set, measure the unfine-tuned baseline, then run LoRA. Only after evidence of ASR benefit proceed to decoding, inference, and product work.

### In scope
- Authorized/public-licensed Mandarin and code-switch speech data ingestion, cleaning, stratification, transcript quality review, and train/val/test manifests.
- Whisper-small multilingual starting baseline; LoRA supervised fine-tuning, reproducible logs/checkpoints, diagnostics.
- Contextual prompting/hotword experiments; optional custom logit-/beam-level biasing with an explicit implementation path.
- CTranslate2/faster-whisper serving, quantization, VAD, latency benchmarking; macOS/client dictation prototype.
- Separate **verbatim ASR** from **formatting/post-processing** with semantic fidelity checks.

### Out of scope for MVP
- Training Whisper from scratch, proprietary Wispr Flow internals, production-scale streaming guarantees, guaranteed superiority over Wispr Flow, or using company-owned recordings without explicit authorization.

## 2. Success metrics and experimental discipline

Track performance by slice, not one aggregate score:

| Dimension | Metric | Measurement |
|---|---|---|
| Verbatim Mandarin | CER and MER | Reference normalization policy fixed **before** evaluating |
| Mixed-language | MER; English word accuracy; error by switch count | Freeze Chinese/English tokenization conventions and normalization |
| Hotwords | Term recall, precision, false insertion per hour; no-hotword negatives | Dictionary-versioned, exact and alias-aware matching |
| Verbatim stability | CER/MER on generic Mandarin + unseen speakers | Compare against frozen baseline; flag regressions |
| Punctuation | Punctuation precision/recall/F1 | Separate from verbatim recognition |
| Dictation text | Human blind-rated readability and semantic preservation | Original meaning, numbers, negation, names, actions must be preserved |
| Runtime | ASR RTF, stop-to-text p50/p95, peak VRAM, throughput | Specify device, precision, concurrency, clip lengths, network |
| End-to-end | Successful insertion rate, crash/timeouts | Mac client and service tests |

**Comparability rules:**
1. Split by **speaker and/or recording session**, deduplicate near-duplicates; all augmented variants remain in the same split.
2. Never train, manually tune, or choose checkpoints using the final locked test set. Use validation for iteration; use test only at predeclared milestones.
3. Distinguish raw/verbatim recognition from cleaned dictation; a formatter must not be judged solely by strict WER/CER against verbatim transcripts.
4. Report N, hours, number of speakers, slice composition, bootstrap CIs where feasible, decoding parameters, random seeds, model SHA, tokenizer, hardware, and dependency versions.
5. Compare Wispr Flow, base Whisper, fine-tuned Whisper, and full pipeline on the **same authorized audio**, timing from the same user-visible endpoint. Wispr Flow behavior, availability, settings, and version must be verified at experiment time. Do not assume access to Wispr internal ASR, raw transcripts, or model architecture.
6. Mark any quality numbers below as **aspirational planning targets only**, not predictions or observed results. Example: if baseline MER were 10%, test whether 8.5% (LoRA), 7% (with hotwords), 6% (full system) are feasible; do not present them as forecasts.

## 3. Dataset design

### 3.1 Initial pilot: 15 hours total
- **10h training**, 2h validation, 3h **locked test**. Prefer 20–40 diverse consenting speakers for the longer program, but do not claim this is a statistical minimum.
- Training composition, **mutually exclusive primary labels** (other metadata labels can overlap):
  - 2.5h ordinary Mandarin
  - 3.5h Mandarin-English code-switch
  - 2h technical terms and names
  - 1h spontaneous disfluencies/self-corrections
  - 1h dictation/formatting-related speech
- Build test slices including: pure zh, zh-en mixed, hotword-present, hotword-absent, near-homophones, unseen speakers, noise, self-correction, dictation instructions. Make minimum slice counts visible; if sample size is insufficient, state it.
- Iterate toward **50h verified training** only after pilot shows promising, reproducible improvement. Proposed 50h mix: 17.5h code-switch, 12.5h Mandarin, 10h terms, 5h self-correction, 5h formatting-related; plus separately sourced validation and locked test.
- Long-term possibility: 200h+ if collecting/labeling capacity and empirical learning curves justify it; not an MVP requirement.

### 3.2 Data sourcing and permissions
- Start with openly licensed corpora that permit the intended use; record source URL, license, restrictions, permitted redistribution, provenance, and speaker consent where applicable. Check licenses individually.
- Owner-recorded or volunteer recordings are acceptable with consent and documented scope.
- **Do not use recordings, user edits, logs, corpora, or vocabulary from an employer unless permission is explicit and documented.** No copying sensitive/customer content into third-party LLMs or cloud tools.
- Synthetic text can guide volunteer recordings; synthetic speech is a separate domain and cannot substitute for realistic spontaneous speech without an explicit domain-gap analysis.

### 3.3 Manifest schema (one row per utterance)
Use JSONL or Parquet, with consistent IDs and integrity hashes. Minimum fields:

```json
{
  "utterance_id": "u000001",
  "audio_path": "data/raw/example.wav",
  "sha256": "<sha256>",
  "duration_sec": 8.2,
  "speaker_id": "s001",
  "session_id": "session_001",
  "source_id": "volunteer_v1",
  "license_or_consent_id": "consent_001",
  "split": "train",
  "raw_transcript": "我们先跑一下 benchmark 然后看 CER",
  "formatted_transcript": "我们先跑一下 Benchmark，然后看 CER。",
  "primary_scenario": "zh_en_switch",
  "language_tags": ["zh", "en"],
  "hotwords_present": ["Benchmark", "CER"],
  "recording_device": "phone",
  "noise_tag": "quiet",
  "label_status": "human_verified",
  "reviewer_id": "reviewer_01",
  "quality_flags": []
}
```

The snippet is illustrative, not sample training data. No path or hash is to be fabricated in production.

### 3.4 Annotation rules
- `raw_transcript`: accurate verbatim content; keep meaningful code-switch, numbers, names, negations, repetitions/self-corrections according to a written convention. Avoid inventing punctuation/semantic edits just to make text elegant.
- `formatted_transcript`: separately labeled **desired dictation output**; may add punctuation, structure, capitalization, bullets, remove fillers, but must preserve facts and user intent. Define what constitutes an allowed correction versus paraphrase or hallucination.
- Normalize Unicode, casing, English word boundaries, punctuation, numerals **only by versioned policy**; preserve original labels and transformed evaluation copies separately.
- Review sample-level audio-text alignment, clipping/silence, duplicates, overlap, speech quality and PII. Maintain review status; no automatic pseudo-label enters the trusted set without verification.
- Do not score formatting by raw ASR MER alone.

## 4. Engineering architecture

```text
Authorized audio sources / local recorder
      ↓
Dataset ingestion → validation → dedup → review → manifest + splits
      ↓
Baseline multilingual Whisper-small (fixed decoding)
      ↓
LoRA SFT (Transformers + PEFT / Seq2SeqTrainer, if compatible)
      ↓
Model checkpoints + evaluator → compare on val → freeze candidate
      ↓
Hotword pathway A: initial_prompt/context
Hotword pathway B: custom token/beam bias in PyTorch (research)
      ↓
Serving pathway: convert compatible checkpoint to CTranslate2
   + faster-whisper + VAD + precision tests
      ↓
FastAPI service / streaming-like chunk interface (when supported)
      ↓
Separate text formatter + semantic fidelity checks
      ↓
Mac dictation client (push-to-talk → insert into active app)
      ↓
Frozen A/B benchmark vs baseline / Wispr Flow
```

**Important:** CTranslate2 model conversion transfers supported weights, **not custom Python decoding logic**. Do not assume PyTorch logit modifications survive conversion. Investigate whether the chosen inference backend can implement the algorithm; otherwise evaluate a supported hotword prompting approach separately or maintain a research-mode PyTorch backend.

## 5. Phases, acceptance criteria, and gates

### Phase 0 — Repo/environment inspection and decisions (first task)
**Tasks**
- Inspect existing repo, OS, Python/CUDA/GPU/MPS, VRAM, available storage, runtime budget, data sources and licenses. Report unknowns, never assume.
- Recommend an explicit reproducible dependency set using current compatible versions of PyTorch, Transformers, datasets, accelerate, peft, evaluate, jiwer, soundfile, faster-whisper, CTranslate2, FastAPI. Lock versions once validated.
- Choose Whisper-small multilingual as baseline (or propose a documented downgrade due to hardware). Document why. Define audio input, max clip length, batching, precision and seed.
- Draft normalization policy, metadata schema, benchmark runner design, and proposed data sourcing list with license review.

**Deliver:** `docs/decisions.md`, `docs/dataset_spec.md`, environment report, `requirements` or lock file draft, `README.md` with exact reproduction commands.

**Gate:** **Do not download large data or checkpoints, launch paid GPU, or train without owner approval.**

### Phase 1 — Data pipeline + frozen baseline (pilot priority)
**Tasks**
- Implement `src/data/ingest.py`, `validate.py`, `dedup.py`, `split.py`, `audit.py` (names adaptable to repo); CLI and unit tests for manifest, broken audio, labels, duplicates, leakage, consent/license status.
- Ingest a **small authorized sample first**, e.g. 20–100 clips, generate dataset report with clip durations, speaker counts, domain/language slices, quality failures and split stats.
- Prepare 10h train / 2h val / 3h test pilot **only after authorization and data availability**. If incomplete, report actual hours and do not fill gaps with fake data.
- Implement baseline transcript/evaluation runner for Whisper-small multilingual, store per-utterance outputs and detailed error cases. Fix one normalization/tokenization policy; report CER and mixed-language MER along with domain slices.
- Create `data/README.md`, manifests, reproducibility/configs; never commit raw sensitive recordings or secrets.

**Deliver:** validated manifest, train/val/test split audit, baseline metrics report, error-case dashboard/CSV.

**Acceptance:** all manifest records traceable; no split leakage under specified checks; evaluation re-runnable on the same checkpoint and data; baseline numbers measured, not estimated.

**Gate:** owner signs off dataset and evaluation policy before training.

### Phase 2 — Whisper-small LoRA fine-tuning
**Tasks**
- Implement LoRA seq2seq fine-tuning (begin with a short smoke test). Specify adapter target modules by introspecting the actual Whisper model rather than assuming Qwen-style names.
- Verify trainable parameter counts, train/eval loss, optimizer/scheduler, gradient accumulation, masking/padding (-100 label handling), mixed precision, checkpoint save/resume, seeds, gradient clipping.
- Run a small sanity overfit test, then fixed-budget LoRA baseline; compare with original Whisper under identical decoding and held-out metrics.
- Evaluate generic-Mandarin regression, code-switch, terms, unseen speakers and errors. Preserve experiments in `reports/experiments.csv` with full settings.

**Deliver:** adapter and optional merged model, scripts, training curves, val results, model card, comparison report.

**Acceptance:** repeatable training and measurable comparative evaluation; no claims of improvement unless demonstrated beyond expected variance.

**Gate:** decide whether LoRA yields enough benefit to justify expanding to 50h or revising the dataset.

### Phase 3 — Hotword contextual biasing
**Baselines:** unmodified model, `initial_prompt` and supported backend hotword prompting (verify exact semantics).

**Advanced experiment:** explore custom token-logit/beam path scoring with multi-token hotword prefix matching, varying bias strength, dictionary size and placement. Understand Whisper tokenization and generation constraints before editing. Include hotword-absent and phonetically confusable negatives.

**Deliver:** term recall/precision vs MER/false insertion/latency plots and ablations; reproducible decoder code and tests.

**Acceptance:** no uncontrolled hallucination or unacceptable generic-domain regression at chosen operating point. Otherwise retain the simpler prompting baseline.

### Phase 4 — Inference optimization and serving
- Convert a compatible fine-tuned model/checkpoint to CTranslate2, documenting exact export/merge steps and parity checks; test fp16/int8 as hardware permits.
- Benchmark Whisper/PyTorch vs CTranslate2/faster-whisper on **identical clips, beam parameters when feasible, device, concurrency, and decoder features**. Document non-identical decoding features.
- Add VAD, queuing, batching where appropriate and FastAPI `/health`, `/transcribe` endpoints; log request IDs, durations, latency breakdown and errors without saving raw speech by default.
- Define stop-to-text wall-clock latency, distinct from RTF; calculate p50/p95 and resource use.

**Deliver:** Docker/service scripts, model deployment guide, benchmark report and regression tests.

**Acceptance:** deployable on documented hardware with tested requests, reasonable failure handling, and verified accuracy parity/regressions.

### Phase 5 — Formatting and dictation client
- Build a separate formatter using rules or a chosen LLM. Train/test on `raw_transcript → formatted_transcript` examples; evaluate punctuation, list formatting, filler removal and meaning preservation.
- Preserve numbers, negation, proper nouns, technical content and user commands. Record changes; allow users to choose raw vs formatted text.
- Build minimal macOS push-to-talk client with permission-aware microphone capture, explicit stop, local/service invocation and active-app text insertion. Do not assume privileged accessibility APIs are already granted.
- Protect recordings by default (short retention/no cloud upload without user choice). Report latency at every stage.

**Deliver:** installable developer prototype, end-to-end timing metrics, failure handling and usability test notes.

### Phase 6 — Wispr Flow comparative evaluation
- Verify current official features and publicly observable product behavior at benchmark time; capture product version/settings/hardware/network.
- Prepare a consented **same-audio**, controlled input benchmark with blinded output quality rating. If identical audio injection isn't supported by the product, disclose and document any procedural differences; don't claim direct model-only comparisons.
- Separate: verbatim transcription, code-switch and terms, formatting semantic fidelity, end-to-end stop-to-text latency, insertion reliability, user perceived quality.
- Break out per-slice results and uncertainty, and identify cases where each system wins or fails.

**Deliver:** transparent comparative report with reproducible challenger metrics and clearly documented limitations of black-box product measurements.

## 6. Repository layout (proposed, adapt to existing code)

```text
.
├── plan.md
├── README.md
├── pyproject.toml / requirements.lock
├── configs/
│   ├── dataset.yaml
│   ├── eval.yaml
│   ├── train_lora.yaml
│   └── serve.yaml
├── data/
│   ├── README.md
│   ├── manifests/          # non-sensitive metadata only
│   └── private/            # gitignored; authorized recordings
├── docs/
│   ├── decisions.md
│   ├── dataset_spec.md
│   └── annotation_policy.md
├── src/
│   ├── data/
│   ├── eval/
│   ├── training/
│   ├── decoding/
│   ├── serving/
│   └── formatting/
├── clients/macos/
├── tests/
├── reports/
└── scripts/
```

## 7. Estimated effort & constraints (planning estimates, not guarantees)

| Milestone | Approximate calendar time at 12–16 personal hours/week |
|---|---|
| Dataset pilot + baseline | Weeks 1–2, conditional on data availability |
| Expand authorized dataset toward 50h | Weeks 3–4+; labeling may dominate |
| LoRA training + contextual bias | Weeks 3–6 (overlap only when justified) |
| Serving + dictation demo + comparison | Weeks 6–8+ |

- **Data acquisition and annotation is likely the bottleneck**. If manually verifying all 50 hours, allow roughly **100–250 reviewer-hours** as a rough planning range, depending on source/quality. The owner should approve smaller staged collection.
- On suitable NVIDIA GPU(s), a 10h pilot may take several hours of training, but speed depends strongly on GPU, model, batch, seq length, precision, and implementation. **Benchmark on actual hardware before estimating cost**.
- Track cloud spend, storage and egress; prefer a <$small agreed pilot budget, never silently allocate resources.
- Launch a smaller, demonstrable project rather than promising to surpass a large commercial product globally.

## 8. Decision log & open questions for owner

Only ask what is necessary to proceed; inspect the workspace first:
1. Hardware and budget: local Mac/MPS vs accessible NVIDIA GPU; VRAM and paid compute limit.
2. Authorized data sources and any existing transcribed Mandarin/code-switch clips.
3. Licensing/sharing intention: private demo vs public GitHub or model release.
4. Target deployment: personal Mac dictation, self-hosted GPU, or remote endpoint.
5. Which precise style conventions and hotwords are in scope for v1.

Record each decision in `docs/decisions.md`. If missing, use conservative reversible defaults **only for documentation and code scaffolding**, not cost-incurring or private-data actions.

## 9. Instructions to Claude Code — immediate next turn

You are the engineer implementing this plan, not merely writing prose. Start with:

1. `pwd`, inspect repository tree, git status and current files, Python version and available GPU/MPS (`nvidia-smi` if present). Do not overwrite existing work.
2. Summarize what exists vs what's missing; propose minimal stack compatible with this environment.
3. Create/refine `README.md`, `docs/dataset_spec.md`, `docs/annotation_policy.md`, `docs/decisions.md`, `configs/` templates and a tiny testable data validation scaffold with **synthetic manifest records only**. No fabricated audio, results, or metrics.
4. Run available tests/linters, report real outputs; fix failures where in scope.
5. Stop at the Phase 0 gate with a concise list of decisions needed from the owner. Do **not** start GPU training, large downloads or paid services yet.

After owner approval, implement **Phase 1 before Phase 2**. At every phase, show actual changes, reproducible commands, test results, measured metrics, and next gate. Prefer small working increments. Never substitute inferred success for actual evidence.
