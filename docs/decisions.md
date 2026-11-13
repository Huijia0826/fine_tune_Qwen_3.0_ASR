# Decision log

Status legend: **DECIDED** (implemented, reversible via config) · **PROPOSED** (default chosen for
scaffolding; needs owner sign-off) · **OPEN** (blocked on owner).

## D1 — Hardware target: Apple M5 / MPS for Phase 0–1 — DECIDED (for eval), OPEN (for training)
- Measured: Apple M5 (10 CPU / 10 GPU cores, Metal 4), 24 GB unified memory, 711 GB free disk,
  no CUDA. PyTorch 2.14.1 MPS backend available and functional (see `docs/environment.md`).
- Whisper-small (241.7M params, ~0.97 GB fp32) fits comfortably for inference and for LoRA
  training on 24 GB unified memory. **No training throughput has been measured yet**; any
  training-time estimate must come from a benchmark on this machine.
- Why no downgrade: whisper-base/tiny are much weaker on Mandarin; small is the smallest
  multilingual size that is a credible baseline, and the plan names it.
- Open: whether a CUDA GPU (local/cloud) is available for Phase 2, and the budget.

## D2 — Python 3.12 + uv lockfile — DECIDED
- System Python is 3.9.6 (too old for current transformers). uv-managed CPython 3.12.15.
- `uv.lock` pins all 119 packages transitively. Core deps (data pipeline) exclude torch so
  manifests can be validated anywhere; `ml` and `serve` are optional extras.
- Resolved versions: torch 2.14.1, transformers 5.21.0, peft 0.21.2, accelerate 1.15.0,
  datasets 5.1.0, librosa 1.0.0, pydantic 2.14.1, opencc 1.4.2, jiwer (cross-check only).
  `serve` extra (faster-whisper, ctranslate2, fastapi) not installed until Phase 4.

## D3 — Audio contract — DECIDED
- 16 kHz mono float32 at model input (Whisper feature extractor requirement). Canonical
  stored format after ingestion: 16 kHz mono PCM16 WAV (`wf-ingest --convert-16k`).
- Max clip length 30 s (Whisper encoder window). Longer clips are a validation warning and
  must be segmented before training; we do not silently truncate.
- Min 0.3 s (error), <0.5 s warning.

## D4 — Normalization policy `zh_en_v1` — PROPOSED (needs sign-off before first real eval)
Implemented in `src/whisperflow/text/normalize.py`; any change bumps the version.
1. NFKC → 2. Traditional→Simplified (OpenCC t2s) → 3. lowercase → 4. tokens = each CJK char +
   each ASCII alnum run (keeping intra-token `.` `'` and trailing `+` `#`: `node.js`, `c++`, `3.5`)
   → 5. drop all other punctuation/symbols.
- **Why t2s:** Whisper frequently outputs Traditional characters for Mandarin; that's a
  script choice, not a recognition error. We still *count* how often it happens
  (`n_hyp_traditional_script`) because a dictation product must output Simplified.
- **Why no numeral normalization in v1:** converting 二零二四↔2024 needs context
  (年份 vs 数量 vs 电话号码) and any rule embeds bias. Instead the `digits_in_ref_or_hyp`
  slice isolates the effect. Expect this slice to look bad for the baseline.
- **Why punctuation excluded:** plan.md requires punctuation be scored separately.

## D5 — Metric definitions — DECIDED
- MER over mixed tokens (above); equals CER on pure-zh and WER on pure-en.
- Corpus-level = Σerrors/Σref (micro-average).
- Language-attributed rates: S/D attributed to the reference token's language, insertions
  to the hypothesis token's language; alignment tie-breaks prefer same-language substitutions.
- 95% CI via **speaker-cluster bootstrap** (1000 resamples, seeded) — utterances from one
  speaker are correlated, so utterance-level bootstrap would be overconfident.
- Slices with N < 30 are flagged "insufficient N" (configurable).

## D6 — Baseline decoding — PROPOSED
- `openai/whisper-small`, forced `language=zh`, `task=transcribe`, beam 5, no sampling, no
  temperature fallback, no timestamps, `max_new_tokens=200`, fp32, unprompted.
- **Why force zh:** language ID on code-switched audio is unstable (English-heavy clips get
  `en` and then Chinese is translated/transliterated). Forcing zh is the standard approach
  for zh-en CS. Auto-detect can be an ablation.
- **Why no temperature fallback:** fallback makes output depend on sampling; we want a
  deterministic reference. Note: Whisper's paper numbers use fallback, so ours are not
  directly comparable to published figures.
- **Why fp32 on MPS:** removes precision as a variable for the reference baseline; fp16 is
  a separate runtime experiment. MPS kernels are not guaranteed bitwise-deterministic
  across torch versions — versions are recorded in every run.
- Pin `revision: 973afd24965f72e36ca33b3055d56a652f456b4d` (current HF main of
  openai/whisper-small, observed 2026-10-27) once the download is approved.

## D7 — Model download path — OPEN
- `huggingface.co` timed out (15 s) from this network on 2026-10-27; `hf-mirror.com`
  responds. The tokenizer/config files (~4.2 MB, no weights) were fetched via the mirror at
  the pinned commit for integration tests.
- Weights needed: `model.safetensors` 967.0 MB (LFS sha256 prefix `1d7734884874f1a1`, as
  reported by the mirror's API). A mirror is a third party; before trusting weights from
  it we should verify the full sha256 against a value obtained from huggingface.co (via
  another network/VPN) or download directly from huggingface.co.

## D8 — Split strategy — DECIDED
See `docs/dataset_spec.md §Splits`. Union-find grouping over speaker/session/lineage/audio
hash; stratified greedy by scenario hours; deterministic; test lock file.

## D9 — Package layout — DECIDED
`src/whisperflow/{data,eval,text,training,decoding,serving,formatting}` instead of
`src/data` etc. (plan §6 allows adaptation): a named package avoids import collisions with
`datasets`/`data` and gives installable console scripts (`wf-*`).

## D10 — LoRA target modules (Phase 2, introspected, not yet used) — PROPOSED
Introspected the real whisper-small architecture on the `meta` device (no weights):
12 enc + 12 dec layers, d_model 768; Linear names `self_attn.{q,k,v,out}_proj`,
`encoder_attn.{q,k,v,out}_proj` (decoder cross-attention), `fc1`, `fc2`, `proj_out` (tied to
embeddings). LoRA r=32 on all q/k/v/out_proj = **7.08M trainable params (2.93% of 241.7M)**.

## Owner decisions needed (Phase 0 gate)
1. Approve downloading whisper-small weights (967 MB) and from which endpoint (D7).
2. Approve datasets + intended use (private demo vs public release) — `docs/data_sources.md`.
3. Sign off normalization policy `zh_en_v1` (D4) and baseline decoding (D6).
4. Hardware/budget for Phase 2 (D1): local M5 only, or a CUDA GPU with a $ cap?
5. Whether you will record own/volunteer audio (consent form needed) and the v1 hotword list.

## Owner decisions — 2026-10-27
- **Model download approved.** whisper-small weights fetched via hf-mirror.com (huggingface.co
  unreachable) at commit 973afd2; sha256 verified against the LFS hash reported by the mirror API
  (same third party, so this is an integrity check, not an independent authenticity check).
- **Private use only; the model will not be released.** Non-commercial / NC-ND corpora are
  acceptable in principle, but each license still needs individual review against private
  research use.
- **Normalization is required for scoring.** The exact policy must match the existing benchmark
  v1.0 so regression numbers stay comparable (pending: v1.0 spec).
- **Hardware:** local M5 for now; the owner will rent GPUs later when needed.
- **Owner can record audio** and **will supply a hotword list per case.**
- **Existing regression benchmark v1.0** is the baseline reference. Location, format and
  scoring rules are pending; we integrate with it before building any new test set.

## Owner decisions — 2026-10-29
- **Spanish clips excluded from benchmark v1.** 7 clips tagged `en` are Spanish speech. They are excluded like the
  Japanese ones (`split: unassigned`, flag `excluded_non_target_language_es`). Benchmark v1 = **412 clips**. Old
  manifest kept as `benchmark_v1.pre_es_exclusion.jsonl`. Report: `reports/benchmark_v1_fixed.md`.
- **Reference baseline confirmed by owner (2026-10-29):** whisper-small + decoding fixes `allowed_languages: [zh, en]` and `output_script: simplified`
  (`configs/experiments/benchmark_v1_fixed.yaml`). = **12.63% official MER on 412 clips** (VoiceCursor 7.63%). Phase 2 models are compared against this, same decoding settings.
- **ASCEND full download approved** (train/val/test, ~1.2 GB) from huggingface.co (reachable again on 2026-10-29),
  pinned to dataset commit 737e980; sha256 checked against the LFS hashes from huggingface.co (`data/private/ascend/raw/SHA256SUMS`).
  The val parquet fetched from hf-mirror on 2026-10-28 matches the official hash.
- **benchmark_chinese approved as training data** (`~/Desktop/build_benchmark/benchmark_chinese`, VoiceCursor production
  recordings; owner states they are authorized to use it for training). 408 clips with audio + GT, 1.86 h, 85 speakers.
  - The 5 recordings that are also in benchmark v1 are **removed** (identical audio = test leakage).
  - 20 speakers (140 clips) also appear in benchmark v1: owner chose to **keep** them. Consequence: benchmark v1 results
    may be inflated for those speakers, so benchmark v1 is reported both overall and split by seen / unseen speaker.
  - `screen_text` / `audio_context` (PII) is not copied into this project.
  - OPEN: whether this audio may be uploaded to rented cloud GPUs. Until answered, training on it stays local.
- **Training-label style: option B (owner, 2026-10-29).** Train on the labels as the corpora provide them (ASCEND:
  lowercase, no punctuation; benchmark_chinese: punctuated, cased). No manual rewrite. Formatting stays a separate
  stage (plan Phase 5). Risks to watch: the fine-tuned model may stop producing punctuation/casing. So every eval also
  reports the share of outputs that contain punctuation. If it drops, try LLM-restored punctuation on ASCEND
  (spot-checked) as a second experiment. ASCEND label quality is estimated by a 50-clip listening spot check,
  not by full review.
- **Filler removal in ASCEND training labels (owner, 2026-10-29).** Delete only **呃** and **嗯** (including runs such as
  嗯嗯) from ASCEND transcripts. The audio is not changed. If a transcript is empty afterwards, drop the clip. Keep 啊
  (sentence-final particle, kept in our refs), 哦/噢 ("哦，对了"), um/uh (kept in benchmark v1 English refs) and 额
  (part of 额度). Why: ASCEND transcribes hesitations verbatim (呃 128×, 嗯 81× in val). Our references almost never
  contain them (呃 0×, 嗯 1× across benchmark v1 + benchmark_chinese), so a model trained on them would insert them.
- **Clips < 0.3 s are dropped** (validation error; almost all single-syllable backchannels 嗯/对/啊). Clipping warnings
  (mostly speaker ascend-12) and 0.3–0.5 s clips are kept.
- **ASCEND keeps its official speaker-disjoint splits (owner, 2026-10-29).** train → `train` (9000 clips, 8.17 h);
  validation → `val` (1013, 0.84 h); test → `test` (1151, 0.85 h), used only as a **diagnostic** set: never for
  training or for any choice. The final test set remains benchmark v1 (412 clips). The ASCEND-test vs benchmark-v1 gap
  shows whether gains transfer to our users. Validation will be extended with a speaker-disjoint slice of
  benchmark_chinese when that is ingested.
- **benchmark_chinese ingested (2026-10-29, option A for clips > 30 s)** → `data/manifests/benchmark_chinese.jsonl`.
  403 clips after dropping 92 without audio and 5 identical to benchmark v1. Val = 22 speakers not in benchmark v1
  (seeded order, until ≥ 0.12 h of ≤ 30 s audio): 68 clips / 0.41 h, including their > 30 s clips. Train = all other
  speakers' ≤ 30 s clips: 289 clips / 0.78 h / 60 speakers (19 of them also in benchmark v1). Not used: 34 clips > 30 s
  from train speakers, plus 2 clips whose transcript is far longer than the audio. Splitting long clips into ≤ 30 s
  pieces (option B) is a possible later experiment.
- **Pilot training data:** ASCEND train 8.17 h + benchmark_chinese train 0.78 h = **8.95 h**. Validation: ASCEND val 0.84 h +
  benchmark_chinese val 0.41 h.
- **ASCEND label style v2 (owner, 2026-10-30).** (1) English number words → digits with `text2num` (`alpha2digit`,
  threshold 9): 1–9 stay spelled out, ≥ 10 become digits. This matches benchmark v1, where one/two/three…
  are written as words 130× and numbers ≥ 10 almost always as digits. Chinese numerals are never touched (一个/一些/第一 are
  words). (2) English **um/uh are now removed too** (revises 2026-10-29). lora_v1 inserted 80 more um/uh than the baseline
  on benchmark v1 English clips ≤ 30 s, and spelled-out numbers cost ~67 errors there. Applied in the ASCEND adapter
  (`strip_fillers=True, en_digits=True`) to all three ASCEND splits, so ASCEND test references follow the same style.
  The original labels remain in the raw parquet files.
- **arena_019f8e51 (English, "Refined V3 English E2E", 883 clips / 3.07 h) used for TRAINING, option B (owner, 2026-10-30).**
  Train = clips ≤ 30 s with human-edited, human-reviewed **or 3-model auto-passed** labels. The auto-passed labels were
  never listened to: arena run 01a01e3f, average strict WER < 5 % / 7 % across 3 models. They carry the flag
  `label_auto_passed_3model` for a later A-vs-B ablation. Excluded: 70 `rw.self_correction` labels (the label drops
  spoken content), 3 unreviewed, clips > 30 s. No val (no speaker ids). Caveats: these may be the same internal speakers
  as benchmark v1's arena uploads (2 near-identical scripts were found and excluded), so benchmark v1 English results
  may be optimistic. Labels are written-form (punctuation, casing, digits, no fillers), matching the house style.
- **Cloud training approved (owner, 2026-10-30).** All training data, including the internal recordings
  benchmark_chinese and arena_019f8e51, may be uploaded to a rented AutoDL GPU for lora_v2. The cloud copies are to be
  deleted after training.
- **Reference decoding now includes temperature fallback (owner confirmed, 2026-10-30).** `long_form.temperature_fallback:
  [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]`, `compression_ratio_threshold: 1.35` (HF token-level, ≈ OpenAI's 2.4),
  `logprob_threshold: -1.0`, seeded per clip. Long-form only. Chosen on dev data: it fixed the reproduced lora_v1 loop
  (MER 475% → 8%) and left the baseline unchanged. The workaround for the transformers 5.21 MPS sampling bug lives in
  `transcribe.py`. The baseline and every model are re-scored on benchmark v1 with this setting.
- **Training: early stopping + resumable runs (owner, 2026-10-30).** Max 3 epochs; stop after 3 evals without a val-loss
  gain of 0.001. Early stopping only decides *when to stop*; the checkpoint is still chosen by decoded val MER (all kept).
  `wf-train --resume` restores LoRA weights, optimizer, LR schedule and counters from `training_state.pt`.
- **lora_v2 trained on AutoDL (RTX PRO 6000 Blackwell, bf16, batch 16, no gradient checkpointing, 0.19 s/step).**
  Early-stopped at step 1200 (≈1.9 epochs). Checkpoint **step 1000** (≈1.6 epochs) chosen on validation only:
  users' val tied with the base model (14.01% vs 13.89%, 5587 tokens, CIs ≈10–20%), so ASCEND val (13.97%, best) and
  long-form val (14.43%, no loops) broke the tie. Step 1200 showed short-clip loops ("对对对…") on ASCEND val.
  The cloud copy is kept for the next round (owner, 2026-10-30): instance shut down, not released. Delete when the
  experiments end.
- **New arena datasets (owner, 2026-10-30).** 01a0e713: nothing added (all duplicates of benchmark_chinese, 2 of them benchmark
  v1 test). 019f6442: train only the 33 human-edited clips (0.11 h). Its 68 unreviewed GPT-5.5 intent labels are **not used**.
  6dbd0a4e ("E2E Intent ZH-EN", a public evaluation benchmark used in 17 arena batches) is **protected as an eval set**:
  never trained on.
- **Correction, 019f6442 review status (owner, 2026-10-30).** That dataset records review at dataset level ("reviewed by
  default (lenient)"), not with a per-item `reviewed` tag, so the earlier "70 never reviewed" was a wrong reading of the
  019f8e51 rule. Owner: use them. Unedited labels get the flag `label_reviewed_by_dataset_policy`. Still excluded:
  self-correction and > 30 s.

## Owner decisions — 2026-10-31

- **E3 segmented long clips go into lora_v3** (owner spot check: cuts 9/10 clean, 1 edge). Piece 2 label corrected, piece 7
  dropped via `configs/label_corrections.yaml`. lora_v3 config = lora_v2 + arena_019f6442 + segmented_long (train 12.37 h).
- **Target output style = cleaned text (owner).** Label spot check (reports/dataset/label_style_spotcheck.md): 10/12
  benchmark_chinese/arena references are cleaned, not verbatim (fillers, repeats, self-corrections, grammar, list
  formatting); owner says benchmark v1 references are mostly cleaned too. The system should output cleaned text.
  Open: how ASCEND's verbatim labels (8.16 h) fit with this, and whether cleaning is learned end-to-end by Whisper or by a
  separate post-processing step (plan.md §1 assumed the latter).
- **lora_v3 on AutoDL (owner, 2026-10-31).** Upload approved for arena_019f6442 (wavs only) and the E3 segmented pieces,
  same terms as lora_v2 (cloud copies deleted after the experiments). Protocol as lora_v2: early stopping, checkpoint
  chosen on val only, benchmark v1 once at the end. Adopt only if clearly better than base (11.36%, CI excluding 0).
- **lora_v3 not adopted (2026-10-31).** benchmark v1 11.64% vs baseline 11.36% (+0.28 [−0.65, +0.99]); zh-en −1.51
  [−3.04, −0.09], same weak signal as lora_v2. Report: reports/lora_v3.md. Checkpoint chosen on val only (step 1200).
  Note: 33/271 benchmark v1 English references are verbatim (keep um/uh/you know), unlike the rest of the set.
- **Zero-shot bigger models (owner, 2026-10-31).** Owner asked about large-v3 and booted the cloud box after the
  proposal: zero-shot medium / large-v3 / large-v3-turbo on the three val sets first, then decide what to fine-tune.
  Weights (model.safetensors only, 7.8 GB) downloaded on the cloud box via hf-mirror.com with Xet disabled; pinned
  revisions medium abdf7c39, large-v3 06f233fe, large-v3-turbo 41f01f3f. sha256 checked against the LFS hashes, which
  match huggingface.co's own (reached through AutoDL's `source /etc/network_turbo`). Configs:
  configs/experiments/zeroshot_*.yaml; script scripts/zeroshot_val.sh.
- **Production-traffic data collection approved (owner, 2026-10-31).** Plan: docs/data_collection_plan.md (5 hard
  scenarios; locked test 900 / val 200 / train 1,800 clips; Phase 1 = test S1+S2 + val). Owner decisions:
  (1) production user audio may be used for test and training; (2) drafts from **Gemini** (third-party upload of
  production audio approved for drafting; OpenAI drafts not approved), then the owner corrects every label;
  (3) reference style = `formatted_transcript` rules in docs/annotation_policy.md; (4) the owner is the reviewer;
  (5) new training audio may go to AutoDL on the same terms; (6) target counts accepted.
- **Speaker overlap allowed for production data (owner, 2026-10-31).** Train and test may share users; every test clip
  is tagged seen / unseen (user present in train or not) and both are reported. **Adoption is judged on all users**
  (owner chose B; the stricter unseen-only rule was recommended). Text near-duplicates of test clips are removed from
  train. Review time estimate corrected to ~20–40 s per clip (median clip 5.2 s), to be measured on the first 50.
- **Base model = whisper-large-v3 (owner, 2026-10-31).** Zero-shot on users' val 9.97% vs small 13.89%
  (reports/zeroshot_bigger_models.md). Inference speed (RTF 0.091 on the cloud GPU vs turbo 0.025) is deferred:
  study acceleration later (e.g. CTranslate2 / faster-whisper int8, batching, turbo as a draft model).
- **Test recordings excluded; Whisper drafts instead of Gemini (owner, 2026-11-01).** Owner: many clips are test data
  (一二三四…) and Gemini drafts are poor. scripts/test_utterances.py removes counting, mic checks / QA scripts
  ("testing testing", "validation number …", "这是一段…测试"), short texts read by ≥ 2 users, and 6 tester accounts
  (≥ 50% test-like traffic): 2,003 of 12,831 scanned requests. Drafts now come from zero-shot whisper-large-v3 run
  locally (scripts/whisper_drafts.py), so no production audio goes to a third party. The Gemini round is kept in
  data/private/online_phase1/v1_gemini/.

## Owner decisions — 2026-11-02

- **benchmark_chinese labels were production model output, not human labels.** In ~/Desktop/build_benchmark/
  benchmark_chinese all 500 ground_truth.csv transcripts equal index.csv asr_text (gpt-4o-transcribe 392,
  gpt-transcribe 56, gemini-3.5-flash 52); the index ground_truth column is empty. Our manifest had called them
  `source_provided`. So all earlier users'-val numbers measured agreement with gpt-4o-transcribe.
- **Calibrated labels applied (owner).** ~/Downloads/01a194ac…/source_ground_truth/sample benchmark中文专项.csv replaces all
  403 benchmark_chinese transcripts (120 changed; old vs new 6.5% MER); label_status human_verified, reviewer
  `voicecursor_calibration_2026-11-02`, flag `label_calibrated`. The owner's own correction of 05e575ab (heard
  2026-11-02) overrides the calibrated text; 60f9ed33 uses the calibrated text. Old manifest kept as
  benchmark_chinese.pre_calibration.jsonl. 14 long parents changed, so segmented_long is regenerated.
- **benchmark v1 = v1.1 already.** ~/Downloads/01a194b2…/ground_truth.xlsx (425 rows) has the same ground_truth as our
  manifest for all 425 clips (0 changes). Its `verdict` / `notes` judge the product output (VoiceCursor), not the label.
  v1.1 references closely track VoiceCursor's own output (pass rows: 1.6% MER, 280 identical), so they were probably
  made by correcting VoiceCursor output: report this caveat with any VoiceCursor comparison.
- **lora_large_v3 A / B done (2026-11-02, reports/lora_large_v3_ab.md).** Users' val (calibrated): zero-shot 8.70%,
  A step 1400 8.19%, B step 800 8.22% (both n.s. vs zero-shot); ASCEND val ≈ halved. B ≈ A with 1/3 of ASCEND, so
  the next round keeps ASCEND down-sampled. Candidate checkpoint: B step 800, to be judged on the new locked test set.
  Lesson: instance copies sometimes carry only the directory tree; key checkpoints are now backed up locally
  (runs/lora_large_v3_{A,B}/checkpoints, sha256 checked).
- **benchmark v1 upload to AutoDL approved for evaluation (owner, 2026-11-02).** Delete the cloud copy after the
  experiments. Evaluated once: large-v3 zero-shot, B step 800 (candidate chosen on val beforehand), A step 1400
  (reference only, may not change the choice).
- **Numeral style is a learning target (owner).** Official MER stays the headline metric (wrong numeral style counts).
  official_mer_num ("三点五十" = "3:50") is a diagnostic only. The real fix: one numeral rule for all references,
  to be decided after the owner's review of the new data.
- **benchmark v1 large-v3 results (reports/benchmark_v1_large_v3.md).** small 11.36% → large-v3 zero-shot 10.33% →
  B step 800 8.67% (A 1400 8.55%, reference only) → VoiceCursor 7.63%. 90% of B's gain over zero-shot is one 2 s
  clip where zero-shot loops; without it B − zero-shot = −0.70 [−2.32, +0.30]. Next: short-clip loop guard (E5).
- **online phase 1 reviewed → lora_large_v3_C (owner, 2026-11-02).** Owner reviewed 399/400 clips (ok 83, fix 204,
  drop 112 = low quality; 7 s per clip). The 230 kept "test" clips become TRAIN (owner's choice; no new locked test
  set yet), the 57 val clips join users' val → data/manifests/online_phase1.jsonl (0 identical audio with benchmark v1;
  10 train speakers also in benchmark v1). C = B recipe + these 230 + calibrated benchmark_chinese + regenerated
  segments. **Predeclared selection rule:** lowest pooled official MER on users' val (benchmark_chinese val 68 +
  online val 57, scripts/pick_checkpoint.py), then benchmark v1 once for that checkpoint, compared with B 800 (8.67%).
- **Spelled-out letters → the assembled word (owner, 2026-11-02).** "把部署平台换成 Vercel，V-E-R-C-E-L" is labelled
  "把部署平台换成 Vercel。" (option A), like applying a self-correction. Training-pool selection should favour clips that
  teach: proper nouns (711 clips / 2.3 h in the pool), intonation (questions / exclamations 910), self-correction and
  spelling. The last two are rare in production (≈ 69 / 9 rule hits, mostly false positives), so they get scripted
  recordings instead.
- **lora_large_v3_C result (2026-11-02).** Picked step 600 by the predeclared rule; benchmark v1 8.35% vs B 800 8.67%
  (−0.32 [−0.82, +0.31], n.s.; zh / zh-en flat, en −0.61). New best checkpoint by a small margin: C step 600.
  RTX PRO 6000 vs RTX 5090 decoding: identical outputs (68/68), so cross-GPU comparisons are fine.
- **H100 cluster approved (owner, 2026-11-02).** SLURM cluster coding.bitoasis-ai.cn (account `mingyang`, shared
  `h100users` storage, 8×H100 nodes). Owner confirms the account may be used and that the training / evaluation data
  (incl. production recordings and benchmark v1) may be stored there. Data lives in a private directory (chmod 700)
  and is deleted after the experiments; jobs use `srun` for one GPU and leave the account's other jobs alone. The SSH
  private key was pasted into the chat, so it should be rotated after this work.
- **Permission slip on the H100 cluster (2026-11-02).** ~/whisperflow was created 700, but the first code rsync
  copied the local 755 onto it; from about 03:39 to 05:05 (cluster time) the directory, including uploaded audio,
  was readable by the `h100users` group. Fixed (700 on ~/whisperflow, data/private, runs, hf_cache), and
  scripts/cloud_sync.sh now re-applies 700 after every sync. No access log is available to check whether anyone read it.
- **Reference decoding v3 = v2 + short-form loop guard (E5, 2026-11-02).** decoding.short_form.loop_guard: temperature
  fallback [0, .2, .4, .6, .8, 1] for clips ≤ 30 s, triggered only by compression ratio > 1.35 (no log-prob trigger).
  Validated on val with zero-shot large-v3 on the H100: online val 18.87% → 6.31% (only the looping clip changed:
  "我只想死死死…" → "我只想死死现在"), users' val 8.70% → 8.70% (0 of 68 outputs changed). Other outputs identical to
  the RTX PRO 6000 runs. Config: configs/experiments/zeroshot_large_v3_guard.yaml; use it (with adapter_path) from now on.

## Owner decisions — 2026-11-04

- **lora_large_v3_G result.** D recipe + online_pool batch 1 full (451 clips, +184): users' val 7.73% (D 7.42%), benchmark
  v1 8.06% (D 8.10%; G − D −0.04 [−0.25, +0.26]). No change; D step 600 stays the candidate (reports/lora_large_v3_G.md).
- **benchmark v1 cannot resolve our steps.** 58% of its token weight sits in 69 clips > 30 s (the 10 longest clips carry
  34% of the errors); the paired G − D standard error is 0.13 points, so ≈ 640 clips are needed to see 0.3 points.
- **benchmark v2 (owner).** A new locked test set of 1–30 s production clips. Planned zh 40% / zh-en 30% / en 30% with
  unseen speakers only (stricter than the 2026-10-31 rule), but production has too little Chinese (30 days ≈ 130
  zh / zh-en users, most already in our data). Owner chose: every eligible zh / zh-en clip (≤ 5 per user, unseen users
  first, then users seen in training or the pool, tagged `seen`; never val / test users) + 240 unseen en users (1 clip
  each). Result: 537 clips / 1.28 h (zh 163 / 81 users, zh-en 134 / 59, en 240 / 240; 72 seen clips), 3 review packs
  of 179 (data/private/benchmark_v2, scripts/select_benchmark_v2.py → fetch_benchmark_v2.py → prepare_review_benchmark_v2.py).
  Drafts: D step 600 (local model, no third party). arena_019f8e51 speakers cannot be excluded (no user ids; internal
  recordings). Pre-filled with the production text, so v2 comparisons with VoiceCursor may lean toward it.
  Consequence for training data: the remaining unreviewed pool (≈ 6.7 h zh / zh-en) is close to all the Chinese
  production audio there is; +5 h of Chinese training data needs another source (scripted recordings, new traffic).
- **lora_large_v3_H (owner recordings ×1) does not help (reports/lora_large_v3_H.md).** Same 10/13 trained terms on the
  owner's held-out freeform test as G; benchmark v1 zh-en +0.94 [+0.23, +1.63]. D step 600 stays the candidate.
- **Layering: ASR transcribes, post-processing cleans (owner, 2026-11-04).** Spelled-out letters are now labelled as
  spoken, upper case, joined by "-" ("Vercel，V-E-R-C-E-L"); this replaces the 2026-11-02 rule (assembled word) for ASR
  labels — the post-processing LLM assembles them. Self-corrections stay verbatim (annotation policy raw); the
  post-processing LLM applies them. Fillers 嗯/呃/um/uh stay removed.
- **Recording script v2 for a new speaker (scripts/build_recording_script_v2.py, 205 prompt cards).** Prompt cards
  instead of fixed sentences (v1 lesson); blocks S spelling 69 / C self-correction 38 / E entities 68 (Chinese products,
  places, names, homophones; English tools inside Chinese) / G general 30. 83 cards (40%) are TEST only: one card of
  every entity group plus terms that only occur in test (unseen), so it measures both learning and generalisation.
- **Trial server (src/whisperflow/serving/server.py).** OpenAI-compatible /v1/audio/transcriptions + test page, D step
  600, bf16 greedy, one request at a time; reached through an SSH tunnel only. Latency spikes seen in the trial come from
  the short-form loop-guard retries (spelled letters), > 30 s sequential long-form, and sharing the GPU with evaluation.
- **Next track: distil the production post-processor into our own fast model (owner, 2026-11-04).** Teacher = the
  production Gemini cleanup, student = a small open LLM (Qwen3 1.7B / 4B) fine-tuned on (Langfuse post-process input →
  PostHog `processed`) pairs from production, filtered with the online.py flags, then improved with human corrections;
  served with vLLM / TensorRT-LLM (FP8, speculative decoding with the input as draft) on the same GPU as the ASR.
  Target ≈ 0.7 s end to end on average. Plan, data, steps and open questions (Gemini terms of use, consent):
  docs/postprocess_distillation_plan.md.
- **Planned: LoRA on whisper-large-v3-turbo, to choose the speed / accuracy trade-off (owner, 2026-11-04).** turbo =
  the large-v3 encoder with 4 decoder layers instead of 32. E7 (H100, HF bf16 beam 5): turbo zero-shot <5 s p50 0.12 s,
  5–15 s 0.22 s vs large-v3 + LoRA 0.33 / 0.67 s (≈ 2.7–3× faster), users' val 8.94% vs 7.52% (large-v3 zero-shot
  8.70%). Experiment: the G (or then-current) recipe on turbo, same selection rule, then report MER (users' val,
  benchmark v1/v2, owner test) next to latency (p50/p95/p99 by length, faster-whisper and HF) for large-v3+LoRA vs
  turbo+LoRA, so the serving model is picked on both axes. Hypothesis: turbo+LoRA ≈ 7.5% at ≈ 0.1 s for short clips.
- **The trial server is NOT yet accelerated (2026-11-04).** It runs transformers bf16 greedy (faster-whisper /
  CTranslate2 is not installed on the AutoDL box and Blackwell / Ada support is unverified). Next speed steps:
  faster-whisper fp16 (E7: ≈ 2× on short clips at equal MER), dedicated GPU (no evaluation on the serving GPU),
  turbo+LoRA, TensorRT. Trial latency on the 4090 D was ≈ 2× the H100 numbers for <5 s clips (memory bandwidth
  ≈ 1 vs 3.35 TB/s) plus contention with the H4 evaluation.
- **lora_large_v3_H4 (recordings ×4) result (2026-11-04).** Picked step 800 (users' val 7.43%); benchmark v1 7.95%
  (lowest of all runs; H4 − G −0.12 [−0.53, +0.36]); owner freeform test trained terms 12/13 (G / H 10/13). Proposed as
  the new candidate over D step 600 (owner to confirm). Lesson: one-speaker scripted data only registered when
  oversampled to ≈ 10% of training clips (reports/lora_large_v3_H.md).

## Owner decisions — 2026-11-05

- **Spelled letters stay in ASR labels; assembly belongs to post-processing (owner confirmed option A).** Evidence: the
  letters carry the information that fixes a misheard word (V023: ASR "Repl" + "R-E-P-L-I-T" → Replit), and the
  production post-processor handles spelled letters inconsistently today (21 v2 takes: 5 left the letters as they were,
  others turned them into "MOBBIN" / duplicated "Figma，Figma"), so the rule has to be explicit in the post-processing
  spec and its eval set (docs/postprocess_distillation_plan.md).
- **recording_v2 ingested** (scripts/recordings_v2_to_ingest.py): recorded 2026-11-05 mostly by the owner on the owner's
  production account (speaker id = owner, as recording_v1). Train 90 clips / 6.9 min (12 spelling labels written by the
  assistant to the policy at the owner's request, reviewer_id owner+assistant_expected); test 41 clips / 2.8 min (only
  owner-verified labels; 10 spelling clips with non-conforming labels dropped rather than moved to train, so the
  test-only words stay unseen; 1 clip identical to a v1 training label removed). v2 test therefore measures the owner's
  voice on unseen sentences and unseen words, not other speakers.
- **Next run I**: H4 recipe (recording_v1 ×4) + recording_v2 train ×4; evaluate on users' val (selection), benchmark v1
  once, owner freeform test and recording_v2_test, against G / H4.
- **Post-processing distillation approved (owner, 2026-11-05).** Gemini outputs may be used to train our own
  post-processing model (owner's decision on the terms-of-use question in docs/postprocess_distillation_plan.md), and
  users' production text is authorised for training, on the same basis as the ASR training data.
- **Post-processing phase 1 = dictation only (owner, 2026-11-05).** Edit mode (select text + spoken instruction) moves
  to phase 2. Phase 1 covers dictate cleanup with formats (plain / email / list), tone presets (casual / formal /
  very casual) and the user's free-text tone preference.
- **Languages: Chinese and English only (owner, 2026-11-05),** for both ASR and post-processing, although production
  post-processed traffic is en 63% / pt 32% / zh 4.3% (data audit). Portuguese stays on the production models.
- **Tone is chosen by the app (bundle id → casual / formal) (owner, 2026-11-05).** The student gets the tone as an input
  field derived from the bundle id; the bundle-id → tone mapping must come from the backend (the audit did not find the
  preset tone text in the logged post-processing or ASR prompts).
- **ASR labels are strict transcripts (owner, 2026-11-05; option A).** Write what was said; remove only 嗯/呃/啊/um/uh;
  keep self-corrections, repetitions and slips verbatim; spelled letters as "X-Y-Z"; "at" / "dot" as spoken. The
  post-processing model applies corrections, assembles letters and formats. lora_large_v3_I learned to apply
  corrections inside the ASR because 26 of 30 v2 self-correction labels (and 4 v1 labels) had the correction applied;
  these and 22 other non-verbatim recording labels (omitted repeats / words / letters, formatted e-mail) are being
  re-labelled (review package strict-v1v2, 50 clips, drafts merged from the production ASR text and the old label),
  then run I2 = I recipe with the corrected labels.
- **lora_large_v3_I2 = new candidate (proposed 2026-11-05, reports/lora_large_v3_I.md).** I recipe with strict-transcript
  labels. Picked step 600 (users' val 7.49%); benchmark v1 7.74% (best so far; I2 − H4 −0.20 [−0.60, +0.29]; vs
  VoiceCursor +0.12 n.s.); recording_v2 test 9.43% (spelling 6.45%, self-correction verbatim 11.72%, entities 9.83%).
  Checkpoints backed up locally.
- **I2 step 600 is the candidate (owner confirmed, 2026-11-05).** Replaces H4 step 800; the trial server should load
  runs/lora_large_v3_I2/checkpoints/step_600 next time it is started.

## Owner decisions — 2026-11-06

- **Post-processing status (2026-11-06).** Decided: phase 1 = dictation, Chinese + English only; student = Qwen3-1.7B /
  4B (LoRA SFT) with a BERT-style tagger baseline for the plain-cleanup segment; training pairs = PostHog
  `transcription` → `processed` (equals the teacher's real input / output, 16/16 checked), filtered with the online.py
  flags, de-duplicated, capped per user; Gemini outputs and users' text authorised; ASR transcribes strictly, the
  post-processor removes fillers, applies self-corrections, assembles spelled letters and formats. Data audit:
  reports/postprocess_data_audit/report.md (zh / zh-en only ≈ 4% of post-processed traffic → Chinese pairs will be
  generated by running the teacher prompt on our own Chinese ASR texts). Waiting for: the bundle-id → tone mapping and
  the owner's training-data design.
- **Outputs must be conditioned (owner, 2026-11-06).** The same transcript can need different outputs by scenario (app),
  tone (casual / formal by bundle id) and, later, user tags. The student must learn this conditioning: every training
  pair carries these fields as input, and the data / eval sets must contain the same transcript-type under different
  conditions. Data design to be discussed with the owner before the first SFT run.
- **Next experiment: decoding-time contextual biasing (FST / prefix-tree "hot-word" boosting) on I2**, compared with no
  biasing and with prompt biasing, on the owner tests (dictionary terms), benchmark v1 with random distractor
  dictionaries (false triggers), and production user dictionaries; metrics B-WER / U-WER.

## Owner decisions — 2026-11-09

- **Qwen3-ASR zero-shot check (owner asked, 2026-11-09; reports/qwen3_asr_zeroshot.md).** 1.7B with 25 s chunks
  (chosen on users' val): users' val 8.30%, benchmark v1 9.72% (I2 7.74%, VoiceCursor 7.63%); bv1 zh 8.59% (I2 9.30%),
  en 9.71% (I2 6.99%, significant gap). 0.6B clearly weaker. Benchmark v1 run once per setting (logged). Next step
  (fine-tune Qwen3-ASR-1.7B on the I2 mix) pending owner decision.
- **Post-processing candidate data v0 (owner asked, 2026-11-09; reports/postprocess_data/candidates_v0.md).** Organised
  from 30 days of production pairs (read-only, no LLM calls). Open owner questions are listed in that report.
- **Qwen3-ASR-1.7B fine-tune approved (owner, 2026-11-09):** train on I2's exact data mix (same manifests, same
  oversampling, same strict labels), compare against I2 with the same selection rule and benchmark v1 once.
- **Post-processing rules (owner, 2026-11-09):** casual tone → drop only the sentence-final period (mid-sentence
  periods stay). "帮我写封邮件给 X 说…": in dictation mode keep the instruction in the output text; in the other mode
  (owner said "animal" — to confirm which mode) write the literal e-mail. Bundle id → tone mapping file: owner will
  send it.
- **qwen3asr_I2 result (2026-11-09, reports/qwen3_asr_zeroshot.md):** Qwen3-ASR-1.7B full fine-tune on I2's data,
  step 1000: users' val 7.04% (I2 7.49), benchmark v1 9.66% (I2 7.74; en +2.66 significant, zh / zh-en n.s.),
  recording_v2 test 3.70% (I2 9.43). Loss concentrated in > 30 s English (cleaning style + weaker long-form recognition).
  Not adopted; I2 step 600 stays the candidate. Next direction pending owner.
- **Tone by app (owner, 2026-11-09):** mapping saved as configs/reference/tone_by_app.tsv (79 apps; casual / neutral /
  formal / unknown). Keyed by app name; bundle-id join to be built from production logs.
- **End-to-end first (owner, 2026-11-09):** try one model (Qwen3-ASR-1.7B): audio + context (hotwords / vocabulary,
  text before / after cursor, tone) → output in two parts (strict transcript, then final text). No rewriting / edit
  commands in this phase. Option 1 (ASR + post-processing cascade) is the comparison. First step: pull the context
  fields production logged for the clips we already have and review coverage.
- **ASCEND out of the end-to-end training data (owner, 2026-11-09).** Its labels read like raw ASR output (repeats
  not cleaned, no punctuation, casual) and are not the output we want. Consistent with ablation F (no harm on real
  users' audio without ASCEND).
- **Label conventions (owner, 2026-11-09):** spaces before and after every English word inside Chinese text; prefer
  Arabic numerals (scope of "prefer" to confirm); 啊 depends on its role (to confirm: meaningful sentence particle
  kept, hesitation filler removed). Arena English labels that conflict with the strict-transcript rules may be
  re-annotated (owner).
- **5a + 5b together (owner, 2026-11-09):** the owner will provide an audio set with both a transcript and a final
  output per clip, so the end-to-end model's two-part output (strict transcript, final text) can be trained and
  evaluated from the start.
- **Label conventions confirmed (owner, 2026-11-09):** 啊 kept as a meaningful sentence-final particle, removed as a
  hesitation filler; Arabic numerals for quantities / money / time / dates / numbers, characters kept in fixed words,
  idioms and approximate numbers. Written into docs/annotation_policy.md v1.
- **Run J1 = qwen3asr_J1 (owner, 2026-11-09):** Qwen3-ASR-1.7B full fine-tune on I2's data without ASCEND, labels in
  policy v1, production context (app · tone, user vocabulary, text before / after cursor; screen text left out) as
  the system prompt; cursor fields that share an 8-character run with the label are dropped (train and eval; on eval
  this is an oracle step). Output = strict transcript only (final output comes with the owner's reviews).
  **Predeclared before results:** checkpoint picked by pooled users' val MER against references brought to policy v1
  (scripts/pick_checkpoint.py --refs v1); all systems are compared on v1 references as well as the original ones.
  The job script picks on the original references; if the v1 pick differs, the v1 pick is the one tested.
- **Unified review pages (owner, 2026-11-09):** scripts/build_e2e_review.py — audio + context + strict transcript
  (v1) + final output + production output as reference. Packs: e2e-dict (207), e2e-val (125), e2e-prod-1…6 (1,198).
- **qwen3asr_J1 result (2026-11-09, reports/qwen3asr_J1.md):** step 400; users' val (v1 refs) 6.80% without context
  (best so far), 7.47% with context (copies context text into the output); dictionary benchmark zh/en 12.22% with
  context, but target-word recall 63% vs zero-shot Qwen 75% (training taught it to ignore the mostly irrelevant
  logged dictionary). benchmark v1 9.51% (long English gap unchanged). J2 (augmentation) proposed.
- **Dictionary benchmark reviewed (owner, 2026-11-09):** e2e-dict 207 → ok 97 / fix 100 / drop 6 / 4 not reviewed
  (110 min, median 14 s per clip). 39 strict transcripts changed (missing words in the original ground truth,
  punctuation), 75 final outputs differ from the transcript (casual: no final period, repeats / fillers removed, light
  tightening). data/manifests/dictionary_bench.jsonl now uses the reviewed transcript (formatted_transcript = final
  output); old version dictionary_bench.pre_review.jsonl. Rescored: J1 + context 11.38% zh/en, Qwen zero-shot +
  context 11.36%, Whisper I2 15.06%; target-word recall 63% vs 75% unchanged.
- **Run J2 = qwen3asr_J2 (owner approved 2026-11-09):** J1 + context augmentation (scripts/make_qwen_e2e_data.py
  --augment): vocabulary A 40% logged / B 40% label terms + 5–30 distractors incl. look-alikes (falls back to A when
  the label has no term, so B ≈ 15% in practice) / C 20% distractors only; cursor text A 45% own / B 20% previous
  utterance trap / C 15% another clip's / D 20% none, lengths cut to 50–200 / 30–120; segmented_long cursor text only
  on the first / last piece. Two leaks closed for J2 (and its eval contexts): the owner's recordings get no cursor
  text (the script, incl. recording_v2 TEST sentences, was open in the same document) and the vocabulary is the
  user's dictionary only (ctx_vocabulary = screen-extracted words). So J1's recording_v2-test "with context" score
  (4.07%) is contaminated. Predeclared: pick on users' val with context, v1 references; report MER, target-word
  recall, copy rate, distractor insertions (scripts/context_metrics.py), with / without context / with copy guard
  (decoding.copy_guard); J1 re-decoded with the same J2-form contexts.
- **Synthetic data, owner decisions (2026-11-09):** paid TTS allowed — owner compares vendors and picks one; no voice
  cloning (stock voices only); text written by an LLM (for the 500-sample pilot: a Claude subagent, nothing sent to
  external APIs); start with a 500-sample pilot; terms invented by content (high-frequency style), not copied from
  production; the dictionary benchmark stays a pure test set (its target words banned from synthetic text).
- **qwen3asr_J2 result (2026-11-09, reports/qwen3asr_J2.md):** step 552. Copying fixed (0/125), context neutral on
  users' val (J2 ctx 6.85% vs noctx 6.66%, n.s.), target-word recall 72% (J1 64%, zero-shot 75%), but dictionary
  benchmark MER worse than J1 (+1.84 [+0.33, +4.03]; English text instability, distractor insertions 12 vs 2).
  Not adopted. Next: synthetic pilot (English-heavy contextual data) + second seed for variance.
- **label_style.py numeral fixes (2026-11-09; found by the synthetic-text agent):** 一万五 → 15000 (was 10005), estimates
  kept as characters (一两天 / 三四十个 / 六七十 / 上万 / 千人千面 were wrongly turned into 12 / 40 / 70 / 10000 / 1000),
  第三方 / 第一次 kept, place / brand names kept (五道口, 七天酒店, 茶百道). 14 existing labels change (users' val 1,
  production train 13; dictionary benchmark 0); J1 / J2 trained on the old forms (13 of 2,253 clips). e2e-val and
  e2e-prod review packs rebuilt with the fixed prefill (review progress is kept per item in the browser).
- **References for the end-to-end model (owner, 2026-11-09):** the ASR part (strict transcript) is scored against
  review column D (`transcript`), the final output against column E (`final_output`); vocabulary corrections belong
  to the final output. scripts/context_metrics.py reports target-word recall against both. Decoding fix: a last
  25-s chunk shorter than 5 s is merged into the previous chunk (a 1.1-s silent tail made J1 / J2 invent Chinese
  text on a 26.1-s English clip; decoding.min_tail_sec).
- **Dictionary benchmark D-column fixes (owner, 2026-11-09):** two hearing fixes the owner had made only in the
  final-output box were synced to the strict transcript: b2978a73 经验 → 经历, 710047f7 "Raymond" → "Raemon" (last
  occurrence; the clip spells R-A-E-M-O-N). Logged in data/private/e2e/reviewed/e2e-dict_D_fixes.json. Review page
  rule going forward: hearing corrections go into box ① (strict transcript); box ② is only for final-output edits.
- **Synthetic data QC (owner approved 2026-11-10):** round trip with Qwen3-ASR-1.7B zero-shot; style differences
  relaxed (numerals, English number words, spelled letters with / without hyphens, split compounds); fail = MER
  > 25% or empty, listen = 10–25% or a term not recognised, pass otherwise (scripts/synth/qc.py). Pilot 500:
  pass 426 / listen 67 / fail 7, pooled 1.73%. Fails are mostly names zero-shot Qwen does not know (Ollama, Bluesky,
  Labubu…), kept; only synth-pilot-0315 dropped (HK-accent voice read 我的 as Cantonese 我哋; data/private/synth/
  qc_drop.json). Owner: augmentation light and on 15% of clips only; long dictations 100 × ~45 s (not 400).
- **Run J3 = qwen3asr_J3 submitted (2026-11-10, owner: "use MiniMax to generate more audio … beat production"):**
  J1 data (policy v1) + J2 cursor-context augmentation + light vocabulary augmentation (--vocab-mild: label term +
  3–10 distractors only when the label has a term, distractors-only 10%) + 100 whole real long recordings (30–180 s)
  + MiniMax synthetic data: pilot 499 (1.15 h) + long 100 (~45 s each, 1.22 h) + more 1,500 (3.48 h), 15% lightly
  augmented (17 / 222 / 72 clips), QC on the cluster (TTS misreadings dropped, unknown-name fails kept). 5,136 training
  rows (4,452 unique; synthetic ≈ 41%). Batch 4 × grad_acc 4. Same selection rule and test sets as J2; label_style
  also keeps 四周 / 二次… as characters. scripts/run_qwen3_asr_J3.sh.
- **qwen3asr_J3 result (2026-11-10, reports/qwen3asr_J3.md):** step 600. Best on the dictionary benchmark (10.79%,
  recall 80%), the owner's freeform test (6.50% no context) and benchmark v1 ≤ 30 s (7.71% vs production 9.08%);
  benchmark v1 overall 9.44–9.54% (production 7.63%) because long English clips stay at ≈ 10.7% (production 6.58%).
  Decoding window checked on validation only: 25 s ≈ 60 s, whole clip much worse → keep 25 s. Post-hoc combination
  (J3 ≤ 30 s, Whisper I2 > 30 s) ties production on benchmark v1 (7.57% vs 7.63%, n.s.).
- **Label fix (owner, 2026-11-10):** "DeepTV" → "LibTV" in two online_pool batch-1 training labels
  (online_pool-7a6cc1280792c28a, online_pool-9b06659707219adf), all manifest versions; found while listening to
  concatenated sample concat-samespeaker-0083; log data/private/label_fixes_2026-11-10.json. J1–J3 were trained on the
  wrong form; J4 uses the fixed labels.
- **Zero-shot Qwen3-4B post-processing rejected (2026-11-10, reports/summary_vs_production_2026-11-10.md §3):** with
  the production prompts it rewrites and sometimes answers the dictated question; benchmark v1 9.44% → 14.29%,
  dictionary vs owner finals 11.50% → 12.23%. Post-processing stays on the fine-tuned-student plan.
- **Run J4 = qwen3asr_J4 submitted (2026-11-10):** J3 recipe + 300 MiniMax acronym sentences (RL, ARR, RLHF, … read
  letter by letter inside Chinese) + 150 long English dictations (median 38.7 s, 1.67 h) + 300 concatenated real long
  clips (30–60 s, 3.09 h; scripts/synth/concat_long.py) + the LibTV label fix. 5,886 training rows (5,202 unique).
  Extra test: the two production clips with spoken acronyms (adhoc_mianmian), with / without a vocabulary.
- **Blind A/B preference review (owner, 2026-11-10):** production final output vs J3 + Qwen3-4B post-processing on
  100 benchmark v1 clips (scripts/build_ab_review.py; key kept outside the page; scored by scripts/ab_score.py).
- **qwen3asr_J4 result (2026-11-11, reports/qwen3asr_J4.md):** step 1101; benchmark v1 9.33% (J3 9.54%); long zh-en
  13.7 → 10.6% (concatenated real long clips work), long English unchanged (10.2% vs production 6.1%); dictionary
  benchmark 10.12% (best); ARR recognised with a vocabulary, RL still not. Routing long English to Whisper I2 and the
  rest to J4 (decided by duration + J4's detected language) ties production on benchmark v1 (7.70% vs 7.63%, n.s.).
- **Dictionary benchmark recheck + arena re-upload package (owner, 2026-11-12):** the arena copy still had the
  pre-review labels. Owner reviewed the 4 never-reviewed items + 2a176c9b (data/private/e2e/reviewed/
  e2e-dict-recheck-review.csv): 6f919ffa / b7f1cbf8 ok, 07a559af / 526c9616 drop, 2a176c9b "stella" → "stalla"
  (matches its dictionary word). data/manifests/dictionary_bench.jsonl 201 → 199 (previous version
  dictionary_bench.pre_recheck.jsonl); scores before this date are on 201 clips. Upload package
  data/private/dictionary_bench/upload/ (199 rows, transcript = strict transcript, original production ogg
  re-downloaded; 3 Android clips whose production ogg is sped up 3× — 3a68ed10, 3bb81e3f, af83babc — encoded from the
  reviewed wav instead), built by scripts/dictionary_bench_upload.py. Do not re-run
  scripts/dictionary_bench_to_manifest.py: it rebuilds from the pre-review labels.
- **Benchmark v2.0 eval of qwen3asr_J3 step 600 with context (2026-11-12, reports/arena/bv2_0_J3_vs_voicecursor.md):**
  decoded on AutoDL (4090 D). Overall 4.54% vs VoiceCursor streaming 7.48%, but 641 / 911 clips are J3 training clips
  and 57 are its val clips, so only the 213 held-out clips count: 8.82% vs 9.05% (−0.24 [−2.34, +1.70], n.s.). Held-out
  English (benchmark_v2 en, 151 clips) J3 worse 9.61 vs 7.56; recording_v2_test better (zh-en 1.68 vs 25.14). On the 57
  val clips J3 is worse (5.91 vs 3.92, significant).
