# Dataset specification (schema v1.0)

Implemented in `src/whisperflow/data/schema.py` (Pydantic, `extra="forbid"`).

## Manifest record (JSONL, one utterance per line)

| field | type | rule |
|---|---|---|
| utterance_id | str | `[A-Za-z0-9][A-Za-z0-9_.-]{0,127}`, unique. Ingest assigns `<source_id>-<sha256[:16]>` (content-addressed). |
| audio_path | str | relative to `data_root`; no absolute paths, no `..` |
| sha256 | hex64 | verified against file by `wf-validate` |
| duration_sec | float > 0 | must match decoded audio within 0.05 s |
| speaker_id, session_id | id | namespaced by source at ingest (`aishell1-S0002`) so IDs from different corpora never collide |
| source_id, license_or_consent_id | id | must resolve in `configs/sources.yaml`, source `approved`, split allowed |
| split | enum | train · val · test · unassigned |
| raw_transcript | str | verbatim; must contain ≥1 scoreable token |
| formatted_transcript | str? | desired dictation output (separate label) |
| primary_scenario | enum | zh_general · zh_en_switch · technical_terms · self_correction · dictation_formatting (mutually exclusive) |
| language_tags | [zh,en,other] | non-empty; zh_en_switch requires en |
| hotwords_present | [str] | each should occur in raw_transcript (warning otherwise) |
| recording_device, noise_tag | str / enum | noise: quiet · moderate · noisy · unknown |
| label_status | enum | source_provided · machine_draft · human_verified · rejected |
| reviewer_id | str? | required when human_verified |
| quality_flags | [str] | free tags (e.g. `scenario_inferred`, `converted_16k_mono`) |
| source_utterance_id, parent_utterance_id, sample_rate_hz, num_channels | optional | traceability; parent links augmented variants to their source |

## Error vs warning
**Errors block** (exit 1): schema violations, duplicate IDs/paths, unknown/unapproved source,
split not allowed by license, missing/unreadable/empty/silent audio, hash or duration mismatch,
rejected records inside a split, any cross-split leakage.

**Warnings require review**: language tags vs transcript mismatch, hotword not in transcript,
>30 s, <0.5 s, implausible tokens/sec (>12 or <0.5), sample rate ≠16 k, multichannel, clipping,
>3 s leading/trailing silence, machine_draft in a split, non-human-verified test record,
transcripts shared across splits.

Known gap: the validator cannot tell whether audio is *speech* or whether it *matches* the
transcript. Planned: flag clips whose baseline-model CER is extreme for human re-review
(never auto-correct labels from model output).

## Splits
1. **Atomic groups via union-find** over `speaker_id`, `session_id`, `parent_utterance_id`
   (augmentation lineage) and `sha256`. A two-speaker session merges both speakers; an
   augmented copy follows its parent. Optionally `group_identical_transcripts` for read
   corpora that reuse prompts (stricter, can over-merge).
2. **Eligibility:** only `source_provided`/`human_verified` records are assigned;
   `machine_draft`/`rejected` stay `unassigned`.
3. **Test lock:** `wf-split --lock-test` writes `data/manifests/locked_test_groups.json`
   (speakers, sessions, utterance IDs + digest). Afterwards: locked speakers' new data goes
   to test, new speakers go only to train/val (unless `--allow-new-test-groups`), and any
   locked utterance leaving test is an error. The test set can only change deliberately.
4. **Stratified greedy:** groups sorted largest-first (seeded hash tie-break) and placed
   where they most reduce squared deviation from per-(split, scenario) hour targets.
   Targets 10/2/3 h are scaled proportionally to available data; shortfall is reported
   as `insufficient_data`, never padded.
5. **Official corpus splits** can be kept with `--respect-existing`; it errors if they
   conflict with our grouping (i.e. the official split would leak a speaker/session).
6. **Independent audit:** `wf-audit` re-checks leakage on the output (speaker, session,
   audio hash, augmentation lineage = errors; shared transcript = warning).

Why speaker-disjoint matters: Whisper adapts to voice/channel quickly; a test set sharing
speakers with training measures memorisation of voices, not generalisation, and inflates
gains from fine-tuning — exactly the comparison we care about.

## Normalization (scoring only)
Policy `zh_en_v1` — see `docs/decisions.md` D4. Original labels are never modified; scoring
works on normalized copies.
