# Candidate data sources and license review (2026-10-27)

Research only: **nothing downloaded.** Facts gathered from primary pages (OpenSLR, dataset
cards, LDC, license files). huggingface.co was unreachable from this network, so HF dataset
cards were read via hf-mirror.com. Items marked UNVERIFIED could not be confirmed from a
primary source. **Licenses must be re-checked by the owner before use; this is not legal advice.**

| Dataset | Type | Hours | Spk IDs | License | Commercial | Size | Suggested role |
|---|---|---|---|---|---|---|---|
| ASCEND (CAiRE/ASCEND) | zh-en CS, spontaneous conversation (HK) | 10.62 | yes (23 spk) + session | CC BY-SA 4.0 | yes (ShareAlike) | 1.2 GB | core code-switch train/val/test (re-split by speaker) |
| FLEURS cmn_hans_cn | Mandarin read | ~12 | **no** spk field (card says train spk ≠ dev/test spk) | CC BY 4.0 | yes | 3.2 GB | Mandarin train/val/test, use official splits |
| Common Voice zh-CN (via Mozilla Data Collective) | Mandarin read, crowd | large; v26 figures UNVERIFIED | yes (hashed client_id) | CC0-1.0 + MDC terms (no re-hosting, no speaker re-identification) | yes | ~21 GB | optional extra Mandarin train |
| AISHELL-1 (SLR33) | Mandarin read | ~178 | yes (400) | Apache-2.0 on OpenSLR vs "free for academic use" on the same pages | **ambiguous** | 15 GB | only if owner accepts ambiguity |
| AISHELL-3, THCHS-30 | Mandarin read | 85 / ~30 | yes | Apache-2.0 (AISHELL-3 same ambiguity) | ambiguous / yes | 19 / 6.4 GB | optional |
| AISHELL-4, AliMeeting | Mandarin meetings | 120 / 119 | yes | CC BY-SA 4.0 | yes | 51 / 23 GB | domain mismatch (far-field meetings) |
| MAGICDATA (SLR68), Primewords, ST-CMDS, MagicData-RAMC | Mandarin read / conversation | 100–755 | mostly yes | CC BY-NC-ND 4.0 | **no**; ND likely bars fine-tuned weights | 8–52 GB | private evaluation only (MAGICDATA test 2.2 GB) |
| CS-Dialogue (BAAI, 2025) | zh-en CS spontaneous dialogue (mainland) | 104 | yes, speaker-disjoint splits | CC BY-NC-SA 4.0 + stricter access terms (academic, no redistribution, derived works stay in group) | **no** | UNVERIFIED | best mainland CS data, private research only |
| aidatatang_200zh | Mandarin read | 200 | yes | retracted by owner | — | — | **do not use** |
| AISHELL-2, KeSpeech, WenetSpeech, SEAME (LDC, paid), TALCS (no longer available), ASRU-2019 CS (no access route found), DOTA-ME-CS (no license) | | | | | | | **not usable / not recommended** |

Notes
- No public zh-en corpus focused on IT/technical vocabulary was found → our `technical_terms`,
  `self_correction`, `dictation_formatting` slices need **owner/volunteer recordings**.
- ASCEND is Hong Kong bilingual speech (accent/domain gap vs mainland dictation). With only
  23 speakers, its val/test slices will be small; CIs will be wide.
- Whisper training-data overlap is **unknown** for every set. FLEURS/Common Voice are standard
  Whisper *eval* sets; useful for comparability but should not be the only test.
- ShareAlike (ASCEND) may affect a public model release — note in model card; decide with owner.

## Proposed staged acquisition (needs approval)
**Stage A (~4.4 GB, permissive licenses, compatible with public release):**
ASCEND (pool and re-split by speaker, `group_by` session) + FLEURS cmn_hans_cn (keep official
speaker-disjoint splits via `--respect-existing`) + 30–60 min of owner-recorded IT-domain
dictation, **test only**, human-verified. Realistic yield ≈ 10 h train / 2 h val / 3 h test
but exact hours will be reported after ingestion, not assumed.
**Stage B:** Common Voice zh-CN subset if more Mandarin is needed (~21 GB download).
**Stage C (private use only):** CS-Dialogue, MAGICDATA test set for extra evaluation.
