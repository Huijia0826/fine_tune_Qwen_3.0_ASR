# Data collection plan: hard scenarios from production traffic (owner approved 2026-10-31, see docs/decisions.md)

Why: lora_v2/v3 showed that more ASCEND-like data does not move users' benchmark v1 (docs/research_notes.md). The
literature says target-domain data matters most and saturates after a few hours. So: collect users' dictation audio
for the hard scenarios, with **a bigger locked test set first** (so we can measure ~1-point changes), then training data.

Tool: trace-lookup `bench.py` (build → draft-gt → human review → apply-review). It filters by spoken language, mode,
platform, app, duration and quality flags, with max 3 clips per user. It has **no zh-en / term / self-correction
filter**, so scenarios are assigned by our own rules on the candidate pool (below), then confirmed by a human.

## Scenarios (hard, by evidence)

| # | Scenario | Evidence | Selection rule on the pool (production ASR text) |
|---|---|---|---|
| S1 | zh-en code-switch | worst slice: benchmark v1 15.83% vs VoiceCursor 8.35% | ≥ 1 Latin word and ≥ 30% CJK characters |
| S2 | pure Mandarin | 12.28% vs 6.38% | no Latin words |
| S3 | technical terms / names in Chinese speech | spot checks: Hawking Face, Matrix AI, Mistral, Baba Is AI; recognition errors dominate (reports/cleaning_diagnostic.md) | S1 rule + a Latin token that is an acronym, CamelCase, brand/product, or in a term list |
| S4 | very short clips (≤ 3 s) | < 5 s slice 22% before language fixes; 27% of ASCEND baseline errors were short zh clips detected as en | duration ≤ 3 s, any language mix |
| S5 | self-correction & dictation commands | plan.md scenario; references delete self-corrections and turn "第一个…" into lists | keywords: 不对 / 不是 / 我是说 / 改成 / 换行 / 第一 / sorry / I mean / actually, or flag `pp_shrink` |

Primary label is exclusive (S3 before S1; S4 and S5 before S1/S2). Rules are coarse; the reviewer confirms the label.

## Target counts

Speakers = production user ids, **disjoint across test / val / train**, max 3 clips per user.

| Set | S1 zh-en | S2 zh | S3 terms | S4 short | S5 self-corr. | Total | ≈ hours |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Locked test** | 300 (≥ 100 spk) | 200 (≥ 70 spk) | 150 | 150 | 100 | **900** | 2–2.5 |
| Val | 80 | 50 | 30 | 20 | 20 | **200** | ~0.5 |
| Train | 800 | 450 | 250 | 150 | 150 | **1,800** | ~5 |

Why these numbers:
- **Test S1 300 / ≥ 100 speakers**: benchmark v1's zh-en slice (63 clips, 34 speakers) gives a ±1.5-point interval.
  Interval width shrinks ≈ 1/√speakers, so 100 speakers → ≈ ±0.9 points, enough to see the 1.5-point zh-en signal that
  lora_v2 and lora_v3 both showed. S2 200 / 70 speakers → ≈ ±1.4 (now ±1.7).
- **S3–S5 100–150 each**: enough to report the slice and to see large effects (≥ 3 points); not for fine comparisons.
- **Train ~5 h**: code-switch adaptation saturates around 1–10 h of in-domain data; 5 h triples users'-style zh/zh-en
  (now ~1.6 h) and keeps review effort bounded.
- Val 200: checkpoint selection on users' data (now 68 clips).

## Phases

1. **Phase 1 (first): locked test S1 + S2 (500) and val (200).** Makes every later experiment measurable. Also gives
   the first users'-domain comparison of base small vs large-v3.
2. **Phase 2: train 1,800 clips (~5 h).** Then lora_v4 (and a large-v3 LoRA if zero-shot results justify it).
3. **Phase 3: test S3–S5 (400).**

Windows: test and train from **different weeks** as well as different users, so the test set also checks time drift.

## Labels

- Style = **cleaned dictation text**, following the `formatted_transcript` rules in docs/annotation_policy.md (owner
  decision 2026-10-31: the system should output cleaned text). Allowed: punctuation, case, removing fillers, applying
  self-corrections, list structure, digits. Not allowed: paraphrase, adding information, changing numbers / negation /
  names / commands. Signed off by the owner 2026-10-31.
- Drafts: **local whisper-large-v3** (owner 2026-11-01; the first round used Gemini, judged too poor).
  The production output stored in the trace (VoiceCursor) is shown next to it in review.csv as a second opinion; rows
  where they disagree most are reviewed first. Caveat: drafts can bias references toward the drafter's wording, so the
  owner listens to every clip, and `gt_source` records accepted-as-is vs edited.
- Review effort ≈ 1–1.5 min per clip: test 900 ≈ 15–22 h, val 200 ≈ 4 h, train 1,800 ≈ 30–45 h. Train could use a
  lighter review (listen only where drafts disagree, plus a 10% random spot check), but the arena spot check showed
  "auto-passed" labels missing content (1 of 3), so test and val get a full review.

## Hygiene / leakage

- Exclude users already in benchmark v1, benchmark_chinese and arena sets from the test set; dedupe audio and text.
- Exclude the onboarding app (`com.voicecursor.android.type`) and `error_*` flags; 1 s ≤ duration ≤ 30 s for train
  (test may go to 60 s).
- PII (private names, phone numbers, addresses): reject or redact.

## Owner decisions (2026-10-31)

1. Production user audio may be used for test and training.
2. Drafts from Gemini, then the owner corrects every label.
3. Reference style = `formatted_transcript` rules (annotation_policy.md).
4. Reviewer: the owner.
5. New training audio may go to AutoDL on the same terms as before.
6. Target counts as above.

Open: the 33 verbatim English references in benchmark v1 (keep, or restyle to cleaned text).

## Capacity check (read-only scans, 2026-10-31)

Scans: `online.py scan --mode dictate --lang zh`, 2026-06-01 → 2026-10-31 (data start 2026-08-31, ~9 weeks), files in
data/private/online_scan/; scenario rules in scripts/scan_scenarios.py. 12,683 requests from 358 users; 217 users have
a zh scenario clip of 1–30 s, and only **82 users code-switch** (~9 new code-switch users / week, ~35 new zh users / week).

**Users, not clips, are the limit**: splits must be user-disjoint, and code-switchers are mostly heavy users. Best
allocation found (code-switch users first: 70% of them to test, ≤ 5 clips per user per scenario in test/val; other users
light → test, heavy → train):

| Set | zh-en (S1+S3) | zh (S2) | short (S4) | self-corr. (S5) |
|---|---|---|---|---|
| test | ~104 clips / 57 spk | ~250 / 124 spk | ~146 / 79 spk | ~10 (too few) |
| val | ~30 / 6 spk | ~60 / 17 spk | ~42 / 16 spk | ~4 |
| train | ~4.3 h / 19 spk | ~4.3 h / 41 spk | ~0.9 h / 40 spk | ~0.2 h |

So the 300-clip / 100-speaker zh-en test is not reachable now; training hours are (≈ 9.5 h available).

## Audio quality filter (owner request, 2026-10-31)

Owner found many clips too low quality to review. scripts/audio_quality.py measures speech level, noise floor / SNR,
clipping and 4–8 kHz vs 0–4 kHz energy (narrowband: Bluetooth call mode, muffled mics). Pass = speech ≥ −35 dBFS,
SNR ≥ 20 dB, clipping ≤ 0.1%, hf ≥ −40 dB: **421 of 690** pass (noisy 140, narrowband 95, quiet 72, clipped 1).
Review package with the 421: data/private/online_phase1/review_online-zh-phase1_hq.zip. The 269 low-quality clips are
kept (data/private/online_phase1/audio_quality.jsonl), not deleted. Open: a clean-only test set over-states accuracy on
real user audio, so low-quality clips should probably become a separate "noisy / narrowband" slice later.
