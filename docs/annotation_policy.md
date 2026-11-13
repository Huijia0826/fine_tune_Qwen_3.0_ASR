# Annotation policy (v1, owner sign-off 2026-11-09; history in docs/decisions.md 2026-11-04 → 11-09)

## raw_transcript = strict transcript (the ASR output we train and score)
- Write what was said, in Simplified Chinese characters; English words in their standard spelling and natural
  case (`GitHub`, `CER`).
- Keep repetitions, self-corrections and slips as spoken: `我们周三，不对，周四开会` (decision 2026-11-05).
- Fillers: remove `嗯` `呃` `um` `uh` (incl. runs like `嗯嗯`). `啊` depends on its role (2026-11-09):
  sentence-final particle that carries meaning → keep (`好啊` `对啊` `是啊` `快点啊`); hesitation filler while
  thinking → remove (`啊，那个我们明天…` → `那个我们明天…`, `然后啊，就是…` → `然后，就是…`).
- Numbers (2026-11-09): Arabic numerals for quantities, money, time, dates, room numbers, versions, phone-like
  strings: `3 个人` `15 块` `下午 3 点` `第 2 版` `1806` `15,000`. Keep characters in fixed words and idioms
  (`一下` `一起` `一般` `一样` `十分` `万一` `一定`) and in approximate numbers (`两三个` `十几个` `几百`).
- Spaces (2026-11-09): one space before and after every English word / Latin token inside Chinese text
  (`用 Cursor 写`, `下午 sync 一下`); no space next to Chinese punctuation (`在 Figma 里，F-I-G-M-A。`).
- Spelled-out letters: capitals joined by hyphens, `V-E-R-C-E-L` (decision 2026-11-04); acronyms read as letters
  without hyphens (`API`, `MVP`); read as a word → the word (`NASA`).
- `at` / `dot` in spoken e-mails and URLs as spoken (`lisa at gmail dot com`).
- Punctuation: write natural punctuation (it is stripped for MER, but the product shows it).
- Mark unintelligible segments `<unk>` and set quality_flag `has_unintelligible`; such clips are excluded from test.

## formatted_transcript (dictation target)
Allowed: punctuation, capitalization, removing fillers, applying a spoken self-correction
(`周三，不对，周四` → `周四`), applying a spoken spelling (`换成 Vercel，V-E-R-C-E-L` → `换成 Vercel`; owner 2026-11-02), list/paragraph structure, digits for quantities/dates.
Not allowed: paraphrase, adding information, changing numbers/negation/names/commands,
"improving" wording.

## Review workflow
1. Listen to the whole clip. Check text-audio alignment, truncation, overlap, PII.
2. Set `label_status=human_verified` + `reviewer_id`, or `rejected` with a quality flag.
3. Test-set records must be human_verified. Pseudo-labels (`machine_draft`) never enter any
   split until a human verifies them.
4. PII (names of private persons, phone numbers, addresses): reject or redact the clip;
   do not upload private audio to third-party tools for annotation.
   2026-10-31 the owner allowed Gemini drafts for production clips; 2026-11-01 replaced by local whisper-large-v3
   drafts, so no production audio goes to third-party tools.
