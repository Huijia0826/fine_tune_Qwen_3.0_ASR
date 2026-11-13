# Synthetic contextual dictation data (~10k clips) — plan (draft 2026-11-09, needs owner decisions)

Goal (owner, 2026-11-09): train the end-to-end model (audio + context → strict transcript + final output) on ~10k
synthetic clips that look like the owner's dictionary benchmark — product / people / place names that need the
vocabulary, and final outputs that follow the app → tone rules — without using the benchmark itself (it stays a
pure test set; its 320 target words are banned from any training text).

## What the owner's review of the dictionary benchmark says the targets must do (197 reviewed clips)
| tone | n | final output vs strict transcript |
|---|---:|---|
| casual | 92 | 56 only drop the sentence-final punctuation; 29 identical (no final mark / question); 7 other edits |
| formal | 13 | 12 identical (keep the period); 1 tightened |
| neutral | 51 | 43 identical; tightening or formatting otherwise |
| unknown | 41 | 38 identical |
Other edits: light tightening (然后 / 就是 / 些 removed), formatting (e-mail line breaks, quotes for reported speech);
name fixes (AppsFlyer, Whop) went into the strict transcript.

## Pipeline (one sample = audio, context, strict transcript, final output)
1. **Scenario**: app sampled from configs/reference/tone_by_app.tsv weighted by production traffic → tone; language
   mix zh / zh-en / en (weights to decide; production dictation with post-processing is mostly en, our users' val is
   zh / zh-en).
2. **Terms**: 1–3 target terms per sample from a term bank — brands / products / tech words / people and place names
   (public lists + our glossary + de-identified terms from production dictionaries), never a dictionary-benchmark
   target word. Some samples get no term (the model must still work).
3. **Text**: an LLM writes a spoken-style sentence for the scenario using the terms (code-switching, fillers removed per
   policy v1, some self-corrections / repeats / spelled letters as in real speech) = **strict transcript**; plus
   plausible **before / after cursor** text for that app (e.g. the previous chat message, the e-mail being written).
4. **Context**: Vocabulary = the target terms mixed with distractors (incl. look-alikes) in random order, or
   distractors only, or empty; cursor text as generated, or another sample's, or none (same proportions as J2).
5. **Final output**: rules first — casual → drop only the sentence-final period; formal / neutral → keep; e-mail apps →
   greeting / sign-off line breaks; term casing as in the vocabulary — then light tightening for a share of samples
   (owner-approved examples as few-shot). Rule-generated, so it is consistent.
6. **Audio**: TTS of the strict transcript with many voices; mic / room / noise augmentation; 16 kHz.
7. **QC**: decode every clip with Qwen3-ASR zero-shot (+ its context); drop clips whose non-term words disagree a lot
   (bad TTS) or whose terms are mispronounced (English names in Chinese TTS are the main risk). Owner listens to ~30
   random clips of the pilot.
8. **Training mix**: synthetic is never the majority per epoch at first (e.g. 1:1 with the 2.2k real clips, then
   ablate); every evaluation stays on real audio only (dictionary benchmark, users' val, owner tests, benchmark v2,
   e2e-val final outputs once reviewed).

## Decisions for the owner
1. **TTS engine**: open-source on our H100 (CosyVoice 2 / 3: Chinese–English code-switching and zero-shot voice
   cloning; licence to verify before use) vs a paid API (e.g. ElevenLabs / MiniMax: better English names, costs money,
   text leaves our machines). Recommendation: open-source first, API only for a hard-term subset if QC fails.
2. **Voices**: never clone production users. Options: TTS built-in speakers, public corpora with permissive licences
   (e.g. AISHELL-3 / LibriTTS speakers — licences to verify), the owner's own consented recordings, colleagues who
   agree.
3. **Text generator**: a local open model on the H100 (e.g. a Qwen3 instruct model via vLLM: nothing leaves our
   machines) vs an API model (Gemini / Claude: better text; only the term bank and scenario go out, never production
   context). Recommendation: API for quality if the owner allows sending the term bank, else local.
4. **Size and language mix**: pilot 500 → owner listens to 30 → 10k (≈ 15–20 h of audio; ≈ 2–3 GPU hours of TTS).
5. **Term bank sources**: may de-identified words from production user dictionaries be used (they can contain
   personal names)?
