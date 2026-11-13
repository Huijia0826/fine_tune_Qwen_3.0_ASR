# Data format

A **manifest** is a JSON Lines file: one clip per line. Only `audio` and `text` are required.

```json
{"id": "clip-001", "audio": "audio/clip-001.wav", "text": "我们周三，不对，周四用 Vercel 部署", "speaker": "u42", "language": "zh"}
{"id": "clip-002", "audio": "/data/asr/clip-002.flac", "text": "Ship it to staging first.", "speaker": "u7",
 "vocabulary": ["Vercel", "PostHog"], "app": "Slack", "tone": "casual", "before_cursor": "Can you deploy today?"}
```

| field | required | meaning |
| --- | --- | --- |
| `audio` | yes | path to any file librosa can read (wav, flac, mp3, ogg…); relative paths are resolved against `data.audio_root`, else the manifest's folder. Resampled to 16 kHz mono on the fly. |
| `text` | yes | the reference transcript = what the model should output |
| `id` | no | unique clip id (default: manifest name + line number) |
| `speaker` | no | speaker / user id. Used for speaker-level confidence intervals and for "previous utterance" augmentation. Default: the clip id (every clip its own speaker). |
| `language` | no | `zh`, `en`, a Qwen language name (`Chinese`, `English`, `Cantonese`, …) or `auto` (default: Chinese if the text contains Chinese characters, else English). Code-switched speech is tagged by its main language. |
| `weight` | no | integer: repeat this clip N times in training (oversampling) |
| `context` | no | free-text context, used as the system prompt as is |
| `app`, `tone`, `vocabulary`, `before_cursor`, `after_cursor` | no | structured context, composed into the prompt (see below). Rename via `data.context.fields`. |
| anything else | no | kept; you can slice results by it (`eval.slice_by: [device, scenario]`) |

Split train / val / test **by speaker**, so no speaker appears in two sets (`scripts/table_to_manifest.py` does this).
Use `val` only to choose the checkpoint; look at `test` once per candidate.

## What the trainer sees

`qasr prepare` turns each manifest row into:

```json
{"audio": "/abs/path/clip-002.flac",
 "prompt": "App: Slack · casual\nVocabulary: Vercel, PostHog\nBefore cursor: Can you deploy today?",
 "text": "language English<asr_text>Ship it to staging first.",
 "id": "clip-002"}
```

- `prompt` goes into Qwen's system turn. Qwen3-ASR treats it as background text that biases recognition (hotwords,
  topic), not as an instruction. Rows without context get an empty prompt, so the model also works without one.
- Cursor text that shares 8+ characters with the label is dropped (`data.context.leak_ngram`): otherwise the model
  learns to copy the answer from the context instead of listening.
- `text` is Qwen3-ASR's output format: `language <Name><asr_text><transcript>`.

## Labelling conventions

The model learns exactly the style of your labels, so pick one convention and apply it everywhere (numbers as digits
or words, fillers kept or removed, spacing between Chinese and English, spelled-out letters…). If your sources
disagree, write a function and set `data.normalize_label: "my_module:normalize"`; it is applied to every training
label. Our experience: keep the ASR target a **verbatim transcript** and do clean-up (removing self-corrections,
formatting) in a separate post-processing step — see LESSONS.md.
