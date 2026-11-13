# How it works: the training code

Full-parameter supervised fine-tuning of Qwen3-ASR (audio encoder + Qwen3 LLM decoder, no LoRA). The trainer is the
official Qwen script with settings taken from the config; the parts that matter are below
(`qwen_asr_ft/train.py`).

## 1. Model

`Qwen3ASRModel` (package `qwen-asr`) wraps the trainable Hugging Face model (`.model`) and the processor (`.processor`,
audio features + tokenizer). The forward pass lives in `model.thinker`, so the wrapper's `forward` is patched to call
it; otherwise `Trainer` cannot use the model.

```python
wrapper = Qwen3ASRModel.from_pretrained(base, dtype=torch.bfloat16, device_map=None)
model, processor = wrapper.model, wrapper.processor
patch_outer_forward(model)            # model.forward(...) -> model.thinker.forward(...)
```

## 2. One training example

The chat template builds a prefix: system turn = context, user turn = one audio, then the assistant header. The
target (`language English<asr_text>…`) follows the prefix.

```python
def build_prefix_messages(prompt, audio_array):
    return [{"role": "system", "content": prompt or ""},
            {"role": "user", "content": [{"type": "audio", "audio": audio_array}]}]

prefix_text = processor.apply_chat_template([build_prefix_messages(prompt, None)],
                                            add_generation_prompt=True, tokenize=False)[0]
```

## 3. Loss only on the output text

The collator encodes each batch twice — prefix + target + EOS, and the prefix alone — and sets the labels of the
prefix (and of padding) to -100. The model conditions on context and audio but is only trained to produce the
transcript, never to reproduce the context.

```python
full_inputs   = processor(text=[p + t + eos for p, t in zip(prefixes, targets)], audio=audios, padding=True, ...)
prefix_inputs = processor(text=prefixes, audio=audios, padding=True, ...)
labels = full_inputs["input_ids"].clone()
for i, n in enumerate(prefix_inputs["attention_mask"].sum(dim=1).tolist()):
    labels[i, :n] = -100
labels[labels == processor.tokenizer.pad_token_id] = -100
```

Audio is loaded and resampled to 16 kHz inside the collator, so `train.num_workers` matters for throughput.
Nothing is truncated: long clips cost memory, lower `train.batch_size` if you run out.

## 4. Trainer

A standard Hugging Face `Trainer` with two additions: floating inputs (audio features) are cast to the model dtype,
and every checkpoint gets the tokenizer / processor files copied in, so any checkpoint can be decoded directly.
Main settings (config `train.*`): lr 2e-5, linear schedule, 5% warm-up, effective batch 16 (4 × 4), 3 epochs,
bf16, a checkpoint every 200 steps, optimiser state not saved (`save_only_model`).

## 5. Choosing the checkpoint

`qasr select` decodes `data.val` with every checkpoint and keeps the lowest **pooled** MER (total errors / total
reference tokens; ties → earlier step). The rule is fixed before looking at test results; test sets are decoded
only with the chosen checkpoint (`qasr test`).

## 6. Decoding (`qwen_asr_ft/infer.py`)

- Greedy decoding through `qwen-asr`. `decode.language: auto` lets the model detect the language.
- Audio longer than `decode.max_chunk_sec` (25 s) is cut at a quiet point; a last piece shorter than
  `decode.min_tail_sec` (5 s) is merged into the previous one. Decoding multi-minute audio in one pass drifted or
  looped in our tests, and a near-silent tail made fine-tuned models invent text.
- Chunk texts are joined with a space where needed (`frontend?Well` → `frontend? Well`).
- Optional `decode.copy_guard` cuts an output tail that repeats the cursor text of the context.

## 7. Scoring (`qwen_asr_ft/metrics.py`, `evaluate.py`)

- Tokens: one per Chinese character, one per English word; punctuation and case ignored; Traditional → Simplified.
  MER = (S + D + I) / reference tokens (CER on Chinese, WER on English).
- Pooled over clips, never the mean of per-clip rates.
- 95% intervals resample **speakers**, not clips (clips of one speaker are correlated).
- `qasr compare` gives B − A on the same clips with a paired speaker bootstrap; an interval containing 0 means
  "no evidence of a difference".
