#!/usr/bin/env bash
# Fetch ONLY the tokenizer/config files of whisper-small (~4.4 MB, no weights) at a pinned
# commit. Used by the optional real-tokenizer test. Set HF_ENDPOINT if huggingface.co is
# unreachable (e.g. HF_ENDPOINT=https://hf-mirror.com) — see docs/decisions.md D7.
set -euo pipefail
REV="${REV:-973afd24965f72e36ca33b3055d56a652f456b4d}"
uv run hf download openai/whisper-small --revision "$REV" \
  added_tokens.json config.json generation_config.json merges.txt normalizer.json \
  preprocessor_config.json special_tokens_map.json tokenizer.json tokenizer_config.json vocab.json
