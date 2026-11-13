# Fine Tune Qwen 3.0 ASR

A small, configurable framework to fine-tune [Qwen3-ASR](https://huggingface.co/Qwen/Qwen3-ASR-1.7B) on your own
speech data and to measure whether the result is actually better. Bring a list of `(audio, transcript)` pairs; the
framework handles prompt building, training, checkpoint selection, long-audio decoding and scoring.

- **Any data**: one JSON line per clip; field names, languages and paths are set in a YAML config, nothing is hard-coded.
- **Contextual biasing**: optional per-clip context (hotword vocabulary, app, text around the cursor) goes into
  Qwen's system prompt, with training-time augmentation so the model uses hotwords without copying or inventing them.
- **Honest evaluation**: mixed Chinese / English error rate (MER), per-slice results, speaker-level bootstrap
  confidence intervals, paired comparison between systems, checkpoint chosen on validation only.

## Quick start (about 10 minutes on one GPU)

```bash
git clone https://github.com/Huijia0826/fine_tune_Qwen_3.0_ASR.git && cd fine_tune_Qwen_3.0_ASR
bash scripts/setup_env.sh .venv && source .venv/bin/activate     # Linux + NVIDIA GPU
pytest                                                            # CPU tests, should all pass

python examples/make_toy_dataset.py --out examples/toy           # 73 public English clips
qasr pipeline -c examples/toy.yaml                                # prepare -> train 5 steps -> select -> test
cat runs/toy/test/finetuned/toy_test/metrics.md
```

If that works, your environment is fine. Then use your own data:

```bash
# 1. your data -> manifests (CSV / TSV / JSONL table, or a folder of a.wav + a.txt)
python scripts/table_to_manifest.py my_data.csv --audio-col path --text-col transcript --speaker-col user --out data/
# 2. copy and edit the config (paths, field names, hyper-parameters)
cp configs/example.yaml configs/my_run.yaml
# 3. run everything, or step by step
qasr pipeline -c configs/my_run.yaml --with-base      # also scores the base model on the test sets for comparison
```

## Commands

| command | what it does | output (under `work_dir`) |
| --- | --- | --- |
| `qasr prepare -c cfg` | manifests → Qwen SFT rows (language tag, context, label) | `sft/train.jsonl`, `sft/val.jsonl`, `sft/summary.json` |
| `qasr train -c cfg` | full-parameter fine-tuning | `train/checkpoint-*` |
| `qasr select -c cfg` | decode `data.val` with every checkpoint, keep the lowest MER | `select/scores.json`, `best.txt` |
| `qasr test -c cfg [--with-base]` | decode and score each `data.test` set once | `test/<system>/<set>/metrics.md`, `per_clip.csv` |
| `qasr pipeline -c cfg` | the four steps above | |
| `qasr transcribe -c cfg --model M --manifest m.jsonl --out p.jsonl` | decode any manifest with any model | predictions jsonl |
| `qasr score -c cfg --manifest m.jsonl --preds p.jsonl [--out DIR]` | score any predictions (also from other systems) | metrics + per-clip CSV |
| `qasr compare -c cfg --manifest m.jsonl --a A.jsonl --b B.jsonl` | B − A with a paired 95% interval | JSON |

Any config value can be overridden on the command line: `--set train.lr=1e-5 --set decode.max_chunk_sec=30`.
On a Slurm cluster: `sbatch scripts/slurm_pipeline.sbatch configs/my_run.yaml`.

## Repository layout

```
qwen_asr_ft/      the package (config, data, context, train, infer, evaluate, metrics, text, cli)
configs/          example.yaml: every option, with comments
scripts/          setup_env.sh, table_to_manifest.py, slurm_pipeline.sbatch
examples/         toy dataset + smoke-test config
docs/             DATA_FORMAT.md · HOW_IT_WORKS.md (training code walk-through) · LESSONS.md (what worked for us)
tests/            CPU-only tests
```

## Requirements

- Training / decoding: Linux, NVIDIA GPU with bf16 (A100 / H100 / RTX 30xx+). Qwen3-ASR-1.7B full fine-tuning with
  batch 4 fits in 80 GB; reduce `train.batch_size` (and raise `grad_acc`) on smaller cards.
- `qwen-asr==0.0.6` pins `transformers==4.57.6`; keep it in its own environment.
- Scoring, data preparation and tests run anywhere (no GPU, no model download).

`qwen_asr_ft/train.py` is adapted from the official Qwen3-ASR fine-tuning script (Apache-2.0); see its header.
