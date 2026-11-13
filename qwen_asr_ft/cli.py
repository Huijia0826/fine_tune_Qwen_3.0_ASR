"""`qasr` command line. Every command takes -c/--config and any number of --set key.sub=value overrides.

    qasr prepare  -c cfg.yaml                    manifests -> <work_dir>/sft/{train,val}.jsonl
    qasr train    -c cfg.yaml                    fine-tune -> <work_dir>/train/checkpoint-*
    qasr select   -c cfg.yaml                    decode val with every checkpoint, keep the best -> best.txt
    qasr test     -c cfg.yaml [--model PATH]     decode + score every data.test set once
    qasr pipeline -c cfg.yaml                    prepare -> train -> select -> test
    qasr transcribe -c cfg.yaml --model PATH --manifest m.jsonl --out preds.jsonl
    qasr score    -c cfg.yaml --manifest m.jsonl --preds preds.jsonl [--out DIR]
    qasr compare  -c cfg.yaml --manifest m.jsonl --a A.jsonl --b B.jsonl
"""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

from .config import load_config


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001
        pass


def cmd_prepare(cfg, _args):
    from .data import prepare

    print(json.dumps(prepare(cfg), indent=2, ensure_ascii=False))


def cmd_train(cfg, _args):
    from .train import train

    print(f"[train] checkpoints in {train(cfg)}")


def base_model_path(cfg) -> str:
    from .train import resolve_base

    return resolve_base(cfg["model"])


def cmd_select(cfg, args):
    from .evaluate import mer_of, pick_best
    from .infer import transcribe_manifest
    from .train import checkpoints

    work = Path(cfg["work_dir"])
    ckpts = checkpoints(work / "train")
    if not ckpts:
        raise SystemExit(f"no checkpoints in {work / 'train'}")
    if not cfg["data"]["val"]:
        best = ckpts[-1]
        print(f"[select] no data.val: using the last checkpoint {best}")
    else:
        scores = {}
        for ck in ckpts:
            preds = work / "select" / ck.name / "preds.jsonl"
            transcribe_manifest(str(ck), cfg["data"]["val"], preds, cfg)
            _free_gpu()
            scores[ck.name] = mer_of(cfg["data"]["val"], preds, cfg)
            print(f"[select] {ck.name}: val MER {scores[ck.name]:.2f}%", flush=True)
        if getattr(args, "include_base", False):
            preds = work / "select" / "base" / "preds.jsonl"
            transcribe_manifest(base_model_path(cfg), cfg["data"]["val"], preds, cfg)
            print(f"[select] base (reference only): val MER {mer_of(cfg['data']['val'], preds, cfg):.2f}%")
        best = work / "train" / pick_best(scores)
        (work / "select" / "scores.json").write_text(json.dumps(scores, indent=2), encoding="utf-8")
    (work / "best.txt").write_text(str(best.resolve()) + "\n", encoding="utf-8")
    print(f"[select] best checkpoint: {best}")


def cmd_test(cfg, args):
    from .evaluate import evaluate, to_markdown
    from .infer import transcribe_manifest

    work = Path(cfg["work_dir"])
    tests = cfg["data"].get("test") or {}
    if not tests:
        raise SystemExit("data.test is empty")
    systems = {}
    if args.model:
        systems[Path(args.model).name] = args.model
    else:
        best = work / "best.txt"
        if not best.exists():
            raise SystemExit("no best.txt: run `qasr select` first or pass --model")
        systems["finetuned"] = best.read_text().strip()
    if args.with_base:
        systems["base"] = base_model_path(cfg)
    for sys_name, path in systems.items():
        for name, manifest in tests.items():
            out = work / "test" / sys_name / name
            preds = transcribe_manifest(path, manifest, out / "preds.jsonl", cfg)
            _free_gpu()
            print(to_markdown(evaluate(manifest, preds, cfg, out)))


def cmd_pipeline(cfg, args):
    for step in (cmd_prepare, cmd_train, cmd_select):
        step(cfg, args)
    if cfg["data"].get("test"):
        cmd_test(cfg, args)


def cmd_transcribe(cfg, args):
    from .infer import transcribe_manifest

    model = args.model or base_model_path(cfg)
    print(f"[transcribe] -> {transcribe_manifest(model, args.manifest, args.out, cfg)}")


def cmd_score(cfg, args):
    from .evaluate import evaluate, to_markdown

    print(to_markdown(evaluate(args.manifest, args.preds, cfg, args.out)))


def cmd_compare(cfg, args):
    from .evaluate import compare

    print(json.dumps(compare(args.manifest, args.a, args.b, cfg), indent=2))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="qasr", description="Fine-tune and evaluate Qwen3-ASR")
    sub = ap.add_subparsers(dest="cmd", required=True)

    def add(name, fn, **extra):
        p = sub.add_parser(name)
        p.add_argument("-c", "--config", help="YAML config (defaults are used for anything not set)")
        p.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override, e.g. train.lr=1e-5")
        for flag, kw in extra.items():
            p.add_argument(flag, **kw)
        p.set_defaults(fn=fn)
        return p

    add("prepare", cmd_prepare)
    add("train", cmd_train)
    add("select", cmd_select, **{"--include-base": {"action": "store_true", "dest": "include_base"}})
    add("test", cmd_test, **{"--model": {"default": None}, "--with-base": {"action": "store_true", "dest": "with_base"}})
    add("pipeline", cmd_pipeline, **{"--model": {"default": None},
                                     "--with-base": {"action": "store_true", "dest": "with_base"},
                                     "--include-base": {"action": "store_true", "dest": "include_base"}})
    add("transcribe", cmd_transcribe, **{"--model": {"default": None}, "--manifest": {"required": True, "nargs": "+"},
                                         "--out": {"required": True}})
    add("score", cmd_score, **{"--manifest": {"required": True, "nargs": "+"}, "--preds": {"required": True},
                               "--out": {"default": None}})
    add("compare", cmd_compare, **{"--manifest": {"required": True, "nargs": "+"}, "--a": {"required": True},
                                   "--b": {"required": True}})
    args = ap.parse_args(argv)
    cfg = load_config(args.config, args.set)
    try:
        args.fn(cfg, args)
    except ModuleNotFoundError as e:
        if e.name and e.name.split(".")[0] in {"torch", "qwen_asr", "transformers", "datasets", "librosa"}:
            raise SystemExit(f"missing package '{e.name}': training / decoding needs the GPU extras — "
                             "run `bash scripts/setup_env.sh` or `pip install -e '.[train]'`") from e
        raise


if __name__ == "__main__":
    main()
