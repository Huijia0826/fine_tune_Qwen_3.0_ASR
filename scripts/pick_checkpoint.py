"""Pick a checkpoint by the predeclared rule (2026-11-02): lowest pooled official MER on users' val =
benchmark_chinese val (68, calibrated) + online_phase1 val (57). Prints the chosen step; ties → earlier step.
Usage: python scripts/pick_checkpoint.py <run name, e.g. lora_large_v3_C> step_600 step_800 … [--refs v1]
--refs v1 (2026-11-09, runs trained on policy-v1 labels, e.g. qwen3asr_J1): references brought to annotation
policy v1 (text/label_style.py) before scoring, so v1-style digits / spacing are not counted as errors.
"""

import json
import sys

from whisperflow.eval.official import official_mer
from whisperflow.text.label_style import to_label_v1

SETS = {"bczh": "data/manifests/benchmark_chinese.jsonl", "online": "data/manifests/online_phase1.jsonl"}


def pooled(prefix: str, v1: bool = False) -> float:
    err = tok = 0
    for tag, man in SETS.items():
        refs = {json.loads(l)["utterance_id"]: json.loads(l)["raw_transcript"]
                for l in open(man, encoding="utf-8") if json.loads(l)["split"] == "val"}
        for line in open(f"reports/runs/{prefix}_{tag}/predictions.jsonl", encoding="utf-8"):
            p = json.loads(line)
            ref = refs[p["utterance_id"]]
            o = official_mer(to_label_v1(ref) if v1 else ref, p["hypothesis"])
            if o.included:
                err, tok = err + o.errors, tok + o.n_ref
    return 100 * err / tok


if __name__ == "__main__":
    args = sys.argv[1:]
    v1 = "--refs" in args and args[args.index("--refs") + 1] == "v1"
    if "--refs" in args:
        del args[args.index("--refs"):args.index("--refs") + 2]
    run, steps = args[0], args[1:]
    scores = {s: pooled(f"sel_{run}_{s}", v1) for s in steps}
    for s, v in scores.items():
        print(f"{s} users' val (bczh + online) {v:.2f}%", file=sys.stderr)
    print(min(steps, key=lambda s: (round(scores[s], 4), int(s.split("_")[1]))))
