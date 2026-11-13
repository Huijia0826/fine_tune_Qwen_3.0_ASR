"""Turn your existing data into a manifest (docs/DATA_FORMAT.md).

From a CSV / TSV / JSONL with at least an audio column and a text column:
    python scripts/table_to_manifest.py my_data.csv --audio-col path --text-col transcript \
        [--speaker-col user_id] [--extra-cols app,vocabulary] [--split-col split] --out data/

From a folder of audio files with a same-named .txt next to each (a.wav + a.txt):
    python scripts/table_to_manifest.py --folder recordings/ --out data/

Without a split column, clips are split by SPEAKER (no speaker in two splits): --val 0.1 --test 0.1 (fractions of
speakers). Writes <out>/train.jsonl, val.jsonl, test.jsonl with audio paths relative to <out>.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from pathlib import Path

AUDIO_EXT = {".wav", ".flac", ".mp3", ".ogg", ".m4a", ".opus", ".webm"}


def from_table(path: Path, a: argparse.Namespace) -> list[dict]:
    if path.suffix == ".jsonl":
        rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    else:
        delim = "\t" if path.suffix in (".tsv", ".tab") else ","
        rows = list(csv.DictReader(open(path, encoding="utf-8-sig"), delimiter=delim))
    out = []
    extra = [c for c in (a.extra_cols or "").split(",") if c]
    for i, r in enumerate(rows):
        audio = Path(str(r[a.audio_col])).expanduser()
        if not audio.is_absolute():
            audio = (path.parent / audio).resolve()
        item = {"id": str(r.get(a.id_col) or f"{path.stem}-{i:06d}"), "audio": str(audio), "text": str(r[a.text_col])}
        if a.speaker_col and r.get(a.speaker_col):
            item["speaker"] = str(r[a.speaker_col])
        if a.language_col and r.get(a.language_col):
            item["language"] = str(r[a.language_col])
        if a.split_col and r.get(a.split_col):
            item["split"] = str(r[a.split_col])
        for c in extra:
            if r.get(c) not in (None, ""):
                item[c] = r[c]
        out.append(item)
    return out


def from_folder(folder: Path) -> list[dict]:
    out = []
    for p in sorted(folder.rglob("*")):
        if p.suffix.lower() in AUDIO_EXT and p.with_suffix(".txt").exists():
            rel = p.relative_to(folder).with_suffix("")
            out.append({"id": "-".join(rel.parts), "audio": str(p.resolve()),
                        "text": p.with_suffix(".txt").read_text(encoding="utf-8").strip(),
                        "speaker": p.parent.name})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("table", nargs="?")
    ap.add_argument("--folder")
    ap.add_argument("--audio-col", default="audio")
    ap.add_argument("--text-col", default="text")
    ap.add_argument("--id-col", default="id")
    ap.add_argument("--speaker-col", default="speaker")
    ap.add_argument("--language-col", default="language")
    ap.add_argument("--split-col", default="split")
    ap.add_argument("--extra-cols", default="")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--test", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if bool(a.table) == bool(a.folder):
        ap.error("give either a table file or --folder")
    rows = from_folder(Path(a.folder)) if a.folder else from_table(Path(a.table), a)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    if not any("split" in r for r in rows):
        spk = sorted({r.get("speaker", r["id"]) for r in rows})
        random.Random(a.seed).shuffle(spk)
        n_val, n_test = int(len(spk) * a.val), int(len(spk) * a.test)
        which = {s: "val" for s in spk[:n_val]} | {s: "test" for s in spk[n_val:n_val + n_test]}
        for r in rows:
            r["split"] = which.get(r.get("speaker", r["id"]), "train")
    counts = {}
    for split in ("train", "val", "test"):
        part = [dict(r, audio=os.path.relpath(r["audio"], out)) for r in rows if r.get("split") == split]
        with open(out / f"{split}.jsonl", "w", encoding="utf-8") as f:
            for r in part:
                r.pop("split", None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[split] = len(part)
    print(json.dumps(counts))


if __name__ == "__main__":
    main()
