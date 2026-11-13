"""ASCEND adapter (exercise 1, see docs/learning/exercises.md): parquet -> WAV files + wf-ingest TSV.

Only translates the corpus format. Hashing, IDs, 16 kHz checks etc. happen in wf-ingest.

Spec: write each row as a .wav under audio_out and one TSV row in meta_out; skip "[UNK]" rows;
TSV columns audio_path (relative to audio_out), transcript, speaker_id, session_id,
source_utterance_id; return {"n_total", "n_skipped_unk", "n_written"}.
"""

import csv
from pathlib import Path

import pandas as pd

from whisperflow.text.fillers import strip_fillers as remove_fillers
from whisperflow.text.numbers import en_numbers_to_digits


def export_ascend(parquet:list[Path],audio_out: Path, meta_out: Path,strip_fillers: bool = False,
                  en_digits: bool = False) -> dict:
    audio_out.mkdir(parents=True, exist_ok=True)
    with open(meta_out, "w", encoding="utf-8", newline="") as f:          # 打开文件
        writer = csv.DictWriter(
            f,
            fieldnames=["audio_path", "transcript", "speaker_id", "session_id", "source_utterance_id"],
            delimiter="\t",
        )
        writer.writeheader()


        stats = {"n_total": 0, "n_skipped_unk": 0, "n_written": 0, "n_empty_after_fillers": 0}
        for line in parquet:

            df = pd.read_parquet(line)
            for i in range(len(df)):
                row = df.iloc[i]
                stats["n_total"] += 1
                if '[UNK]' in row['transcription']:
                    stats["n_skipped_unk"] += 1
                    continue
                text = row['transcription']
                if strip_fillers:
                    text = remove_fillers(text)
                    if not text.strip():
                        stats["n_empty_after_fillers"] += 1
                        continue
                if en_digits:  # after fillers, so "forty uh five" -> "forty five" -> "45"
                    text = en_numbers_to_digits(text)
                name = row['audio']["path"]
                (audio_out / name).write_bytes(row['audio']['bytes'])
                writer.writerow({                                              # 每写出一个音频，就写一行表格
                    "audio_path": name,
                    "transcript": text,
                    "speaker_id": row['original_speaker_id'],
                    "session_id": f"spk{row['original_speaker_id']}_ses{row['session_id']}",
                    "source_utterance_id": Path(name).stem,
                })


                stats["n_written"] += 1
    return stats