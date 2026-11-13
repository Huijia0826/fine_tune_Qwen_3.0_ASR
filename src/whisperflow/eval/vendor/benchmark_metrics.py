#!/usr/bin/env python3
"""Calculate per-record and pooled-batch WER, CER, or mixed-language MER.

The formulas are implemented directly in this file:

    single_error_rate = (substitutions + deletions + insertions) / reference_tokens

    batch_error_rate = sum(edit_distance_i) / sum(reference_tokens_i)

The batch formula is a pooled/micro rate, not the arithmetic mean of the
per-record rates. Empty-reference records receive rate 0 and are excluded from
the pooled denominator, matching the backend's current behavior.

The edit-distance formulas and MER tokenizer mirror the backend behavior. The
dependency-free normalized WER/CER tokenizers intentionally cover the common
case but not every project-specific number, unit, abbreviation, or spelling
rewrite in the larger Go normalizers. Inspect the target JSONL before scoring
when exact normalized parity is required.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, TextIO, Tuple


_DISFLUENCIES = {
    "嗯", "呃", "啊", "唉", "哎", "哦", "咦", "诶", "嗳",
    "uh", "um", "ah", "oh", "eh", "mm", "hm", "hmm", "mhm",
}

_ENGLISH_VARIANTS = {
    "okay": "ok", "ya": "yeah", "yep": "yes", "yup": "yes", "nope": "no",
    "gonna": "going to", "wanna": "want to", "gotta": "got to",
    "kinda": "kind of", "sorta": "sort of", "dunno": "do not know",
    "lemme": "let me", "gimme": "give me", "kindergarten": "kinder garden",
    "iphone": "i phone", "wifi": "wi fi", "alright": "all right",
    "everyday": "every day", "ecommerce": "e commerce", "email": "e mail",
}

_ERHUA_VARIANTS = {
    "一点儿": "一点", "一会儿": "一会", "一块儿": "一块", "没事儿": "没事",
    "有点儿": "有点", "好玩儿": "好玩", "一下儿": "一下", "一阵儿": "一阵",
    "玩儿": "玩", "哪儿": "哪", "这儿": "这", "那儿": "那", "今儿": "今",
    "明儿": "明", "昨儿": "昨", "早儿": "早", "晚儿": "晚",
}

_TRADITIONAL_TO_SIMPLIFIED = str.maketrans({
    "個": "个", "應": "应", "啟": "启", "動": "动", "顯": "显",
    "網": "网", "頁": "页", "瀏": "浏", "覽": "览", "畫": "画",
    "點": "点", "國": "国", "為": "为", "號": "号", "後": "后",
    "臺": "台", "裡": "里", "裏": "里", "們": "们", "學": "学",
    "習": "习", "體": "体", "與": "与", "萬": "万", "億": "亿",
    "廣": "广", "東": "东", "長": "长", "發": "发", "說": "说",
    "聽": "听", "這": "这", "麼": "么", "時": "时", "間": "间",
    "開": "开", "關": "关", "門": "门", "馬": "马", "車": "车",
})

_ZH_DIGITS = {
    "零": 0, "一": 1, "二": 2, "三": 3, "四": 4,
    "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "两": 2,
}
_ZH_UNITS = {"十": 10, "百": 100, "千": 1000, "万": 10_000, "亿": 100_000_000}
_ZH_NUMERAL_RE = re.compile(
    r"(?:[零一二三四五六七八九两]+(?:[十百千万亿][零一二三四五六七八九两]*)+"
    r"|[十百千万亿][零一二三四五六七八九两]*"
    r"|[零一二三四五六七八九]{2,}"
    r"|[零一二三四五六七八九两]+点[零一二三四五六七八九]+)"
)
_LETTER_RUN_RE = re.compile(r"(?<![A-Za-z])(?:[A-Za-z][ \u00a0\t]+){1,}[A-Za-z](?![A-Za-z])")
_BRACKETED_RE = re.compile(r"<[^>]*>|\[[^]]*\]|\([^)]*\)")


@dataclass(frozen=True)
class AlignmentCounts:
    hits: int
    substitutions: int
    deletions: int
    insertions: int

    @property
    def edit_distance(self) -> int:
        # d = S + D + I
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class MetricResult:
    metric_type: str
    method: str
    reference_tokens: List[str]
    hypothesis_tokens: List[str]
    reference_token_count: int
    hypothesis_token_count: int
    hits: int
    substitutions: int
    deletions: int
    insertions: int
    edit_distance: int
    error_rate: float
    included_in_batch_summary: bool


@dataclass(frozen=True)
class BatchResult:
    metric_type: str
    method: str
    record_count: int
    scored_record_count: int
    empty_reference_count: int
    substitutions: int
    deletions: int
    insertions: int
    edit_distance: int
    reference_token_count: int
    pooled_batch_error_rate: Optional[float]


def align_tokens(reference: Sequence[str], hypothesis: Sequence[str]) -> AlignmentCounts:
    """Return H/S/D/I counts from a minimum-edit Levenshtein alignment."""
    rows = len(reference) + 1
    columns = len(hypothesis) + 1
    distance = [[0] * columns for _ in range(rows)]

    for i in range(rows):
        distance[i][0] = i
    for j in range(columns):
        distance[0][j] = j

    for i in range(1, rows):
        for j in range(1, columns):
            substitution_cost = 0 if reference[i - 1] == hypothesis[j - 1] else 1
            distance[i][j] = min(
                distance[i - 1][j - 1] + substitution_cost,
                distance[i - 1][j] + 1,
                distance[i][j - 1] + 1,
            )

    hits = substitutions = deletions = insertions = 0
    i, j = len(reference), len(hypothesis)
    while i > 0 or j > 0:
        if i > 0 and j > 0 and reference[i - 1] == hypothesis[j - 1]:
            hits += 1
            i -= 1
            j -= 1
        elif j > 0 and distance[i][j] == distance[i][j - 1] + 1:
            insertions += 1
            j -= 1
        elif i > 0 and distance[i][j] == distance[i - 1][j] + 1:
            deletions += 1
            i -= 1
        else:
            substitutions += 1
            i -= 1
            j -= 1

    return AlignmentCounts(hits, substitutions, deletions, insertions)


def single_error_rate(counts: AlignmentCounts) -> float:
    """Single item formula: (S + D + I) / N, where N = H + S + D."""
    reference_token_count = counts.hits + counts.substitutions + counts.deletions
    if reference_token_count == 0:
        return 0.0
    return (
        counts.substitutions + counts.deletions + counts.insertions
    ) / reference_token_count


def pooled_batch_error_rate(results: Iterable[MetricResult]) -> Optional[float]:
    """Batch formula: sum(S + D + I) / sum(N), not mean(item rates)."""
    included = [result for result in results if result.included_in_batch_summary]
    total_errors = sum(result.edit_distance for result in included)
    total_reference_tokens = sum(result.reference_token_count for result in included)
    if total_reference_tokens == 0:
        return None
    return total_errors / total_reference_tokens


def is_han(character: str) -> bool:
    codepoint = ord(character)
    return (
        0x3400 <= codepoint <= 0x4DBF
        or 0x4E00 <= codepoint <= 0x9FFF
        or 0xF900 <= codepoint <= 0xFAFF
        or 0x20000 <= codepoint <= 0x2FA1F
        or 0x30000 <= codepoint <= 0x3134F
        or character in {"〇", "○"}
    )


def strict_wer_tokens(text: str) -> List[str]:
    """Case-sensitive WER tokens; punctuation and line breaks are tokens."""
    tokens: List[str] = []
    word: List[str] = []

    def flush_word() -> None:
        if word:
            tokens.append("".join(word))
            word.clear()

    index = 0
    while index < len(text):
        character = text[index]
        if is_han(character):
            flush_word()
            tokens.append(character)
        elif character.isalpha() or character.isdigit():
            word.append(character)
        elif character == "\r":
            flush_word()
            if index + 1 < len(text) and text[index + 1] == "\n":
                tokens.append("\r\n")
                index += 1
            else:
                tokens.append("\r")
        elif character == "\n":
            flush_word()
            tokens.append("\n")
        elif character.isspace():
            flush_word()
        else:
            flush_word()
            tokens.append(character)
        index += 1

    flush_word()
    return tokens


def normalized_wer_tokens(text: str) -> List[str]:
    """Dependency-free normalized WER tokenizer for ordinary transcripts."""
    text = unicodedata.normalize("NFKC", text).lower()
    text = _BRACKETED_RE.sub(" ", text)
    text = re.sub(r"\b(?:hmm|mm|mhm|mmm|uh|um|huh)\b", " ", text)
    text = "".join(
        " " if unicodedata.category(character)[0] in {"P", "S", "M"} else character
        for character in text
    )

    tokens: List[str] = []
    for field in text.split():
        non_han_run: List[str] = []
        for character in field:
            if is_han(character):
                if non_han_run:
                    tokens.append("".join(non_han_run))
                    non_han_run.clear()
                tokens.append(character)
            else:
                non_han_run.append(character)
        if non_han_run:
            tokens.append("".join(non_han_run))
    return tokens


def cer_tokens(text: str) -> List[str]:
    """CER tokens: normalized letters/digits/Han, one Unicode character each."""
    text = unicodedata.normalize("NFKC", text).translate(_TRADITIONAL_TO_SIMPLIFIED)
    return [
        character.lower()
        for character in text
        if is_han(character) or character.isalpha() or character.isdigit()
    ]


def _parse_zh_numeral(text: str) -> Optional[str]:
    if not text:
        return None
    if "点" in text:
        whole, fraction = text.split("点", 1)
        whole_value = _parse_zh_numeral(whole) if whole else "0"
        if whole_value is None or "." in whole_value or not fraction:
            return None
        fraction_digits: List[str] = []
        for character in fraction:
            if character not in _ZH_DIGITS or character == "两":
                return None
            fraction_digits.append(str(_ZH_DIGITS[character]))
        return f"{whole_value}.{''.join(fraction_digits)}"

    result = current = big_accumulator = 0
    for character in text:
        if character in _ZH_DIGITS:
            current = _ZH_DIGITS[character]
        elif character in _ZH_UNITS:
            unit = _ZH_UNITS[character]
            if unit in {10, 100, 1000}:
                if current == 0:
                    current = 1
                result += current * unit
                current = 0
            else:
                section = result + current
                if section == 0:
                    section = 1
                big_accumulator = big_accumulator * unit + section * unit
                result = current = 0
        else:
            return None
    return str(big_accumulator + result + current)


def mixed_tokens(text: str) -> List[str]:
    """MER tokens: Han chars + lowercased ASCII word runs + numeric runs."""
    text = unicodedata.normalize("NFKC", text).translate(_TRADITIONAL_TO_SIMPLIFIED)
    for source, target in _ERHUA_VARIANTS.items():
        text = text.replace(source, target)
    text = _ZH_NUMERAL_RE.sub(
        lambda match: _parse_zh_numeral(match.group(0)) or match.group(0), text
    )
    text = _LETTER_RUN_RE.sub(
        lambda match: re.sub(r"[ \u00a0\t]+", "", match.group(0)), text
    )

    tokens: List[str] = []
    index = 0
    while index < len(text):
        character = text[index]
        if is_han(character):
            tokens.append(character)
            index += 1
        elif character.isascii() and character.isalpha():
            end = index + 1
            while end < len(text) and text[end].isascii() and text[end].isalpha():
                end += 1
            tokens.append(text[index:end].lower())
            index = end
        elif character.isascii() and character.isdigit():
            end = index + 1
            while end < len(text) and (text[end].isdigit() or text[end] == "."):
                end += 1
            token = text[index:end].rstrip(".")
            if token:
                tokens.append(token)
            index = end
        else:
            index += 1

    expanded: List[str] = []
    for token in tokens:
        expanded.extend(_ENGLISH_VARIANTS.get(token, token).split())
    return [token for token in expanded if token not in _DISFLUENCIES]


def tokenize(text: str, metric_type: str, method: str = "normalized") -> List[str]:
    if metric_type == "wer":
        return strict_wer_tokens(text) if method == "strict" else normalized_wer_tokens(text)
    if metric_type == "cer":
        return cer_tokens(text)
    if metric_type == "mer":
        return mixed_tokens(text)
    raise ValueError(f"unsupported metric type: {metric_type!r}")


def score_one(
    reference: str,
    hypothesis: str,
    metric_type: str,
    method: str = "normalized",
) -> MetricResult:
    reference_tokens = tokenize(reference, metric_type, method)
    hypothesis_tokens = tokenize(hypothesis, metric_type, method)

    # The backend returns zero before calculating insertions for an empty reference.
    counts = (
        align_tokens(reference_tokens, hypothesis_tokens)
        if reference_tokens
        else AlignmentCounts(0, 0, 0, 0)
    )
    rate = single_error_rate(counts)

    return MetricResult(
        metric_type=metric_type,
        method=method,
        reference_tokens=reference_tokens,
        hypothesis_tokens=hypothesis_tokens,
        reference_token_count=len(reference_tokens),
        hypothesis_token_count=len(hypothesis_tokens),
        hits=counts.hits,
        substitutions=counts.substitutions,
        deletions=counts.deletions,
        insertions=counts.insertions,
        edit_distance=counts.edit_distance,
        error_rate=rate,
        included_in_batch_summary=bool(reference_tokens),
    )


def score_batch(
    pairs: Iterable[Tuple[str, str]],
    metric_type: str,
    method: str = "normalized",
) -> Tuple[List[MetricResult], BatchResult]:
    results = [score_one(reference, hypothesis, metric_type, method) for reference, hypothesis in pairs]
    included = [result for result in results if result.included_in_batch_summary]
    substitutions = sum(result.substitutions for result in included)
    deletions = sum(result.deletions for result in included)
    insertions = sum(result.insertions for result in included)
    reference_tokens = sum(result.reference_token_count for result in included)

    # Batch formula: sum(S_i + D_i + I_i) / sum(N_i).
    edit_distance = substitutions + deletions + insertions
    batch_rate = edit_distance / reference_tokens if reference_tokens else None

    return results, BatchResult(
        metric_type=metric_type,
        method=method,
        record_count=len(results),
        scored_record_count=len(included),
        empty_reference_count=len(results) - len(included),
        substitutions=substitutions,
        deletions=deletions,
        insertions=insertions,
        edit_distance=edit_distance,
        reference_token_count=reference_tokens,
        pooled_batch_error_rate=batch_rate,
    )


def iter_jsonl(stream: TextIO) -> Iterator[Tuple[int, Dict[str, object]]]:
    for line_number, line in enumerate(stream, start=1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"line {line_number}: expected a JSON object")
        yield line_number, value


def require_string(record: Dict[str, object], field: str, line_number: int) -> str:
    if field not in record:
        raise ValueError(f"line {line_number}: missing {field!r} field")
    value = record[field]
    if not isinstance(value, str):
        raise ValueError(f"line {line_number}: field {field!r} must be a string")
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="input JSONL path, or - for stdin")
    parser.add_argument("--output", help="optional enriched JSONL output path")
    parser.add_argument("--reference-field", default="reference")
    parser.add_argument("--hypothesis-field", default="hypothesis")
    parser.add_argument("--metric-field", default="_asr_metric")
    parser.add_argument("--metric", required=True, choices=("wer", "cer", "mer"))
    parser.add_argument("--method", default="normalized", choices=("normalized", "strict"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.input != "-" and args.output:
        if Path(args.input).resolve() == Path(args.output).resolve():
            raise ValueError("--output must not overwrite --input")

    input_stream = sys.stdin if args.input == "-" else open(args.input, encoding="utf-8")
    output_stream = open(args.output, "w", encoding="utf-8") if args.output else None
    pairs: List[Tuple[str, str]] = []
    records: List[Dict[str, object]] = []

    try:
        for line_number, record in iter_jsonl(input_stream):
            reference = require_string(record, args.reference_field, line_number)
            hypothesis = require_string(record, args.hypothesis_field, line_number)
            pairs.append((reference, hypothesis))
            records.append(record)
    finally:
        if input_stream is not sys.stdin:
            input_stream.close()

    results, summary = score_batch(pairs, args.metric, args.method)
    if output_stream is not None:
        try:
            for record, result in zip(records, results):
                record[args.metric_field] = asdict(result)
                output_stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        finally:
            output_stream.close()

    print(json.dumps(asdict(summary), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"jsonl metrics: {error}", file=sys.stderr)
        raise SystemExit(1)
