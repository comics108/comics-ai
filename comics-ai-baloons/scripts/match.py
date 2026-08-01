#!/usr/bin/env python3
"""Stage 4: fuzzy-match each balloon's OCR'd text to a translation CSV row.

Highest-risk stage in the pipeline (flows/sdd-comics-ai-baloons/03-plan.md Risk Assessment): the
CSV's P<page>_<bubble> numbering does not correspond to a .comics file's local balloon order, and
the user explicitly warned the CSV may be a different version of the text (phrase corrections,
numbering shifts). Matching is therefore purely content-based (OCR text vs CSV text), never
index/sequence-based, and always skips + logs rather than guessing on low confidence.

Primary signal: en OCR vs CSV "en" column. ru is used as (a) a fallback when en OCR text is
missing/empty, and (b) a tie-breaker among near-tied en candidates -- never as a hard filter.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from rapidfuzz import fuzz

import csv_loader
from csv_loader import CsvRow
from models import MatchResult, OcrResult

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_OCR_JSONL = WORK_DIR / "ocr.jsonl"
DEFAULT_OUT = WORK_DIR / "matches.jsonl"

SCORE_THRESHOLD = 75.0  # rapidfuzz 0-100 scale (== 0.75 in Specifications' 0-1 framing)
TIE_MARGIN = 5.0  # points; candidates within this margin of the best are considered "tied"

_PUNCT_RE = re.compile(r"[^\w\s]", flags=re.UNICODE)
_WS_RE = re.compile(r"\s+")


def normalize(text: str) -> str:
    text = text.lower()
    text = _PUNCT_RE.sub("", text)
    text = _WS_RE.sub(" ", text).strip()
    return text


def read_ocr_jsonl(path: Path) -> list[OcrResult]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(OcrResult(**json.loads(line)))
    return out


def group_by_balloon(ocr_results: list[OcrResult]) -> dict[tuple[str, int], dict[int, OcrResult]]:
    grouped: dict[tuple[str, int], dict[int, OcrResult]] = defaultdict(dict)
    for r in ocr_results:
        grouped[(r.source_file, r.layer_index)][r.lang_index] = r
    return grouped


def score_candidates(
    text: str, csv_rows: list[CsvRow], lang: str
) -> list[tuple[CsvRow, float]]:
    norm = normalize(text)
    if not norm:
        return []
    scored = []
    for row in csv_rows:
        candidate = row.texts.get(lang)
        if not candidate:
            continue
        score = fuzz.token_sort_ratio(norm, normalize(candidate))
        scored.append((row, score))
    scored.sort(key=lambda pair: -pair[1])
    return scored


def match_balloon(
    source_file: str,
    layer_index: int,
    en_text: str,
    ru_text: str,
    csv_rows: list[CsvRow],
) -> MatchResult:
    if not en_text and not ru_text:
        return MatchResult(
            source_file=source_file,
            layer_index=layer_index,
            csv_row_id=None,
            match_score=0.0,
            matched_on="",
            status="skipped_no_match",
            reason="no OCR text on either language",
        )

    primary_lang, primary_text = ("en", en_text) if en_text else ("ru", ru_text)
    secondary_lang, secondary_text = ("ru", ru_text) if primary_lang == "en" else ("en", en_text)

    candidates = score_candidates(primary_text, csv_rows, primary_lang)
    if not candidates:
        return MatchResult(
            source_file=source_file,
            layer_index=layer_index,
            csv_row_id=None,
            match_score=0.0,
            matched_on="",
            status="skipped_no_match",
            reason=f"no CSV rows have any '{primary_lang}' text to compare against",
        )

    best_row, best_score = candidates[0]
    second_score = candidates[1][1] if len(candidates) > 1 else 0.0

    if best_score < SCORE_THRESHOLD:
        return MatchResult(
            source_file=source_file,
            layer_index=layer_index,
            csv_row_id=None,
            match_score=best_score / 100.0,
            matched_on=primary_lang,
            status="skipped_low_confidence",
            reason=f"best {primary_lang} score {best_score:.1f} < threshold {SCORE_THRESHOLD}",
        )

    if best_score - second_score < TIE_MARGIN:
        tied = [c for c in candidates if best_score - c[1] < TIE_MARGIN]
        if secondary_text and len(tied) > 1:
            tied_row_ids = {row.row_id for row, _ in tied}
            tied_rows = [r for r in csv_rows if r.row_id in tied_row_ids]
            sec_candidates = score_candidates(secondary_text, tied_rows, secondary_lang)
            sec_best_score = sec_candidates[0][1] if sec_candidates else -1.0
            sec_second_score = sec_candidates[1][1] if len(sec_candidates) > 1 else -100.0
            if sec_candidates and (sec_best_score - sec_second_score >= TIE_MARGIN):
                resolved_row = sec_candidates[0][0]
                return MatchResult(
                    source_file=source_file,
                    layer_index=layer_index,
                    csv_row_id=resolved_row.row_id,
                    match_score=best_score / 100.0,
                    matched_on=f"{primary_lang}+{secondary_lang}_tiebreak",
                    status="matched",
                    reason=(
                        f"{len(tied)} candidates tied on {primary_lang} "
                        f"(top {best_score:.1f}); resolved via {secondary_lang}"
                    ),
                )
        tied_ids = ", ".join(sorted(r.row_id for r, _ in tied)[:5])
        return MatchResult(
            source_file=source_file,
            layer_index=layer_index,
            csv_row_id=None,
            match_score=best_score / 100.0,
            matched_on=primary_lang,
            status="skipped_ambiguous",
            reason=f"{len(tied)} candidates tied within {TIE_MARGIN} points: {tied_ids}",
        )

    return MatchResult(
        source_file=source_file,
        layer_index=layer_index,
        csv_row_id=best_row.row_id,
        match_score=best_score / 100.0,
        matched_on=primary_lang,
        status="matched",
        reason=f"{primary_lang} score {best_score:.1f}, margin {best_score - second_score:.1f}",
    )


def match_all(
    ocr_results: list[OcrResult], csv_rows: list[CsvRow]
) -> list[MatchResult]:
    grouped = group_by_balloon(ocr_results)
    out = []
    for (source_file, layer_index), slots in sorted(grouped.items()):
        en_text = slots.get(0).text if 0 in slots else ""
        ru_text = slots.get(1).text if 1 in slots else ""
        out.append(match_balloon(source_file, layer_index, en_text, ru_text, csv_rows))
    return out


def write_jsonl(results: list[MatchResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ocr", default=str(DEFAULT_OCR_JSONL))
    ap.add_argument("--csv", default=str(csv_loader.DEFAULT_CSV_PATH))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    ocr_results = read_ocr_jsonl(Path(args.ocr))
    csv_rows = csv_loader.load_csv(Path(args.csv))
    results = match_all(ocr_results, csv_rows)
    write_jsonl(results, Path(args.out))

    by_status: dict[str, int] = defaultdict(int)
    for r in results:
        by_status[r.status] += 1
    print(f"Matched {len(results)} balloons -> {args.out}")
    for status, count in sorted(by_status.items()):
        print(f"  {status}: {count}")


if __name__ == "__main__":
    main()
