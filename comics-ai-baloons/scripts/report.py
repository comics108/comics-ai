#!/usr/bin/env python3
"""Stage 7.2: consolidate every stage's output into one per-balloon report.

Every balloon discovered in stage 1 (work/balloons.jsonl -- the full universe, 825 in the current
dataset) gets exactly one row in work/report.jsonl with a terminal status, so nothing is silently
dropped: matched+rendered, matched+hand_lettered (flagged, not rendered), or one of the specific
skip reasons from stage 4's matcher. Also emits work/report.md, a human-readable rollup.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_BALLOONS = WORK_DIR / "balloons.jsonl"
DEFAULT_OCR = WORK_DIR / "ocr.jsonl"
DEFAULT_MATCHES = WORK_DIR / "matches.jsonl"
DEFAULT_LETTERING = WORK_DIR / "lettering.jsonl"
DEFAULT_RENDERS = WORK_DIR / "renders.jsonl"
DEFAULT_JSONL_OUT = WORK_DIR / "report.jsonl"
DEFAULT_MD_OUT = WORK_DIR / "report.md"


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def build_report(
    balloons: list[dict],
    ocr: list[dict],
    matches: list[dict],
    lettering: list[dict],
    renders: list[dict],
) -> list[dict]:
    key = lambda d: (d["source_file"], d["layer_index"])  # noqa: E731

    ocr_en_by_key = {key(r): r["text"] for r in ocr if r["lang_index"] == 0}
    match_by_key = {key(r): r for r in matches}
    lettering_by_key = {key(r): r["label"] for r in lettering}
    renders_by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in renders:
        renders_by_key[key(r)].append(r)

    report = []
    for b in balloons:
        k = key(b)
        match = match_by_key.get(k)
        lettering_class = lettering_by_key.get(k)
        balloon_renders = renders_by_key.get(k, [])
        rendered_langs = sorted(r["lang_code"] for r in balloon_renders if r["rendered"])
        skipped_langs = [
            {"lang": r["lang_code"], "reason": r["reason"]}
            for r in balloon_renders
            if not r["rendered"]
        ]

        if match is None:
            status = "not_matched_no_data"
        elif match["status"] != "matched":
            status = match["status"]
        elif lettering_class == "hand_lettered":
            status = "hand_lettered_flagged"
        elif rendered_langs:
            status = "rendered"
        else:
            status = "matched_no_renders"

        report.append(
            {
                "source_file": b["source_file"],
                "layer_index": b["layer_index"],
                "ocr_text_en": ocr_en_by_key.get(k, ""),
                "match": {
                    "csv_row_id": match["csv_row_id"] if match else None,
                    "score": match["match_score"] if match else None,
                    "matched_on": match["matched_on"] if match else None,
                }
                if match
                else None,
                "lettering_class": lettering_class,
                "languages_rendered": rendered_langs,
                "languages_skipped": skipped_langs,
                "status": status,
            }
        )
    return report


def write_jsonl(report: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in report:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def render_markdown(report: list[dict]) -> str:
    lines = ["# comics-ai-baloons pipeline report", ""]

    status_counts = Counter(r["status"] for r in report)
    lines.append(f"**Total balloons discovered**: {len(report)}")
    lines.append("")
    lines.append("## By status")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|---|---|")
    for status, count in sorted(status_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {status} | {count} |")
    lines.append("")

    lang_counts = Counter(lang for r in report for lang in r["languages_rendered"])
    lines.append("## Languages rendered (balloon count per language)")
    lines.append("")
    lines.append("| Language | Balloons rendered |")
    lines.append("|---|---|")
    for lang, count in sorted(lang_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"| {lang} | {count} |")
    lines.append("")

    by_file: dict[str, Counter] = defaultdict(Counter)
    for r in report:
        by_file[r["source_file"]][r["status"]] += 1
    lines.append("## Per-file rollup")
    lines.append("")
    lines.append("| File | Rendered | Hand-lettered flagged | Skipped | Total |")
    lines.append("|---|---|---|---|---|")
    for source_file, counts in sorted(by_file.items()):
        total = sum(counts.values())
        rendered = counts.get("rendered", 0)
        hand = counts.get("hand_lettered_flagged", 0)
        skipped = total - rendered - hand
        lines.append(f"| {source_file} | {rendered} | {hand} | {skipped} | {total} |")
    lines.append("")

    # All balloons classified hand-lettered, regardless of CSV match status -- the user asked to
    # find and flag *every* hand-lettered balloon (Requirements), not just the ones that happen to
    # also be renderable. A hand-lettered balloon that failed to match still deserves visibility.
    hand_lettered = [r for r in report if r["lettering_class"] == "hand_lettered"]
    lines.append(f"## All balloons classified hand-lettered ({len(hand_lettered)})")
    lines.append("")
    lines.append(
        "Flagged for manual/artist review regardless of whether they also matched a CSV row "
        "and could be rendered (see `status` column)."
    )
    lines.append("")
    if hand_lettered:
        lines.append("| File | Layer | Status | OCR text (en) |")
        lines.append("|---|---|---|---|")
        for r in hand_lettered:
            lines.append(
                f"| {r['source_file']} | {r['layer_index']} | {r['status']} | "
                f"{r['ocr_text_en'][:60]} |"
            )
    else:
        lines.append("None.")
    lines.append("")

    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--balloons", default=str(DEFAULT_BALLOONS))
    ap.add_argument("--ocr", default=str(DEFAULT_OCR))
    ap.add_argument("--matches", default=str(DEFAULT_MATCHES))
    ap.add_argument("--lettering", default=str(DEFAULT_LETTERING))
    ap.add_argument("--renders", default=str(DEFAULT_RENDERS))
    ap.add_argument("--jsonl-out", default=str(DEFAULT_JSONL_OUT))
    ap.add_argument("--md-out", default=str(DEFAULT_MD_OUT))
    args = ap.parse_args()

    balloons = read_jsonl(Path(args.balloons))
    ocr = read_jsonl(Path(args.ocr))
    matches = read_jsonl(Path(args.matches))
    lettering = read_jsonl(Path(args.lettering))
    renders = read_jsonl(Path(args.renders))

    report = build_report(balloons, ocr, matches, lettering, renders)
    write_jsonl(report, Path(args.jsonl_out))
    Path(args.md_out).write_text(render_markdown(report), encoding="utf-8")

    print(f"Report: {len(report)} balloons -> {args.jsonl_out}, {args.md_out}")


if __name__ == "__main__":
    main()
