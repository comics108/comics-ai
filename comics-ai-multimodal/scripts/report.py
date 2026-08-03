#!/usr/bin/env python3
"""Task 9.3: tie together every pipeline stage's output (alignment, cut regions, evaluation,
balloon handoff, character/environment library, packaging) into one final per-photo report,
mirroring comics-ai-baloons' match/skip report pattern.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent / "work"
DEFAULT_OUT_MD = WORK_DIR / "report.md"
DEFAULT_OUT_JSONL = WORK_DIR / "report.jsonl"


def _load_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    entries = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_report(work_dir: Path = WORK_DIR) -> list[dict]:
    alignment = _load_jsonl(work_dir / "alignment.jsonl")
    regions = _load_jsonl(work_dir / "regions.jsonl")
    eval_report = _load_jsonl(work_dir / "eval_report.jsonl")
    handoff = _load_jsonl(work_dir / "balloon_handoff.jsonl")

    regions_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in regions:
        regions_by_page[(r["photo_file"], r["page_index"])].append(r)

    eval_by_page = {(e["photo_file"], e["page_index"]): e for e in eval_report}
    handoff_by_page = {(h["photo_file"], h["page_index"]): h for h in handoff}

    packaged_files = {p.stem for p in (work_dir / "output").glob("*.comics")} if (work_dir / "output").is_dir() else set()

    entries = []
    for a in alignment:
        key = (a["photo_file"], a["page_index"])
        page_regions = regions_by_page.get(key, [])
        kind_counts: dict[str, int] = defaultdict(int)
        for r in page_regions:
            kind_counts[r["predicted_kind"]] += 1

        entry = {
            "photo_file": a["photo_file"],
            "page_index": a["page_index"],
            "status": a["status"],
            "episode_file": a.get("episode_file"),
            "match_confidence": a.get("confidence", 0.0),
            "reason": a.get("reason", ""),
            "regions_cut": len(page_regions),
            "regions_by_kind": dict(kind_counts),
            "packaged": f"{Path(a['photo_file']).stem}_p{a['page_index']}" in packaged_files,
        }

        ev = eval_by_page.get(key)
        if ev:
            entry["mean_kind_count_agreement"] = ev["mean_agreement"]

        ho = handoff_by_page.get(key)
        if ho:
            entry["real_balloon_layers"] = len(ho["real_balloon_layer_indexes"])
            entry["translated_balloon_layers"] = len(ho["translated_layer_indexes"])
            entry["packaged_translation_available"] = ho["packaged_output_available"]

        entries.append(entry)

    return entries


def render_markdown(entries: list[dict]) -> str:
    total = len(entries)
    matched = [e for e in entries if e["status"] == "matched"]
    skipped = [e for e in entries if e["status"] != "matched"]
    packaged = [e for e in matched if e.get("packaged")]

    lines = [
        "# comics-ai-multimodal Pipeline Report",
        "",
        f"- Total photo/pages processed: {total}",
        f"- Matched: {len(matched)} ({len(matched) / total * 100:.0f}%)" if total else "- Matched: 0",
        f"- Skipped: {len(skipped)}",
        f"- Packaged into new `.comics` files: {len(packaged)}",
        "",
    ]

    if matched:
        agreements = [e["mean_kind_count_agreement"] for e in matched if "mean_kind_count_agreement" in e]
        if agreements:
            lines.append(f"- Mean kind-count agreement (Task 6.2 metric) across matched pages: {sum(agreements) / len(agreements):.3f}")
            lines.append("")

    lines.append("## Skip reasons")
    reason_counts: dict[str, int] = defaultdict(int)
    for e in skipped:
        reason_counts[e["reason"]] += 1
    for reason, count in sorted(reason_counts.items(), key=lambda kv: -kv[1]):
        lines.append(f"- {count}x: {reason}")
    lines.append("")

    lines.append("## Matched pages")
    lines.append("")
    lines.append("| photo | page | episode | confidence | regions | packaged |")
    lines.append("|---|---|---|---|---|---|")
    for e in matched:
        lines.append(
            f"| {e['photo_file']} | {e['page_index']} | {e['episode_file']} | "
            f"{e['match_confidence']:.2f} | {e['regions_cut']} | {'yes' if e.get('packaged') else 'no'} |"
        )

    return "\n".join(lines) + "\n"


def write_report(
    work_dir: Path = WORK_DIR, out_md: Path | None = None, out_jsonl: Path | None = None
) -> list[dict]:
    entries = build_report(work_dir)
    out_jsonl = out_jsonl or (work_dir / "report.jsonl")
    out_md = out_md or (work_dir / "report.md")

    with out_jsonl.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    out_md.write_text(render_markdown(entries))
    return entries


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work-dir", type=Path, default=WORK_DIR)
    parser.add_argument("--out-md", type=Path, default=None)
    parser.add_argument("--out-jsonl", type=Path, default=None)
    args = parser.parse_args()

    entries = write_report(args.work_dir, args.out_md, args.out_jsonl)
    out_jsonl = args.out_jsonl or (args.work_dir / "report.jsonl")
    out_md = args.out_md or (args.work_dir / "report.md")

    print(f"{len(entries)} entries -> {out_jsonl}, {out_md}")


if __name__ == "__main__":
    main()
