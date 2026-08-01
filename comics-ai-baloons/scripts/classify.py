#!/usr/bin/env python3
"""Stage 5.3: classify each balloon as machine_set vs. hand_lettered.

Threshold derived empirically (Checkpoint B, flows/sdd-comics-ai-baloons/04-implementation-log.md):
a 39-balloon stratified manual sample across every naming-convention era in the dataset found only
two distinct *uniform* digitally-set fonts and zero organic hand lettering. A follow-up automated
sweep of stroke_width_cv across all 825 balloons found a clean separation: the highest "normal"
balloon scored 0.47, while the one genuine hand-drawn/dynamic panel found (an "AHAHAHAHAHA"
sound-effect treatment: gradient color, slanted, jagged burst outline) scored 1.2-1.56. The
threshold sits well inside that gap.

This is intentionally a simple, auditable rule rather than a trained classifier -- with a true
positive rate of ~1 balloon in 825, there is nowhere near enough data to train one, and Checkpoint B
found no evidence more exist in the current dataset than the outlier sweep already surfaced.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from models import LetteringClass

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_FEATURES_JSONL = WORK_DIR / "lettering_features.jsonl"
DEFAULT_OUT = WORK_DIR / "lettering.jsonl"

STROKE_WIDTH_CV_THRESHOLD = 0.6


def read_features_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def classify_all(feature_rows: list[dict]) -> list[LetteringClass]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for r in feature_rows:
        grouped[(r["source_file"], r["layer_index"])].append(r)

    out = []
    for (source_file, layer_index), rows in sorted(grouped.items()):
        best = max(rows, key=lambda r: r["stroke_width_cv"])
        label = (
            "hand_lettered"
            if best["stroke_width_cv"] > STROKE_WIDTH_CV_THRESHOLD
            else "machine_set"
        )
        out.append(
            LetteringClass(
                source_file=source_file,
                layer_index=layer_index,
                label=label,
                confidence=min(1.0, best["stroke_width_cv"] / (2 * STROKE_WIDTH_CV_THRESHOLD)),
                signals={
                    "stroke_width_cv": best["stroke_width_cv"],
                    "baseline_wobble": best["baseline_wobble"],
                    "ocr_confidence": best["ocr_confidence"],
                    "needed_crop_fallback": float(best["needed_crop_fallback"]),
                },
            )
        )
    return out


def write_jsonl(results: list[LetteringClass], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--features", default=str(DEFAULT_FEATURES_JSONL))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rows = read_features_jsonl(Path(args.features))
    results = classify_all(rows)
    write_jsonl(results, Path(args.out))

    n_hand = sum(1 for r in results if r.label == "hand_lettered")
    print(f"Classified {len(results)} balloons -> {args.out} ({n_hand} hand_lettered)")
    for r in results:
        if r.label == "hand_lettered":
            print(f"  {r.source_file} layer {r.layer_index}: {r.signals}")


if __name__ == "__main__":
    main()
