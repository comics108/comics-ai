#!/usr/bin/env python3
"""Task 3.1 (flows/sdd-comics-ai-positioning/03-plan.md): mine real per-kind height and inter-
region gap statistics from comics-multimodal's ground-truth canvases (all 27 files, not just the
16 with a matched photo -- this doesn't need photo alignment at all, just the known-good canvases),
to calibrate the baseline positioner with real numbers instead of guessed constants.

Confirmed before writing this (not assumed): all 27 canvases share the same width (1080px); height
varies 12000-100900px. So X can be handled as a simple proportional rescale from the 256px
TRAIN_SIZE coordinate space CutRegion bboxes live in (infer_segmenter.py) to this fixed canvas
width -- Y cannot, since Y is exactly the thing this flow predicts (stacking order/spacing), not a
fixed-ratio rescale.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import positioning_bridge as pb

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_OUT = WORK_DIR / "spacing_stats.json"

CANVAS_WIDTH = 1080  # confirmed constant across all 27 real files -- see module docstring


def _height(bbox: list[int]) -> int:
    return bbox[3] - bbox[1]


def compute_stats(exclude_episode_stems: set[str] | None = None) -> dict:
    """`exclude_episode_stems` (episode filename stems, no extension) lets a held-out evaluation
    (evaluate_positioning.py) compute calibration stats from the *training* portion only -- without
    this, evaluating the baseline against a held-out episode using stats mined from all 27 files
    (including that same held-out episode's own canvas) would leak its own layout into its own eval,
    exactly what flows/sdd-comics-ai-positioning/03-plan.md's Task 4.1 says to avoid.
    """
    refs = pb.load_all_canvas_references()
    if exclude_episode_stems:
        refs = {stem: ref for stem, ref in refs.items() if stem not in exclude_episode_stems}

    heights_by_kind: dict[str, list[int]] = defaultdict(list)
    gaps: list[int] = []

    for ref in refs.values():
        regions = sorted(ref["regions"], key=lambda r: r["bbox"][1])
        for i, region in enumerate(regions):
            heights_by_kind[region["kind"]].append(_height(region["bbox"]))
            if i + 1 < len(regions):
                gap = regions[i + 1]["bbox"][1] - region["bbox"][3]
                gaps.append(gap)

    def summarize(values: list[int]) -> dict:
        if not values:
            return {"count": 0}
        sorted_vals = sorted(values)
        return {
            "count": len(values),
            "median": statistics.median(sorted_vals),
            "p25": sorted_vals[len(sorted_vals) // 4],
            "p75": sorted_vals[min(len(sorted_vals) - 1, 3 * len(sorted_vals) // 4)],
        }

    return {
        "canvas_width": CANVAS_WIDTH,
        "height_by_kind": {kind: summarize(vals) for kind, vals in heights_by_kind.items()},
        "gap": summarize(gaps),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    stats = compute_stats()
    args.out.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
