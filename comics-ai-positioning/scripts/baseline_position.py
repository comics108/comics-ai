#!/usr/bin/env python3
"""Task 3.2 (flows/sdd-comics-ai-positioning/03-plan.md): rule-based positioner. Stacks a page's
regions vertically in reading order, using Task 3.1's real spacing_stats.json for both each
region's own height (per-kind median) and the gap to the next region (global median).

Real finding from Task 3.1, not assumed: the real median inter-region Y gap is *negative*
(-356px) -- comic layers of different kinds routinely overlap in Y (e.g. a balloon and the
character it's attached to share most of their Y range; a background spans the same Y range as
everything drawn on top of it). Using the real (signed) median gap as-is, rather than clamping to a
non-negative "margin", is what makes this baseline reproduce that overlap behavior instead of
incorrectly spreading every region out with dead space between it and the next.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from positioning_models import PositionProposal, RegionFeatures

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_STATS_PATH = WORK_DIR / "spacing_stats.json"
TRAIN_SIZE = 256  # matches infer_segmenter.TRAIN_SIZE -- the coordinate space local_bbox is in


def load_stats(path: Path = DEFAULT_STATS_PATH) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"No spacing stats at {path} -- run spacing_stats.py first")
    return json.loads(path.read_text())


def _height_for_kind(kind: str, stats: dict) -> int:
    by_kind = stats["height_by_kind"]
    if kind in by_kind and by_kind[kind]["count"] > 0:
        return int(by_kind[kind]["median"])
    # unseen/rare kind (e.g. a heuristic fallback not among the 4 trained kinds): fall back to the
    # overall median across all kinds rather than crashing or silently returning 0
    all_medians = [v["median"] for v in by_kind.values() if v.get("count", 0) > 0]
    return int(sum(all_medians) / len(all_medians)) if all_medians else 0


def position_page(
    regions: list[RegionFeatures], stats: dict, region_ids: list[str] | None = None
) -> list[PositionProposal]:
    """Sorts by ascending reading_order_index internally -- region_ids (if given) are paired with
    `regions` by list position *before* sorting, so callers may pass either list in any order and
    still get IDs matched to their own region (a real bug caught by
    test_position_page_is_deterministic_and_order_preserving: an earlier version sorted `regions`
    but zipped against the still-unsorted `region_ids`, silently mismatching IDs to positions).
    """
    if region_ids is None:
        region_ids = [f"region_{i}" for i in range(len(regions))]
    ordered = sorted(zip(regions, region_ids), key=lambda pair: pair[0].reading_order_index)

    gap = stats["gap"]["median"] if stats["gap"]["count"] > 0 else 0
    canvas_width = stats["canvas_width"]

    proposals: list[PositionProposal] = []
    cursor_y = 0
    for region, region_id in ordered:
        height = _height_for_kind(region.kind, stats)
        x0, _y0, x1, _y1 = region.local_bbox
        center_x_local = (x0 + x1) / 2.0
        proposed_x = round(center_x_local * canvas_width / TRAIN_SIZE - (x1 - x0) / 2.0)

        proposals.append(
            PositionProposal(
                region_id=region_id,
                proposed_x=proposed_x,
                proposed_y=round(cursor_y),
                source="baseline",
                confidence=None,
            )
        )
        cursor_y += height + gap

    return proposals


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stats", type=Path, default=DEFAULT_STATS_PATH)
    args = parser.parse_args()
    stats = load_stats(args.stats)
    print(f"Loaded stats: canvas_width={stats['canvas_width']}, gap_median={stats['gap']['median']}")
    for kind, summary in stats["height_by_kind"].items():
        print(f"  {kind}: median height = {summary['median']}")


if __name__ == "__main__":
    main()
