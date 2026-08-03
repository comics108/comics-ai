#!/usr/bin/env python3
"""Task 2.2 (flows/sdd-comics-ai-positioning/03-plan.md): join comics-multimodal's predicted
CutRegions (work/regions.jsonl, photo/page-space) against its ground-truth canvas positions
(work/canvas/*.gt.json, filtered to each matched page's ground_truth_cluster) to build real
(region -> target position) training pairs.

Pairing caveat (real, disclosed -- not glossed over): a matched page's predicted region count and
its ground_truth_cluster's region count are NOT guaranteed equal (verified on real data: photo
20260731_153228.jpg page 0 has 16 predicted regions vs. 17 ground-truth cluster regions, with
different per-kind splits too -- there is no per-region ID linking the two sides, only a page-level
match). This module pairs them by **kind, in top-to-bottom (reading) order, truncated to
min(predicted_count, ground_truth_count) per kind** -- an explicit, honest ordinal-matching
heuristic, not a claim of verified per-region correspondence. Leftover unpaired regions on the
larger side are dropped, not guessed.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import positioning_bridge as pb
import scene_text
import text_context
from positioning_models import PositionTrainingPair, RegionFeatures

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_OUT_DIR = WORK_DIR / "train_pairs"


def _load_regions_by_page(regions_path: Path = pb.REGIONS_JSONL) -> dict[tuple[str, int], list[dict]]:
    by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with regions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_page[(row["photo_file"], row["page_index"])].append(row)
    return by_page


def _sort_top_to_bottom(regions: list[dict], bbox_key: str = "bbox") -> list[dict]:
    return sorted(regions, key=lambda r: (r[bbox_key][1], r[bbox_key][0]))


def _cluster_into_rows(regions: list[dict], bbox_key: str = "bbox") -> list[list[dict]]:
    """Groups regions into visual rows by Y-center proximity. Tolerance is half the page's own
    MEDIAN region height (one robust value per page), not each pair's own height -- real
    `regions.jsonl` content spans a huge size range within one page (confirmed: a single photo's
    page had regions from 9px to 245px tall out of a 256px-tall crop, since these are fine-grained
    content segments -- one balloon, one character silhouette -- not uniform panel-sized boxes). An
    earlier version used a per-pair adaptive tolerance (half of the *taller of the two* regions'
    own height); one oversized region (e.g. a near-full-page background) inflated its own local
    tolerance enough to wrongly absorb distant, unrelated regions into its row. A single
    page-median-based tolerance is not vulnerable to that specific failure mode."""
    if not regions:
        return []

    def y_center(r: dict) -> float:
        y0, y1 = r[bbox_key][1], r[bbox_key][3]
        return (y0 + y1) / 2

    def height(r: dict) -> float:
        return r[bbox_key][3] - r[bbox_key][1]

    tolerance = statistics.median(height(r) for r in regions) / 2

    ordered = sorted(regions, key=y_center)
    rows: list[list[dict]] = [[ordered[0]]]
    row_y_centers = [y_center(ordered[0])]
    for r in ordered[1:]:
        if abs(y_center(r) - row_y_centers[-1]) <= tolerance:
            rows[-1].append(r)
            row_y_centers[-1] = sum(y_center(x) for x in rows[-1]) / len(rows[-1])
        else:
            rows.append([r])
            row_y_centers.append(y_center(r))
    return rows


def _sort_reading_order(regions: list[dict], bbox_key: str = "bbox") -> list[dict]:
    """Raster reading order for a source page: row-cluster by Y-proximity, then sort each row
    left-to-right by X. Correctly handles a real multi-panel-per-row page (verified against a
    synthetic 3x3 grid, see test_build_pairs.py) where the naive (y, x)-tuple sort interleaves
    rows since it only breaks ties on *exact* Y equality.

    **NOT currently wired into build_pairs_for_row (2026-08-01)**: a real held-out-episode A/B test
    against `_sort_top_to_bottom` showed this makes the actual positioning-accuracy metric *worse*
    on this dataset (weighted mean error 1467px -> 1641-1665px; rank correlation 0.542 -> 0.39-0.48),
    despite being geometrically more correct for the source page's true layout. Kept here, tested,
    and documented as a real negative result -- not deleted -- in case future work (larger held-out
    sample, or a narrative-order cross-check from `sdd-comics-ai-script-context`) can resolve *why*
    it underperforms rather than just that it does. See
    flows/sdd-comics-ai-positioning/04-implementation-log.md for the full experiment record."""
    rows = _cluster_into_rows(regions, bbox_key)
    result: list[dict] = []
    for row in rows:
        result.extend(sorted(row, key=lambda r: r[bbox_key][0]))
    return result


def _group_by_kind(regions: list[dict], kind_key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for r in regions:
        groups[r[kind_key]].append(r)
    return groups


def build_pairs_for_row(
    row: dict, regions_by_page: dict[tuple[str, int], list[dict]], canvas_cache: dict[str, dict]
) -> list[PositionTrainingPair]:
    photo_file, page_index = row["photo_file"], row["page_index"]
    episode_file = row["episode_file"]
    cluster = set(row["ground_truth_cluster"])

    predicted = regions_by_page.get((photo_file, page_index), [])
    if not predicted:
        return []

    canvas = canvas_cache.setdefault(episode_file, pb.load_canvas_reference(episode_file))
    ground_truth = [r for r in canvas["regions"] if r["layer_index"] in cluster]
    if not ground_truth:
        return []

    # Source page (photo) space: naive (y, x) sort, kept deliberately despite a real, disclosed
    # theoretical flaw -- see _sort_reading_order below. Investigated 2026-08-01 (Anton asked how
    # reading order should work for a real multi-panel-per-row page): built and tested a real
    # row-clustering replacement (_sort_reading_order/_cluster_into_rows), confirmed correct on a
    # synthetic 3x3-grid case the naive sort gets wrong. But on a real held-out-episode A/B test
    # (same protocol as the Phase 5 learned-model comparison), row-clustering made the actual
    # evaluation metric *worse*, not better: weighted mean error rose from 1467px to 1641-1665px
    # (two tolerance variants tried), and reading-order rank correlation dropped from 0.542 to
    # 0.39-0.48. Diagnosed likely cause: real `regions.jsonl` content spans a huge size range
    # within one page (9-245px tall out of a 256px crop, since these are fine-grained content
    # segments -- one balloon, one character silhouette -- not uniform panel-sized boxes), which
    # defeats simple Y-proximity row-clustering more than expected. Per this flow's own established
    # precedent (ship what wins, not what sounds better -- same standard applied to the Phase 5
    # learned model), the naive sort stays the shipped default. Full record:
    # flows/sdd-comics-ai-positioning/_status.md and 04-implementation-log.md.
    predicted_sorted = _sort_top_to_bottom(predicted, "bbox")
    # global reading order across the whole page, before splitting by kind
    reading_order = {id(r): i for i, r in enumerate(predicted_sorted)}

    pred_by_kind = _group_by_kind(predicted_sorted, "predicted_kind")
    gt_by_kind = _group_by_kind(_sort_top_to_bottom(ground_truth, "bbox"), "kind")

    verified = text_context.get(episode_file)
    narrative_excerpt = verified.excerpt if verified is not None else None
    scene_dialogue = scene_text.text_for_cluster(episode_file, sorted(cluster))

    pairs: list[PositionTrainingPair] = []
    for kind, pred_group in pred_by_kind.items():
        gt_group = gt_by_kind.get(kind, [])
        n = min(len(pred_group), len(gt_group))
        for pred_region, gt_region in zip(pred_group[:n], gt_group[:n]):
            bbox = tuple(gt_region["bbox"])
            pairs.append(
                PositionTrainingPair(
                    episode_file=episode_file,
                    photo_file=photo_file,
                    region=RegionFeatures(
                        kind=kind,
                        kind_source="predicted",
                        local_bbox=tuple(pred_region["bbox"]),
                        page_index=page_index,
                        reading_order_index=reading_order[id(pred_region)],
                    ),
                    target_layer_index=gt_region["layer_index"],
                    target_bbox=bbox,
                    target_transform={"x": bbox[0], "y": bbox[1]},  # Must Have: X/Y only
                    match_confidence=row["confidence"],
                    text_context=scene_dialogue,
                    source_narrative_context=narrative_excerpt,
                )
            )
    return pairs


def build_all(out_dir: Path = DEFAULT_OUT_DIR) -> dict[str, int]:
    regions_by_page = _load_regions_by_page()
    canvas_cache: dict[str, dict] = {}
    pairs_by_episode: dict[str, list[PositionTrainingPair]] = defaultdict(list)

    for row in pb.iter_matched_alignment_rows():
        pairs = build_pairs_for_row(row, regions_by_page, canvas_cache)
        pairs_by_episode[row["episode_file"]].extend(pairs)

    out_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, int] = {}
    for episode_file, pairs in pairs_by_episode.items():
        stem = Path(episode_file).stem
        path = out_dir / f"{stem}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for pair in pairs:
                f.write(json.dumps(pair.to_jsonable()) + "\n")
        counts[episode_file] = len(pairs)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    counts = build_all(out_dir=args.out)
    total = sum(counts.values())
    print(f"Built {total} training pairs across {len(counts)} episodes -> {args.out}")
    for episode, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {episode}: {n} pairs")


if __name__ == "__main__":
    main()
