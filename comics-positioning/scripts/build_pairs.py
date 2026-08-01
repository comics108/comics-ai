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
