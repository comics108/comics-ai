#!/usr/bin/env python3
"""Dependency-free overlapping-window instance merge and duplicate-rate evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image

from evaluate_segmenter import border_background_matting


def connected_bboxes(mask: np.ndarray, *, min_area: int = 64) -> list[tuple[int, int, int, int]]:
    mask = mask.astype(bool)
    visited = np.zeros(mask.shape, dtype=bool)
    height, width = mask.shape
    boxes = []
    for y, x in zip(*np.where(mask & ~visited)):
        if visited[y, x]:
            continue
        queue, area = deque([(int(y), int(x))]), 0
        min_x = max_x = int(x)
        min_y = max_y = int(y)
        visited[y, x] = True
        while queue:
            cy, cx = queue.popleft()
            area += 1
            min_x, max_x = min(min_x, cx), max(max_x, cx)
            min_y, max_y = min(min_y, cy), max(max_y, cy)
            for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    queue.append((ny, nx))
        if area >= min_area:
            boxes.append((min_x, min_y, max_x + 1, max_y + 1))
    return boxes


def _intersection(left, right) -> int:
    return max(0, min(left[2], right[2]) - max(left[0], right[0])) * max(
        0, min(left[3], right[3]) - max(left[1], right[1])
    )


def _area(box) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def bbox_iou(left, right) -> float:
    intersection = _intersection(left, right)
    return intersection / max(1, _area(left) + _area(right) - intersection)


def bbox_match(left, right, *, iou_threshold: float = .3, containment_threshold: float = .6) -> bool:
    intersection = _intersection(left, right)
    containment = intersection / max(1, min(_area(left), _area(right)))
    return bbox_iou(left, right) >= iou_threshold or containment >= containment_threshold


def merge_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    pending = list(boxes)
    changed = True
    while changed:
        changed = False
        merged = []
        while pending:
            current = pending.pop(0)
            for index, other in enumerate(pending):
                intersection = _intersection(current, other)
                if bbox_iou(current, other) >= .25 or intersection / max(1, min(_area(current), _area(other))) >= .5:
                    current = (min(current[0], other[0]), min(current[1], other[1]), max(current[2], other[2]), max(current[3], other[3]))
                    pending.pop(index)
                    changed = True
                    break
            merged.append(current)
        pending = merged
    return sorted(pending)


def tiled_instances(image: Image.Image, *, tile_width: int = 512, overlap: int = 128, min_area: int = 64):
    if tile_width <= overlap or overlap < 0:
        raise ValueError("tile width must exceed non-negative overlap")
    origins = list(range(0, max(1, image.width - tile_width + 1), tile_width - overlap))
    last = max(0, image.width - tile_width)
    if not origins or origins[-1] != last:
        origins.append(last)
    raw = []
    for origin in origins:
        tile = image.crop((origin, 0, min(image.width, origin + tile_width), image.height))
        for x0, y0, x1, y1 in connected_bboxes(border_background_matting(tile), min_area=min_area):
            raw.append((x0 + origin, y0, x1 + origin, y1))
    return raw, merge_boxes(raw)


def one_to_one_matches(predictions, truth) -> list[tuple[int, int]]:
    """Maximum bipartite matching prevents one tile-wide prediction satisfying many instances."""
    edges = [[index for index, target in enumerate(truth) if bbox_match(prediction, target)]
             for prediction in predictions]
    truth_owner: dict[int, int] = {}

    def assign(prediction_index: int, seen: set[int]) -> bool:
        for truth_index in edges[prediction_index]:
            if truth_index in seen:
                continue
            seen.add(truth_index)
            owner = truth_owner.get(truth_index)
            if owner is None or assign(owner, seen):
                truth_owner[truth_index] = prediction_index
                return True
        return False

    for prediction_index in range(len(predictions)):
        assign(prediction_index, set())
    return sorted((prediction, truth_index) for truth_index, prediction in truth_owner.items())


def evaluate(manifest: Path, repository_root: Path) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    # This fixture measures full-page overlapping-window behavior on every accepted panorama
    # annotation. Gold v2 deliberately keeps consensus panoramas out of its metric test split, but
    # their immutable geometry remains valid stress-test evidence for merge/duplicate behavior.
    held_out = [item for item in payload["annotations"] if item["accepted"] and item["source_kind"] == "panorama"]
    by_source: dict[str, list[dict]] = {}
    for item in held_out:
        by_source.setdefault(item["source_composition_id"], []).append(item)
    pages = []
    all_matches = all_duplicates = all_predictions = all_collapsed = 0
    for source, items in sorted(by_source.items()):
        page = source.rsplit("-", 1)[-1]
        image_path = repository_root / f"work/bhagavadgita/production/gold-v1/panorama-source/bw-page-{page}.jpg"
        with Image.open(image_path) as opened:
            raw, merged = tiled_instances(opened.convert("RGB"))
        truth = []
        for item in items:
            evidence = next(value for value in item["review_evidence"] if value.startswith("render_bbox:"))
            truth.append(tuple(int(value) for value in evidence.removeprefix("render_bbox:").split(",")))
        per_truth = [sum(bbox_match(prediction, target) for prediction in merged) for target in truth]
        per_prediction = [sum(bbox_match(prediction, target) for target in truth) for prediction in merged]
        matches = len(one_to_one_matches(merged, truth))
        duplicates = sum(max(0, value - 1) for value in per_truth)
        collapsed = sum(max(0, value - 1) for value in per_prediction)
        all_matches += matches
        all_duplicates += duplicates
        all_predictions += len(merged)
        all_collapsed += collapsed
        pages.append({"source": source, "raw_instances": len(raw), "merged_instances": len(merged), "truth_instances": len(truth), "one_to_one_matches": matches, "duplicate_matches": duplicates, "collapsed_truth_matches": collapsed})
    truth_count = sum(item["truth_instances"] for item in pages)
    recall = all_matches / max(1, truth_count)
    precision = all_matches / max(1, all_predictions)
    duplicate_rate = all_duplicates / max(1, all_matches + all_duplicates)
    return {
        "schema_version": 1, "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "predictor": "border-matting", "pages": pages,
        "instance_recall_at_bbox_match": recall,
        "instance_precision_at_bbox_match": precision,
        "duplicate_instance_rate": duplicate_rate,
        "collapsed_truth_match_count": all_collapsed,
        "promotion_gate": "accepted" if truth_count and duplicate_rate <= .03 and recall >= .85 and precision >= .85 and all_collapsed == 0 else "rejected",
        "limitations": ["bbox-level merge evaluation", "semantic classification evaluated separately"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = evaluate(args.manifest, args.repository_root.resolve())
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"duplicate_instance_rate": report["duplicate_instance_rate"], "gate": report["promotion_gate"]}))


if __name__ == "__main__":
    main()
