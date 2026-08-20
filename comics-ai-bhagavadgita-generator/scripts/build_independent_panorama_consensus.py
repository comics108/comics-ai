#!/usr/bin/env python3
"""Build strict symmetric consensus from two independent panorama reviewer families."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mask_iou(left: np.ndarray, right: np.ndarray) -> float:
    intersection = int((left & right).sum())
    return intersection / max(1, int((left | right).sum()))


def maximum_pairs(scores: list[tuple[float, int, int]], threshold: float) -> list[tuple[float, int, int]]:
    """Deterministic maximum-weight one-to-one greedy assignment above a strict threshold."""
    used_left, used_right, result = set(), set(), []
    for score, left, right in sorted(scores, key=lambda item: (-item[0], item[1], item[2])):
        if score < threshold or left in used_left or right in used_right:
            continue
        used_left.add(left); used_right.add(right); result.append((score, left, right))
    return result


def build(sam_path: Path, graph_path: Path, output_root: Path, *, threshold: float = .8) -> dict:
    import cv2

    sam = json.loads(sam_path.read_text(encoding="utf-8"))
    graph = json.loads(graph_path.read_text(encoding="utf-8"))
    if sam["reviewer_family"] == graph["reviewer_family"]:
        raise ValueError("consensus requires distinct reviewer families")
    graph_pages = {item["page_id"]: item for item in graph["pages"]}
    records, rejected_fragments, rejected_backgrounds = [], [], []
    output_root.mkdir(parents=True, exist_ok=True)
    for sam_page in sam["pages"]:
        graph_page = graph_pages[sam_page["page_id"]]
        graph_source = cv2.imread(graph_page["source_file"], cv2.IMREAD_GRAYSCALE)
        if graph_source is None:
            raise ValueError("cannot decode graph-review source for ink gate")
        for window_index in range(sam_page["window_count"]):
            left = [item for item in sam_page["proposals"] if item["window_index"] == window_index]
            right = [item for item in graph_page["proposals"] if item["window_index"] == window_index]
            left_masks = [np.asarray(Image.open(item["mask_file"])) > 0 for item in left]
            right_masks = [np.asarray(Image.open(item["mask_file"])) > 0 for item in right]
            scores = []
            for li, lmask in enumerate(left_masks):
                for ri, rmask in enumerate(right_masks):
                    lb, rb = left[li]["page_bbox"], right[ri]["page_bbox"]
                    if lb[2] <= rb[0] or rb[2] <= lb[0] or lb[3] <= rb[1] or rb[3] <= lb[1]:
                        continue
                    scores.append((mask_iou(lmask, rmask), li, ri))
            for score, li, ri in maximum_pairs(scores, threshold):
                sam_proposal = left[li]
                bbox = sam_proposal["page_bbox"]
                page_height = sam_page["page_resolution"][1]
                width_fraction = (bbox[2] - bbox[0]) / 2048
                height_fraction = (bbox[3] - bbox[1]) / page_height
                complete_enough = (sam_proposal["coverage"] >= .01 and
                                   width_fraction >= .08 and height_fraction >= .15)
                if not complete_enough:
                    rejected_fragments.append({
                        "page_id": sam_page["page_id"], "window_index": window_index,
                        "sam_proposal_id": sam_proposal["id"],
                        "graph_proposal_id": right[ri]["id"], "agreement_iou": score,
                        "coverage": sam_proposal["coverage"], "width_fraction": width_fraction,
                        "height_fraction": height_fraction,
                        "reason": "agreement_is_local_fragment_not_complete_foreground_instance",
                    })
                    continue
                consensus = left_masks[li] & right_masks[ri]
                window_x0 = sam_proposal["window_bbox"][0]
                source_crop = graph_source[:, window_x0:sam_proposal["window_bbox"][2]]
                edges = cv2.Canny(source_crop, 60, 160) > 0
                ink_edge_density = float(edges[consensus].mean()) if consensus.any() else 0.
                dark_pixel_density = float((source_crop[consensus] < 180).mean()) if consensus.any() else 0.
                if ink_edge_density < .01 or dark_pixel_density < .02:
                    rejected_backgrounds.append({
                        "page_id": sam_page["page_id"], "window_index": window_index,
                        "sam_proposal_id": sam_proposal["id"],
                        "graph_proposal_id": right[ri]["id"], "agreement_iou": score,
                        "ink_edge_density": ink_edge_density,
                        "dark_pixel_density": dark_pixel_density,
                        "reason": "agreement_is_low_ink_background_region",
                    })
                    continue
                identifier = f"{sam_page['page_id']}-w{window_index:02}-c{len(records):03}"
                path = output_root / f"{identifier}.png"
                Image.fromarray(consensus.astype(np.uint8) * 255, mode="L").save(path)
                records.append({
                    "id": identifier, "page_id": sam_page["page_id"], "window_index": window_index,
                    "sam_proposal_id": left[li]["id"], "graph_proposal_id": right[ri]["id"],
                    "agreement_iou": score, "mask_file": str(path), "mask_sha256": _sha256(path),
                    "page_bbox": left[li]["page_bbox"], "review_state": "accepted_pair_evidence",
                    "semantic_kind": None, "canonical_entity_id": None,
                })
    required = 30
    return {
        "schema_version": 1, "sam_manifest_sha256": _sha256(sam_path),
        "graph_manifest_sha256": _sha256(graph_path), "agreement_iou_threshold": threshold,
        "consensus_operation": "symmetric_bitmap_intersection",
        "matching": "deterministic_one_to_one_max_weight_greedy_per_window",
        "completeness_gate": {"min_viewport_coverage": .01, "min_width_fraction": .08,
                              "min_page_height_fraction": .15},
        "source_ink_gate": {"min_edge_density": .01, "min_dark_pixel_density": .02,
                            "source": "independent_graph_reviewer_source_pixels"},
        "accepted_pair_count": len(records), "required_pair_count": required,
        "rejected_fragment_count": len(rejected_fragments),
        "rejected_background_count": len(rejected_backgrounds),
        "shortfall": max(0, required - len(records)),
        "state": "ready" if len(records) >= required else "insufficient_for_gold_v2_3",
        "records": records, "rejected_fragments": rejected_fragments,
        "rejected_backgrounds": rejected_backgrounds,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.sam, args.graph, args.output_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    print(json.dumps({key: report[key] for key in ("state", "accepted_pair_count", "shortfall")}))


if __name__ == "__main__":
    main()
