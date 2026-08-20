#!/usr/bin/env python3
"""Validate colour-rendition SAM masks against independent registered B&W ink boundaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(sam_path: Path, bw_path: Path, *, min_boundary_support: float = .8) -> dict:
    import cv2

    sam = json.loads(sam_path.read_text(encoding="utf-8"))
    bw = json.loads(bw_path.read_text(encoding="utf-8"))
    bw_pages = {page["page_id"]: page for page in bw["pages"]}
    accepted, rejected = [], []
    for page in sam["pages"]:
        source = cv2.imread(bw_pages[page["page_id"]]["source_file"], cv2.IMREAD_GRAYSCALE)
        if source is None:
            raise ValueError("cannot decode registered B&W source")
        for proposal in page["proposals"]:
            mask = (np.asarray(Image.open(proposal["mask_file"])) > 0).astype(np.uint8)
            x0, _, x1, _ = proposal["window_bbox"]
            crop = source[:, x0:x1]
            ink_edges = cv2.Canny(crop, 60, 160) > 0
            boundary = (cv2.dilate(mask, np.ones((3, 3), np.uint8)) -
                        cv2.erode(mask, np.ones((3, 3), np.uint8))) > 0
            near_ink = cv2.dilate(ink_edges.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
            boundary_support = float((boundary & near_ink).sum() / max(1, boundary.sum()))
            bbox = proposal["page_bbox"]
            width_fraction = (bbox[2] - bbox[0]) / 2048
            height_fraction = (bbox[3] - bbox[1]) / page["page_resolution"][1]
            dark_density = float((crop[mask > 0] < 180).mean()) if mask.any() else 0.
            page_width, page_height = page["page_resolution"]
            margin_x, margin_y = max(2, round(page_width * .02)), max(2, round(page_height * .02))
            border_truncated = (bbox[0] <= margin_x or bbox[1] <= margin_y or
                                bbox[2] >= page_width - margin_x or bbox[3] >= page_height - margin_y)
            evidence = {
                "page_id": page["page_id"], "sam_proposal_id": proposal["id"],
                "window_index": proposal["window_index"], "page_bbox": bbox,
                "mask_file": proposal["mask_file"], "mask_sha256": proposal["mask_sha256"],
                "boundary_support": boundary_support, "coverage": proposal["coverage"],
                "width_fraction": width_fraction, "height_fraction": height_fraction,
                "dark_pixel_density": dark_density,
                "border_truncated": border_truncated,
            }
            complete = proposal["coverage"] >= .01 and width_fraction >= .08 and height_fraction >= .15
            inky = dark_density >= .02
            if boundary_support >= min_boundary_support and complete and inky and not border_truncated:
                accepted.append({**evidence, "review_state": "proposed_requires_context_qa"})
            else:
                rejected.append({**evidence, "review_state": "rejected_metric_gate"})
    return {
        "schema_version": 1, "sam_manifest_sha256": _sha256(sam_path),
        "bw_reviewer_manifest_sha256": _sha256(bw_path),
        "reviewer_families": [sam["reviewer_family"], "registered-bw-ink-boundary-v1"],
        "independence": "mask_from_author_colour_pixels_boundary_from_registered_bw_pixels",
        "gates": {"min_boundary_support": min_boundary_support, "min_viewport_coverage": .01,
                  "min_width_fraction": .08, "min_page_height_fraction": .15,
                  "min_dark_pixel_density": .02, "reject_border_truncated": True},
        "candidate_count": len(accepted), "rejected_count": len(rejected),
        "state": "proposed_requires_context_qa" if accepted else "no_candidates",
        "candidates": accepted, "rejected": rejected,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sam", type=Path, required=True)
    parser.add_argument("--bw", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); report = build(args.sam, args.bw)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    print(json.dumps({"state": report["state"], "candidates": report["candidate_count"]}))


if __name__ == "__main__":
    main()
