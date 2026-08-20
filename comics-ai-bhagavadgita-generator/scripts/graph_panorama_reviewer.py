#!/usr/bin/env python3
"""Generate independent graph-segmentation panorama proposals from source pixels."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image
from sam_panorama_reviewer import window_starts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def review_page(page_path: Path, output_root: Path, *, page_id: str,
                window: int = 2048, overlap: int = 384) -> dict:
    from skimage.measure import regionprops
    from skimage.segmentation import felzenszwalb

    image = np.asarray(Image.open(page_path).convert("RGB"))
    height, width = image.shape[:2]
    mask_root = output_root / page_id
    mask_root.mkdir(parents=True, exist_ok=True)
    proposals = []
    starts = window_starts(width, min(window, width), overlap)
    for window_index, x0 in enumerate(starts):
        x1 = min(width, x0 + window)
        crop = image[:, x0:x1]
        for scale in (80, 180, 400):
            labels = felzenszwalb(crop, scale=scale, sigma=.8, min_size=300, channel_axis=-1)
            for region in regionprops(labels + 1):
                local_y0, local_x0, local_y1, local_x1 = region.bbox
                area = int(region.area)
                box_area = max(1, (local_x1 - local_x0) * (local_y1 - local_y0))
                coverage = area / (crop.shape[0] * crop.shape[1])
                rectangularity = area / box_area
                touches_both_edges = local_x0 <= 2 and local_x1 >= crop.shape[1] - 2
                if not .001 <= coverage <= .25 or rectangularity >= .97 or touches_both_edges:
                    continue
                mask = np.where(labels == region.label - 1, 255, 0).astype(np.uint8)
                identifier = f"{page_id}-w{window_index:02}-s{scale}-m{len(proposals):04}"
                path = mask_root / f"{identifier}.png"
                Image.fromarray(mask, mode="L").save(path)
                proposals.append({
                    "id": identifier, "page_id": page_id, "window_index": window_index,
                    "graph_scale": scale, "window_bbox": [x0, 0, x1, height],
                    "page_bbox": [x0 + local_x0, local_y0, x0 + local_x1, local_y1],
                    "mask_file": str(path), "mask_sha256": _sha256(path),
                    "review_resolution": [x1 - x0, height], "area": area,
                    "coverage": coverage, "rectangularity": rectangularity,
                    "review_state": "proposed",
                })
    return {"page_id": page_id, "source_file": str(page_path), "source_sha256": _sha256(page_path),
            "page_resolution": [width, height], "window_count": len(starts),
            "proposal_count": len(proposals), "proposals": proposals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    pages = [review_page(path, args.output_root, page_id=path.stem) for path in args.page]
    report = {
        "schema_version": 2, "reviewer_family": "felzenszwalb-pixel-graph-multiscale-v2",
        "implementation_license": "BSD-3-Clause scikit-image",
        "independence": "no_neural_checkpoint_no_coco_no_sam_output_or_prompt",
        "graph_scales": [80, 180, 400],
        "review_state": "proposed_requires_independent_consensus",
        "page_count": len(pages), "proposal_count": sum(item["proposal_count"] for item in pages),
        "pages": pages,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"pages": report["page_count"], "proposals": report["proposal_count"],
                      "state": report["review_state"]}))


if __name__ == "__main__":
    main()
