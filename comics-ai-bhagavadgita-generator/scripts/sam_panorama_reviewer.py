#!/usr/bin/env python3
"""Generate immutable SAM panorama proposals as one independent reviewer family."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def window_starts(width: int, window: int, overlap: int) -> tuple[int, ...]:
    if not 0 <= overlap < window:
        raise ValueError("overlap must be nonnegative and smaller than window")
    if width <= window:
        return (0,)
    starts = list(range(0, width - window + 1, window - overlap))
    if starts[-1] != width - window:
        starts.append(width - window)
    return tuple(starts)


def review_page(page_path: Path, checkpoint: Path, output_root: Path, *, page_id: str,
                window: int = 2048, overlap: int = 384, points_per_side: int = 12,
                crop_n_layers: int = 0) -> dict:
    import cv2
    from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

    image = cv2.imread(str(page_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode panorama: {page_path}")
    height, width = image.shape[:2]
    model = sam_model_registry["vit_b"](checkpoint=str(checkpoint)).to("cpu")
    generator = SamAutomaticMaskGenerator(
        model, points_per_side=points_per_side, points_per_batch=32, pred_iou_thresh=.86,
        stability_score_thresh=.90, crop_n_layers=crop_n_layers, min_mask_region_area=300,
    )
    mask_root = output_root / page_id
    mask_root.mkdir(parents=True, exist_ok=True)
    proposals = []
    for window_index, x0 in enumerate(window_starts(width, min(window, width), overlap)):
        x1 = min(width, x0 + window)
        crop = cv2.cvtColor(image[:, x0:x1], cv2.COLOR_BGR2RGB)
        for raw in generator.generate(crop):
            area = int(raw["area"])
            coverage = area / (crop.shape[0] * crop.shape[1])
            local_x, local_y, box_w, box_h = (int(round(value)) for value in raw["bbox"])
            rectangularity = area / max(1, box_w * box_h)
            touches_both_edges = local_x <= 2 and local_x + box_w >= crop.shape[1] - 2
            if not .001 <= coverage <= .25 or rectangularity >= .97 or touches_both_edges:
                continue
            mask = (raw["segmentation"].astype(np.uint8) * 255)
            identifier = f"{page_id}-w{window_index:02}-m{len(proposals):03}"
            path = mask_root / f"{identifier}.png"
            Image.fromarray(mask, mode="L").save(path)
            proposals.append({
                "id": identifier, "page_id": page_id, "window_index": window_index,
                "window_bbox": [x0, 0, x1, height],
                "page_bbox": [x0 + local_x, local_y, x0 + local_x + box_w, local_y + box_h],
                "mask_file": str(path), "mask_sha256": _sha256(path),
                "review_resolution": [x1 - x0, height], "area": area,
                "coverage": coverage, "rectangularity": rectangularity,
                "predicted_iou": float(raw["predicted_iou"]),
                "stability_score": float(raw["stability_score"]),
                "review_state": "proposed",
            })
    return {"page_id": page_id, "source_file": str(page_path), "source_sha256": _sha256(page_path),
            "page_resolution": [width, height], "window_count": len(window_starts(width, min(window, width), overlap)),
            "points_per_side": points_per_side, "crop_n_layers": crop_n_layers,
            "proposal_count": len(proposals), "proposals": proposals}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--page", type=Path, action="append", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--points-per-side", type=int, default=12)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    args = parser.parse_args()
    pages = [review_page(path, args.checkpoint, args.output_root, page_id=path.stem,
                         points_per_side=args.points_per_side, crop_n_layers=args.crop_n_layers)
             for path in args.page]
    report = {
        "schema_version": 1, "reviewer_family": "meta-sam-vit-b-automatic-mask-generator",
        "model_license": "Apache-2.0", "checkpoint_sha256": _sha256(args.checkpoint),
        "independence": "not_used_by_coco_instance_plus_edge_matting_gold_consensus",
        "configuration": {"points_per_side": args.points_per_side,
                          "crop_n_layers": args.crop_n_layers},
        "review_state": "proposed_requires_second_independent_family",
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
