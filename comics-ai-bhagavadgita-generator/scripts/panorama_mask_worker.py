"""Classical independent-family panorama mask consensus worker (requires OpenCV)."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from torchvision.models.detection import (
    MaskRCNN_ResNet50_FPN_V2_Weights,
    maskrcnn_resnet50_fpn_v2,
)


def _boundary(mask: np.ndarray) -> np.ndarray:
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, kernel) > 0


def _boundary_f1(left: np.ndarray, right: np.ndarray) -> float:
    left_edge, right_edge = _boundary(left), _boundary(right)
    kernel = np.ones((5, 5), np.uint8)
    left_near = cv2.dilate(left_edge.astype(np.uint8), kernel) > 0
    right_near = cv2.dilate(right_edge.astype(np.uint8), kernel) > 0
    precision = float((right_edge & left_near).sum()) / max(1, int(right_edge.sum()))
    recall = float((left_edge & right_near).sum()) / max(1, int(left_edge.sum()))
    return 2 * precision * recall / max(1e-9, precision + recall)


def _bbox_iou(left: tuple[int, int, int, int], right: tuple[int, int, int, int]) -> float:
    x0, y0 = max(left[0], right[0]), max(left[1], right[1])
    x1, y1 = min(left[2], right[2]), min(left[3], right[3])
    intersection = max(0, x1 - x0) * max(0, y1 - y0)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / max(1, left_area + right_area - intersection)


def extract_consensus_masks(image_path: Path, output_dir: Path, limit: int) -> list[dict]:
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot decode panorama: {image_path}")
    height, width = image.shape[:2]
    output_dir.mkdir(parents=True, exist_ok=True)
    model = maskrcnn_resnet50_fpn_v2(weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT)
    model.eval()
    candidates: list[dict] = []
    kept_boxes: list[tuple[int, int, int, int]] = []
    tile_width = 1024
    origins = list(range(0, max(1, width - tile_width + 1), 768))
    if not origins or origins[-1] + tile_width < width:
        origins.append(max(0, width - tile_width))
    for tile_x in origins:
        if len(candidates) >= limit:
            break
        tile = image[:, tile_x:min(width, tile_x + tile_width)]
        rgb = cv2.cvtColor(tile, cv2.COLOR_BGR2RGB)
        tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        with torch.no_grad():
            prediction = model([tensor])[0]
        order = torch.argsort(prediction["scores"], descending=True).tolist()
        for prediction_index in order:
            if len(candidates) >= limit:
                break
            score = float(prediction["scores"][prediction_index])
            if score < .45:
                break
            raw_mask = (prediction["masks"][prediction_index, 0].numpy() >= .5).astype(np.uint8) * 255
            x, y, x_far, y_far = (round(value) for value in prediction["boxes"][prediction_index].tolist())
            w, h = x_far - x, y_far - y
            if w < 40 or h < 40:
                continue
            global_box = (tile_x + x, y, tile_x + x_far, y_far)
            if any(_bbox_iou(global_box, prior) > .55 for prior in kept_boxes):
                continue
            pad = max(8, round(max(w, h) * .08))
            x0, y0 = max(0, x - pad), max(0, y - pad)
            x1, y1 = min(tile.shape[1], x_far + pad), min(height, y_far + pad)
            crop = tile[y0:y1, x0:x1]
            local_a = raw_mask[y0:y1, x0:x1]
            # Independent boundary family: colour/edge foreground optimization initialized with a
            # narrow uncertainty band around the instance proposal, not its final mask values.
            kernel = np.ones((7, 7), np.uint8)
            eroded = cv2.erode(local_a, kernel)
            dilated = cv2.dilate(local_a, kernel)
            gc = np.full(local_a.shape, cv2.GC_BGD, np.uint8)
            gc[dilated > 0] = cv2.GC_PR_BGD
            gc[local_a > 0] = cv2.GC_PR_FGD
            gc[eroded > 0] = cv2.GC_FGD
            try:
                cv2.grabCut(crop, gc, None, np.zeros((1, 65)), np.zeros((1, 65)), 3, cv2.GC_INIT_WITH_MASK)
            except cv2.error:
                continue
            local_b = np.where((gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
            intersection = int(((local_a > 0) & (local_b > 0)).sum())
            union = int(((local_a > 0) | (local_b > 0)).sum())
            agreement = intersection / max(1, union)
            consensus = cv2.bitwise_and(local_a, local_b)
            nonzero = int((consensus > 0).sum())
            coverage = nonzero / max(1, consensus.size)
            bounds = cv2.boundingRect(consensus)
            rectangularity = nonzero / max(1, bounds[2] * bounds[3])
            boundary = _boundary_f1(local_a, local_b)
            # A detector occasionally proposes the complete review tile.  Such a mask can pass
            # coarse overlap metrics while still being an extraction artifact, so reject it via
            # geometry independently of the normal coverage/rectangularity gates.
            consumes_review_window = (
                x0 == 0
                and y0 == 0
                and x1 == tile.shape[1]
                and y1 == height
                and coverage > .90
            )
            if (
                agreement < .85
                or boundary < .75
                or not .01 < coverage < .95
                or rectangularity >= .98
                or consumes_review_window
            ):
                continue
            candidate_id = f"mask-{len(candidates):03}"
            mask_path = output_dir / f"{candidate_id}.png"
            cv2.imwrite(str(mask_path), consensus)
            checksum = hashlib.sha256(mask_path.read_bytes()).hexdigest()
            candidates.append({
                "id": candidate_id,
                "bbox": [tile_x + x0, y0, tile_x + x1, y1],
                "mask_file": str(mask_path.resolve()),
                "mask_sha256": checksum,
                "agreement_iou": agreement,
                "boundary_f1": boundary,
                "coverage": coverage,
                "rectangularity": rectangularity,
                "method_families": ["coco_instance_model", "edge_matting"],
                "review_resolution": [x1 - x0, y1 - y0],
                "page_resolution": [width, height],
                "coco_label": int(prediction["labels"][prediction_index]),
                "coco_score": score,
            })
            kept_boxes.append(global_box)
    return candidates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("image", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--limit", type=int, default=20)
    args = parser.parse_args()
    candidates = extract_consensus_masks(args.image, args.output_dir, args.limit)
    manifest_path = args.output_dir / "candidates.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(candidates, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)
    print(json.dumps(candidates, sort_keys=True))


if __name__ == "__main__":
    main()
