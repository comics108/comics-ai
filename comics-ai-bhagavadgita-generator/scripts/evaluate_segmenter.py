"""Reproducible Gold v1 mask benchmark with fail-closed promotion decisions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageFilter

from gold_segmenter_data import load_gold_dataset, load_review_pair


@dataclass(frozen=True)
class MaskMetrics:
    iou: float
    boundary_f1: float


def mask_metrics(prediction: np.ndarray, target: np.ndarray, *, boundary_radius: int = 2) -> MaskMetrics:
    prediction = prediction.astype(bool)
    target = target.astype(bool)
    intersection = int((prediction & target).sum())
    union = int((prediction | target).sum())
    iou = intersection / max(1, union)

    def boundary(mask: np.ndarray) -> np.ndarray:
        padded = np.pad(mask, 1, constant_values=False)
        eroded = np.ones(mask.shape, dtype=bool)
        for dy in range(3):
            for dx in range(3):
                eroded &= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        return mask & ~eroded

    def dilate(mask: np.ndarray, radius: int = boundary_radius) -> np.ndarray:
        padded = np.pad(mask, radius, constant_values=False)
        result = np.zeros(mask.shape, dtype=bool)
        for dy in range(2 * radius + 1):
            for dx in range(2 * radius + 1):
                result |= padded[dy:dy + mask.shape[0], dx:dx + mask.shape[1]]
        return result

    predicted_boundary, target_boundary = boundary(prediction), boundary(target)
    precision = float((predicted_boundary & dilate(target_boundary)).sum()) / max(
        1, int(predicted_boundary.sum())
    )
    recall = float((target_boundary & dilate(predicted_boundary)).sum()) / max(
        1, int(target_boundary.sum())
    )
    boundary_f1 = 2 * precision * recall / max(1e-9, precision + recall)
    return MaskMetrics(iou=iou, boundary_f1=boundary_f1)


def bbox_fill(image: Image.Image) -> np.ndarray:
    return np.ones((image.height, image.width), dtype=bool)


def ink_threshold(image: Image.Image) -> np.ndarray:
    grayscale = image.convert("L").filter(ImageFilter.MedianFilter(3))
    return np.asarray(grayscale) < 235


def border_background_matting(image: Image.Image) -> np.ndarray:
    """Estimate paper colour from the crop border without Gold/detector mask input."""
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))
    luminance = rgb.mean(axis=2)
    raw = (distance > 22) | (luminance < float(np.median(border.mean(axis=1))) - 18)
    padded = np.pad(raw, 1, constant_values=False)
    votes = sum(
        padded[dy:dy + raw.shape[0], dx:dx + raw.shape[1]]
        for dy in range(3) for dx in range(3)
    )
    return votes >= 3


def compact_checkpoint_predictor(checkpoint: Path, device: str | None = None):
    import torch
    from compact_segmenter import CompactBinaryUNet

    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    if payload.get("model_family") != "compact_gold_unet":
        raise ValueError("unsupported compact segmenter checkpoint")
    model = CompactBinaryUNet().to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    train_height, train_width = payload["train_size"]
    threshold = float(payload["threshold"])

    def predict(image: Image.Image) -> np.ndarray:
        resized = image.resize((train_width, train_height), Image.Resampling.BILINEAR)
        tensor = (
            torch.from_numpy(np.asarray(resized).copy())
            .permute(2, 0, 1).float().unsqueeze(0).to(resolved_device) / 255.0
        )
        with torch.no_grad():
            probability = torch.sigmoid(model(tensor))[0, 0].cpu().numpy()
        prediction = Image.fromarray((probability >= threshold).astype(np.uint8) * 255)
        return np.asarray(prediction.resize(image.size, Image.Resampling.NEAREST)) > 0

    return predict


def legacy_unet_predictor(checkpoint: Path, repository_root: Path, device: str | None = None):
    import torch
    import torch.nn.functional as functional

    multimodal_scripts = repository_root / "apps/comics-ai/comics-ai-multimodal/scripts"
    sys.path.insert(0, str(multimodal_scripts))
    from segmenter_models.unet_baseline import UNetBaseline

    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    model = UNetBaseline(num_classes=4).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    def predict(image: Image.Image) -> np.ndarray:
        resized = image.resize((256, 256), Image.Resampling.BILINEAR)
        tensor = (
            torch.from_numpy(np.asarray(resized).copy())
            .permute(2, 0, 1).float().unsqueeze(0).to(resolved_device) / 255.0
        )
        with torch.no_grad():
            # Legacy class 0 (`art`) is the rasterized default outside its box labels. Treat only
            # explicit background/character/balloon regions as its foreground proposal.
            predicted = model(tensor).argmax(dim=1, keepdim=True).float()
            binary = functional.interpolate(
                (predicted != 0).float(), size=(image.height, image.width), mode="nearest"
            )[0, 0]
        return binary.cpu().numpy() > 0

    return predict


def legacy_maskrcnn_predictor(checkpoint: Path, repository_root: Path, device: str | None = None):
    import torch

    multimodal_scripts = repository_root / "apps/comics-ai/comics-ai-multimodal/scripts"
    sys.path.insert(0, str(multimodal_scripts))
    from segmenter_models.maskrcnn import build_model

    resolved_device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
    payload = torch.load(checkpoint, map_location=resolved_device, weights_only=False)
    model = build_model(pretrained=False).to(resolved_device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    def predict(image: Image.Image) -> np.ndarray:
        tensor = (
            torch.from_numpy(np.asarray(image).copy())
            .permute(2, 0, 1).float().to(resolved_device) / 255.0
        )
        with torch.no_grad():
            output = model([tensor])[0]
        keep = output["scores"] >= .45
        if not bool(keep.any()):
            return np.zeros((image.height, image.width), dtype=bool)
        return (output["masks"][keep, 0] >= .5).any(dim=0).cpu().numpy()

    return predict


def evaluate_predictor(
    manifest: Path,
    repository_root: Path,
    *,
    predictor_name: str,
    predictor: Callable[[Image.Image], np.ndarray],
    family: str,
) -> dict:
    dataset = load_gold_dataset(manifest, repository_root)
    annotations = [item for item in dataset.annotations if item.accepted and item.split == "test"]
    per_instance = []
    for annotation in annotations:
        image, mask = load_review_pair(annotation, repository_root)
        prediction = predictor(image)
        target = np.asarray(mask) > 0
        if prediction.shape != target.shape:
            raise ValueError(f"prediction geometry mismatch for {annotation.id}")
        metrics = mask_metrics(prediction, target)
        nonzero = int(prediction.sum())
        ys, xs = np.where(prediction)
        if nonzero:
            predicted_bbox_area = (int(xs.max()) - int(xs.min()) + 1) * (
                int(ys.max()) - int(ys.min()) + 1
            )
            rectangularity = nonzero / predicted_bbox_area
        else:
            rectangularity = 1.0
        per_instance.append({
            "id": annotation.id,
            **asdict(metrics),
            "prediction_coverage": nonzero / prediction.size,
            "prediction_rectangularity": rectangularity,
        })
    mean_iou = sum(item["iou"] for item in per_instance) / len(per_instance)
    mean_boundary = sum(item["boundary_f1"] for item in per_instance) / len(per_instance)
    recall = sum(item["iou"] >= .5 for item in per_instance) / len(per_instance)
    artifact_failures = sum(
        item["prediction_rectangularity"] >= .98
        or not .01 < item["prediction_coverage"] < .95
        for item in per_instance
    )
    circular_family = family in {
        "coco_instance_model", "edge_matting", "box_supervised_maskrcnn"
    }
    failures = []
    if mean_iou < .75:
        failures.append("mask_iou_below_0.75")
    if mean_boundary < .70:
        failures.append("boundary_f1_below_0.70")
    if recall < .85:
        failures.append("instance_recall_below_0.85")
    if artifact_failures:
        failures.append("automated_mask_artifact_failures")
    # This crop-level benchmark cannot certify tiled cross-window merging or semantic-kind quality.
    # Keep them explicit and fail closed instead of silently treating N/A as a pass.
    failures.extend(("duplicate_instance_rate_not_measured", "semantic_macro_f1_not_measured"))
    if circular_family:
        failures.append("evaluation_family_participated_in_gold_consensus")
    return {
        "schema_version": 1,
        "dataset_version": dataset.version,
        "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "predictor": predictor_name,
        "family": family,
        "test_instances": len(per_instance),
        "metrics": {
            "mean_mask_iou": mean_iou,
            "mean_boundary_f1": mean_boundary,
            "instance_recall_at_iou_0_5": recall,
            "automated_artifact_failure_count": artifact_failures,
            "duplicate_instance_rate": None,
            "semantic_kind_macro_f1": None,
        },
        "per_instance": per_instance,
        "known_bias": (
            "Panorama Gold uses COCO proposals plus independent edge matting; participating "
            "families cannot self-promote on this release."
        ),
        "promotion": "accepted" if not failures else "rejected",
        "promotion_failures": failures,
    }


def write_report(report: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--baseline", choices=("bbox-fill", "ink-threshold", "border-matting"))
    group.add_argument("--checkpoint", type=Path)
    group.add_argument("--legacy-unet", type=Path)
    group.add_argument("--legacy-maskrcnn", type=Path)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    repository_root = args.repository_root.resolve()
    predictors = {
        "bbox-fill": (bbox_fill, "geometry_baseline"),
        "ink-threshold": (ink_threshold, "classical_ink_threshold"),
        "border-matting": (border_background_matting, "classical_border_background_matting"),
    }
    selected_checkpoint = args.checkpoint or args.legacy_unet or args.legacy_maskrcnn
    if selected_checkpoint:
        checkpoint = (
            selected_checkpoint
            if selected_checkpoint.is_absolute()
            else repository_root / selected_checkpoint
        )
    if args.checkpoint:
        predictor = compact_checkpoint_predictor(checkpoint, args.device)
        predictor_name, family = checkpoint.stem, "compact_gold_unet"
    elif args.legacy_unet:
        predictor = legacy_unet_predictor(checkpoint, repository_root, args.device)
        predictor_name, family = "legacy-unet-baseline", "box_supervised_unet"
    elif args.legacy_maskrcnn:
        predictor = legacy_maskrcnn_predictor(checkpoint, repository_root, args.device)
        predictor_name, family = "legacy-maskrcnn", "box_supervised_maskrcnn"
    else:
        predictor, family = predictors[args.baseline]
        predictor_name = args.baseline
    report = evaluate_predictor(
        repository_root / "work/bhagavadgita/production/gold-v1/manifest.json",
        repository_root,
        predictor_name=predictor_name,
        predictor=predictor,
        family=family,
    )
    if selected_checkpoint:
        report["checkpoint_sha256"] = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    output = args.out if args.out.is_absolute() else repository_root / args.out
    write_report(report, output)
    print(json.dumps({"report": str(output), "promotion": report["promotion"], **report["metrics"]}))


if __name__ == "__main__":
    main()
