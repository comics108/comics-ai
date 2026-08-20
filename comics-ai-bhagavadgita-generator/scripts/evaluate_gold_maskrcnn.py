#!/usr/bin/env python3
"""Independent true-mask and semantic evaluation for Gold Mask R-CNN checkpoints."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from evaluate_segmenter import mask_metrics
from gold_segmenter_data import load_gold_dataset, load_review_pair
from train_gold_maskrcnn import KINDS, build_model


def per_class_f1(truth: list[int], predicted: list[int | None]) -> dict[int, float]:
    result = {}
    for label in sorted(set(truth)):
        tp = sum(t == label and p == label for t, p in zip(truth, predicted))
        fp = sum(t != label and p == label for t, p in zip(truth, predicted))
        fn = sum(t == label and p != label for t, p in zip(truth, predicted))
        precision, recall = tp / max(1, tp + fp), tp / max(1, tp + fn)
        result[label] = 2 * precision * recall / max(1e-9, precision + recall)
    return result


def macro_f1(truth: list[int], predicted: list[int | None]) -> float:
    scores = per_class_f1(truth, predicted).values()
    return sum(scores) / len(scores)


def evaluate(manifest: Path, repository_root: Path, checkpoint: Path, *, device="cpu", score_threshold=.3):
    import torch

    payload = torch.load(checkpoint, map_location=device, weights_only=False)
    if payload.get("model_family") != "gold_true_mask_maskrcnn_v1":
        raise ValueError("unsupported Mask R-CNN checkpoint")
    model = build_model(pretrained=False).to(device)
    model.load_state_dict(payload["model_state"]); model.eval()
    gold = load_gold_dataset(manifest, repository_root)
    test = [item for item in gold.annotations if item.accepted and item.split == "test"]
    rows, truth_labels, predicted_labels = [], [], []
    with torch.no_grad():
        for item in test:
            image, target_image = load_review_pair(item, repository_root)
            tensor = torch.from_numpy(np.asarray(image).copy()).permute(2, 0, 1).float().to(device) / 255
            output = model([tensor])[0]
            keep = output["scores"] >= score_threshold
            target = np.asarray(target_image) > 0
            boundary_radius = max(2, math.ceil(2 * max(target.shape) / 768))
            best = None
            for index in torch.where(keep)[0].tolist():
                prediction = output["masks"][index, 0].cpu().numpy() >= .5
                metrics = mask_metrics(prediction, target, boundary_radius=boundary_radius)
                candidate = (metrics.iou, metrics.boundary_f1, int(output["labels"][index]), float(output["scores"][index]))
                if best is None or candidate[0] > best[0]:
                    best = candidate
            truth_label = KINDS.index(item.semantic_kind) + 1
            predicted_label = best[2] if best else None
            truth_labels.append(truth_label); predicted_labels.append(predicted_label)
            rows.append({"id": item.id, "semantic_kind": item.semantic_kind,
                         "iou": best[0] if best else 0, "boundary_f1": best[1] if best else 0,
                         "predicted_kind": KINDS[predicted_label - 1] if predicted_label else None,
                         "score": best[3] if best else None, "boundary_radius": boundary_radius})
    mean_iou = sum(row["iou"] for row in rows) / len(rows)
    mean_boundary = sum(row["boundary_f1"] for row in rows) / len(rows)
    recall = sum(row["iou"] >= .5 for row in rows) / len(rows)
    semantic = macro_f1(truth_labels, predicted_labels)
    semantic_by_label = per_class_f1(truth_labels, predicted_labels)
    majority = max(set(truth_labels), key=truth_labels.count)
    majority_baseline = macro_f1(truth_labels, [majority] * len(truth_labels))
    failures = []
    if mean_iou < .75: failures.append("mask_iou_below_0.75")
    if mean_boundary < .70: failures.append("boundary_f1_below_0.70")
    if recall < .85: failures.append("instance_recall_below_0.85")
    if semantic <= majority_baseline: failures.append("semantic_macro_f1_not_above_majority_baseline")
    if any(score == 0 for score in semantic_by_label.values()): failures.append("semantic_test_class_f1_zero")
    failures += ["tiled_duplicate_instance_gate_pending"]
    return {"schema_version": 1, "dataset_version": gold.version,
            "dataset_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
            "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            "predictor": checkpoint.stem, "family": "gold_true_mask_maskrcnn_v1",
            "test_instances": len(rows), "score_threshold": score_threshold,
            "metrics": {"mean_mask_iou": mean_iou, "mean_boundary_f1": mean_boundary,
                        "instance_recall_at_iou_0_5": recall, "semantic_kind_macro_f1": semantic,
                        "semantic_majority_baseline_macro_f1": majority_baseline,
                        "semantic_f1_by_kind": {KINDS[label - 1]: score for label, score in semantic_by_label.items()}},
            "per_instance": rows, "promotion": "accepted" if not failures else "rejected",
            "promotion_failures": failures}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--manifest", type=Path, required=True); parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True); parser.add_argument("--device", default="cpu")
    args = parser.parse_args(); root=args.repository_root.resolve()
    report=evaluate(args.manifest if args.manifest.is_absolute() else root/args.manifest, root,
                    args.checkpoint if args.checkpoint.is_absolute() else root/args.checkpoint, device=args.device)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream: json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2); stream.write("\n")
    print(json.dumps({"promotion":report["promotion"],**report["metrics"]}))


if __name__ == "__main__": main()
