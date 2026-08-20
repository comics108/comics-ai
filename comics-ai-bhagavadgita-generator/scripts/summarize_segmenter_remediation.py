#!/usr/bin/env python3
"""Immutable summary of autonomous production-segmenter remediation evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(gold: Path, border: Path, maskrcnn: Path) -> dict:
    gold_payload = json.loads(gold.read_text(encoding="utf-8"))
    border_payload = json.loads(border.read_text(encoding="utf-8"))
    maskrcnn_payload = json.loads(maskrcnn.read_text(encoding="utf-8"))
    if border_payload["promotion"] != "rejected" or maskrcnn_payload["promotion"] != "rejected":
        raise ValueError("remediation summary expects finalized rejected candidates")
    gold_hash = digest(gold)
    if maskrcnn_payload["dataset_manifest_sha256"] != gold_hash:
        raise ValueError("Mask R-CNN decision does not reference supplied Gold manifest")
    return {
        "schema_version": 1,
        "state": "production_cutting_blocked",
        "gold": {
            "version": gold_payload["dataset_version"], "manifest_sha256": gold_hash,
            "accepted": sum(item["accepted"] for item in gold_payload["annotations"]),
            "test": sum(item["accepted"] and item["split"] == "test" for item in gold_payload["annotations"]),
        },
        "candidates": [
            {"id": border_payload["candidate"], "promotion": "rejected",
             "decision_sha256": digest(border), "failures": border_payload["failures"]},
            {"id": maskrcnn_payload["candidate"], "promotion": "rejected",
             "decision_sha256": digest(maskrcnn), "failures": maskrcnn_payload["failures"]},
        ],
        "exhausted_paths": [
            "binary_border_background_matting",
            "compact_binary_unet_on_isolated_crops",
            "true_mask_maskrcnn_on_isolated_native_alpha_crops",
            "additional_epochs_without_new_panorama_or_rare_class_supervision",
        ],
        "forbidden_shortcut": {
            "action": "train_and_promote_coco_family_on_coco_edge_consensus_panorama_gold",
            "reason": "evaluation_family_participated_in_label_consensus",
        },
        "next_autonomous_input_contract": {
            "independent_panorama_instance_masks": 30,
            "source_disjoint_panorama_compositions": 2,
            "rare_semantic_examples_per_test_class": 5,
            "independent_reviewer_families": 2,
            "required_metrics": [
                "mask_iou", "boundary_f1", "semantic_macro_f1", "tiled_recall",
                "tiled_precision", "duplicate_rate", "collapsed_instance_count",
            ],
        },
        "human_participation_required": False,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--border-decision", type=Path, required=True)
    parser.add_argument("--maskrcnn-decision", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); report = summarize(args.gold, args.border_decision, args.maskrcnn_decision)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    print(json.dumps({"state": report["state"], "candidates": len(report["candidates"])}))


if __name__ == "__main__": main()
