#!/usr/bin/env python3
"""Final non-compensating promotion decision for Gold v2.2 Mask R-CNN."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(crop_report: Path, tiled_report: Path) -> dict:
    crop = json.loads(crop_report.read_text(encoding="utf-8"))
    tiled = json.loads(tiled_report.read_text(encoding="utf-8"))
    if crop["dataset_manifest_sha256"] != tiled["dataset_manifest_sha256"]:
        raise ValueError("crop and tiled reports use different Gold manifests")
    if crop["checkpoint_sha256"] != tiled["checkpoint_sha256"]:
        raise ValueError("crop and tiled reports use different checkpoints")
    metrics = crop["metrics"]
    gates = {
        "mask_iou": metrics["mean_mask_iou"] >= .75,
        "boundary_f1": metrics["mean_boundary_f1"] >= .70,
        "crop_instance_recall": metrics["instance_recall_at_iou_0_5"] >= .85,
        "semantic_above_majority": metrics["semantic_kind_macro_f1"] > metrics["semantic_majority_baseline_macro_f1"],
        "semantic_every_test_class_nonzero": all(value > 0 for value in metrics["semantic_f1_by_kind"].values()),
        "tiled_recall": tiled["instance_recall_at_bbox_match"] >= .85,
        "tiled_precision": tiled["instance_precision_at_bbox_match"] >= .85,
        "tiled_duplicate_rate": tiled["duplicate_instance_rate"] <= .03,
        "tiled_no_collapsed_instances": tiled["collapsed_truth_match_count"] == 0,
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1, "candidate": crop["predictor"], "family": crop["family"],
        "dataset_version": crop["dataset_version"], "dataset_manifest_sha256": crop["dataset_manifest_sha256"],
        "checkpoint_sha256": crop["checkpoint_sha256"],
        "inputs": {"crop_report_sha256": digest(crop_report), "tiled_report_sha256": digest(tiled_report)},
        "gates": gates, "failures": failures, "promotion": "accepted" if not failures else "rejected",
        "deployment_effect": "production_cutting_enabled" if not failures else "production_cutting_remains_fail_closed",
    }


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--crop-report",type=Path,required=True); parser.add_argument("--tiled-report",type=Path,required=True); parser.add_argument("--out",type=Path,required=True)
    args=parser.parse_args(); report=decide(args.crop_report,args.tiled_report); args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open("x",encoding="utf-8") as stream: json.dump(report,stream,ensure_ascii=False,sort_keys=True,indent=2); stream.write("\n")
    print(json.dumps({"promotion":report["promotion"],"failures":report["failures"]}))


if __name__=="__main__": main()
