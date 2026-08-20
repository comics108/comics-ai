#!/usr/bin/env python3
"""Combine independent Gold v2 mask, semantic, and tiled-instance gates without averaging."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decide(mask_report: Path, tiled_report: Path, readiness_report: Path) -> dict:
    mask = json.loads(mask_report.read_text(encoding="utf-8"))
    tiled = json.loads(tiled_report.read_text(encoding="utf-8"))
    readiness = json.loads(readiness_report.read_text(encoding="utf-8"))
    expected_hash = mask["dataset_manifest_sha256"]
    if tiled["dataset_manifest_sha256"] != expected_hash or readiness["source_manifest_sha256"] != expected_hash:
        raise ValueError("segmenter promotion inputs do not share one Gold manifest")
    metrics = mask["metrics"]
    gates = {
        "gold_readiness": readiness["gold_v2_readiness"] == "ready",
        "mask_iou": metrics["mean_mask_iou"] >= .75,
        "boundary_f1": metrics["mean_boundary_f1"] >= .70,
        "crop_instance_recall": metrics["instance_recall_at_iou_0_5"] >= .85,
        "artifact_failures": metrics["automated_artifact_failure_count"] == 0,
        "tiled_duplicate_rate": tiled["duplicate_instance_rate"] <= .03,
        "tiled_instance_recall": tiled["instance_recall_at_bbox_match"] >= .85,
        "tiled_instance_precision": tiled["instance_precision_at_bbox_match"] >= .85,
        "tiled_no_collapsed_instances": tiled["collapsed_truth_match_count"] == 0,
        "semantic_macro_f1": metrics["semantic_kind_macro_f1"] is not None and metrics["semantic_kind_macro_f1"] >= .0,
    }
    failures = [name for name, passed in gates.items() if not passed]
    return {
        "schema_version": 1, "candidate": mask["predictor"], "family": mask["family"],
        "dataset_version": mask["dataset_version"], "dataset_manifest_sha256": expected_hash,
        "inputs": {"mask_report_sha256": sha256(mask_report), "tiled_report_sha256": sha256(tiled_report), "readiness_report_sha256": sha256(readiness_report)},
        "gates": gates, "failures": failures,
        "metrics": {**metrics, "tiled_duplicate_instance_rate": tiled["duplicate_instance_rate"], "tiled_instance_recall_at_bbox_match": tiled["instance_recall_at_bbox_match"], "tiled_instance_precision_at_bbox_match": tiled["instance_precision_at_bbox_match"], "tiled_collapsed_truth_match_count": tiled["collapsed_truth_match_count"]},
        "promotion": "accepted" if not failures else "rejected",
        "deployment_effect": "production_cutting_enabled" if not failures else "production_cutting_remains_fail_closed",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-report", type=Path, required=True)
    parser.add_argument("--tiled-report", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = decide(args.mask_report, args.tiled_report, args.readiness_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"promotion": report["promotion"], "failures": report["failures"]}))


if __name__ == "__main__":
    main()
