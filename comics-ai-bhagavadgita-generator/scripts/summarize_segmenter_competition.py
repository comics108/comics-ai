"""Publish an immutable, evidence-ranked Task 11.2 competition decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def summarize(report_paths: list[Path]) -> dict:
    if not report_paths:
        raise ValueError("at least one segmenter report is required")
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in report_paths]
    dataset_versions = {report["dataset_version"] for report in reports}
    manifest_hashes = {report["dataset_manifest_sha256"] for report in reports}
    test_counts = {report["test_instances"] for report in reports}
    if len(dataset_versions) != 1 or len(manifest_hashes) != 1 or len(test_counts) != 1:
        raise ValueError("competition reports must use one identical Gold evaluation set")
    ranking = sorted(
        ({
            "predictor": report["predictor"],
            "family": report["family"],
            "report_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "promotion": report["promotion"],
            "promotion_failures": report["promotion_failures"],
            **report["metrics"],
        } for report, path in zip(reports, report_paths)),
        key=lambda item: (
            -item["mean_mask_iou"],
            -item["mean_boundary_f1"],
            item["predictor"],
        ),
    )
    eligible = [item for item in ranking if item["promotion"] == "accepted"]
    return {
        "schema_version": 1,
        "task": "11.2-compact-local-segmenter-competition",
        "dataset_version": next(iter(dataset_versions)),
        "dataset_manifest_sha256": next(iter(manifest_hashes)),
        "test_instances": next(iter(test_counts)),
        "ranking": ranking,
        "decision": "promoted" if eligible else "no_candidate_promoted",
        "selected_predictor": eligible[0]["predictor"] if eligible else None,
        "deployment_effect": (
            "production_cutting_remains_fail_closed; legacy checkpoints remain references only"
            if not eligible
            else "publish selected checkpoint only after immutable version-store validation"
        ),
        "license_evidence": {
            "compact_gold_unet": "repository-owned model code; torch 2.13.0 License-Expression metadata",
            "box_supervised_unet": "repository-owned model code; torch 2.13.0 License-Expression metadata",
            "box_supervised_maskrcnn": (
                "repository-owned fine-tuning code; torchvision 0.28.0 installed metadata says BSD; "
                "pretrained-weight provenance does not override failed metrics/circularity"
            ),
            "classical_baselines": "repository-owned evaluation code; Pillow 12.3.0 MIT-CMU metadata",
            "classical_border_background_matting": "repository-owned NumPy/Pillow implementation",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    summary = summarize(args.reports)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({
        "decision": summary["decision"],
        "ranking": [item["predictor"] for item in summary["ranking"]],
    }))


if __name__ == "__main__":
    main()
