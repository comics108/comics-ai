#!/usr/bin/env python3
"""Audit whether current Gold can support non-circular production segmenter promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def audit(manifest: Path, tiled_report: Path | None = None) -> dict:
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    annotations = [item for item in payload["annotations"] if item["accepted"]]
    held_out = [item for item in annotations if item["split"] == "test"]
    principal = [item for item in annotations if item["principal_character"]]
    semantic_counts: dict[str, int] = {}
    for item in held_out:
        semantic_counts[item["semantic_kind"]] = semantic_counts.get(item["semantic_kind"], 0) + 1
    independent_held_out = [
        item for item in held_out
        if item["label_origin"] == "human_corrected"
        or "families:native_alpha" in item.get("review_evidence", [])
    ]
    tiled_payload = None
    if tiled_report is not None:
        tiled_payload = json.loads(tiled_report.read_text(encoding="utf-8"))
        if tiled_payload.get("dataset_manifest_sha256") != hashlib.sha256(manifest.read_bytes()).hexdigest():
            raise ValueError("tiled fixture was evaluated against a different Gold manifest")
    requirements = {
        "accepted_instances_at_least_120": len(annotations) >= 120,
        "held_out_instances_at_least_30": len(held_out) >= 30,
        "independent_held_out_instances_at_least_30": len(independent_held_out) >= 30,
        "held_out_semantic_kinds_at_least_2": len(semantic_counts) >= 2,
        "principal_identity_labels_present": bool(principal) and all(item["canonical_entity_id"] for item in principal),
        "tiled_duplicate_metric_fixture_present": tiled_payload is not None,
    }
    missing = [name for name, passed in requirements.items() if not passed]
    return {
        "schema_version": 1,
        "source_dataset_version": payload["dataset_version"],
        "source_manifest_sha256": hashlib.sha256(manifest.read_bytes()).hexdigest(),
        "accepted_count": len(annotations),
        "held_out_count": len(held_out),
        "independent_held_out_count": len(independent_held_out),
        "principal_identity_count": len(principal),
        "held_out_semantic_counts": semantic_counts,
        "tiled_fixture": None if tiled_payload is None else {
            "report_sha256": hashlib.sha256(tiled_report.read_bytes()).hexdigest(),
            "duplicate_instance_rate": tiled_payload["duplicate_instance_rate"],
            "instance_recall_at_bbox_match": tiled_payload["instance_recall_at_bbox_match"],
            "instance_precision_at_bbox_match": tiled_payload["instance_precision_at_bbox_match"],
            "collapsed_truth_match_count": tiled_payload["collapsed_truth_match_count"],
            "candidate_gate": tiled_payload["promotion_gate"],
        },
        "requirements": requirements,
        "gold_v2_readiness": "ready" if not missing else "blocked",
        "missing": missing,
        "autonomous_actions": [
            "derive_non_circular_held_out_masks_from_native_alpha_or independently reviewed corrections",
            "add source-disjoint semantic-kind and canonical-principal-identity labels",
            "add full-panorama overlapping-window merge fixture and duplicate-instance metric",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--tiled-report", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.manifest, args.tiled_report)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"readiness": result["gold_v2_readiness"], "missing": len(result["missing"])}))


if __name__ == "__main__":
    main()
