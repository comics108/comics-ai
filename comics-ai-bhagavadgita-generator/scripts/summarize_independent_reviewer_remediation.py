#!/usr/bin/env python3
"""Summarize exhausted independent-reviewer paths and the next non-circular input contract."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(paths: list[Path]) -> dict:
    artifacts = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        artifacts.append({
            "path": str(path), "sha256": _sha256(path),
            "state": payload.get("state", payload.get("review_state")),
            "accepted": payload.get("accepted_pair_count", payload.get("candidate_count")),
            "rejected_fragments": payload.get("rejected_fragment_count", 0),
            "rejected_backgrounds": payload.get("rejected_background_count", 0),
        })
    return {
        "schema_version": 1, "state": "production_cutting_supervision_blocked",
        "human_participation_required": False,
        "artifacts": artifacts,
        "exhausted_paths": [
            "sam_sparse_plus_multiscale_graph_region_iou",
            "sam_dense_plus_multiscale_graph_region_iou",
            "sam_on_author_colour_plus_graph_on_registered_bw_region_iou",
            "sam_sparse_dense_crop_refined_cross_rendition_bw_boundary_support",
            "parameter_search_without_new_source_or_model_family",
        ],
        "forbidden_shortcuts": [
            "count_high_iou_fragments_as_complete_instances",
            "count_low_ink_background_regions_as_foreground",
            "count_border_truncated_or_compound_regions_as_assets",
            "lower_iou_or_completeness_gates_to_reach_quota",
            "reuse_coco_consensus_family_as_independent_reviewer",
        ],
        "next_autonomous_input_contract": {
            "new_object_level_reviewer_family": True,
            "license_and_checkpoint_provenance_required": True,
            "training_lineage_must_not_overlap_existing_coco_edge_consensus": True,
            "complete_non_border_foreground_instances": 30,
            "source_disjoint_compositions": 2,
            "rare_semantic_examples_per_test_class": 5,
            "mandatory_gates": ["bitmap_agreement", "object_completeness", "source_ink",
                                "border_truncation", "source_context_contact_sheet"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(); report = build(args.input)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2); stream.write("\n")
    print(json.dumps({"state": report["state"], "artifacts": len(report["artifacts"])}))


if __name__ == "__main__":
    main()
