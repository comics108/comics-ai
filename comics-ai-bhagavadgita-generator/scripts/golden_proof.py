#!/usr/bin/env python3
"""Build the reproducible golden-chapter proof bundle and scale-out decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_proof(
    production_root: Path,
    *,
    lettering_relative: str = "lettering/fixtures-v1.json",
    validation_relative: str = "releases/golden-validation-v1.json",
    identity_relative: str = "identity-style/retrieval-v2.json",
) -> dict:
    relative_paths = (
        "gold-v1/manifest.json",
        "segmenter-competition/summary-v1.json",
        "identity-style/catalog-v2.json",
        identity_relative,
        "colour-registration/manifest-v2.json",
        "colourization/deterministic-v2.json",
        "colourization/learned-v1.json",
        "story-coverage/beats-v1.json",
        "story-coverage/coverage-v1.json",
        "lettering/authoritative-v1.json",
        lettering_relative,
        "compositions/golden-summary-v1.json",
        validation_relative,
    )
    records = []
    payloads = {}
    for relative in relative_paths:
        path = production_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"proof input missing: {relative}")
        payloads[relative] = json.loads(path.read_text(encoding="utf-8"))
        records.append({"path": relative, "sha256": file_sha256(path), "byte_size": path.stat().st_size})

    coverage = payloads["story-coverage/coverage-v1.json"]
    composition = payloads["compositions/golden-summary-v1.json"]
    validation = payloads[validation_relative]
    lettering = payloads[lettering_relative]
    chapter_proof = []
    composition_by_chapter = {item["chapter_order"]: item for item in composition["chapters"]}
    for chapter in coverage["chapters"]:
        comp = composition_by_chapter[chapter["chapter_order"]]
        chapter_proof.append({
            "chapter_order": chapter["chapter_order"],
            "beat_count": chapter["beat_count"],
            "accepted_coverage_count": chapter["accepted_coverage_count"],
            "generation_required_count": chapter["generation_required_count"],
            "composition_candidate_count": comp["candidate_count"],
            "release_state": "blocked",
        })
    gates = {item["dimension"]: item["state"] for item in validation["gates"]}
    next_actions = [
        {"order": 1, "action": "promote_or_replace_production_segmenter", "unblocks": ["technical", "asset_generation"]},
        {"order": 2, "action": "resolve_canonical_identity_and_accept_palette_pipeline", "unblocks": ["identity_style"]},
        {"order": 3, "action": "materialize_and_accept_assets_for_all_12_golden_beats", "unblocks": ["cultural_editorial", "art_direction"]},
        {"order": 4, "action": "supply_real_text_region_masks_and_reach_exact_ocr_6_of_6", "unblocks": ["lettering"]},
        {"order": 5, "action": "compose_and_validate_chapters_1_and_11_on_editor_viewer_devices", "unblocks": ["art_direction", "runtime"]},
    ]
    scale_allowed = all(item["release_state"] == "accepted" for item in chapter_proof) and all(
        state == "approved" for state in gates.values()
    )
    return {
        "schema_version": 1,
        "proof_scope": {"golden_chapters": [1, 11], "target_scroll_type": "vertical", "target_orientation": "portrait"},
        "artifacts": records,
        "chapters": chapter_proof,
        "lettering": {"accepted": lettering["accepted_count"], "total": lettering["fixture_count"], "state": lettering["release_state"]},
        "review_dimensions": gates,
        "golden_release_state": validation["release_state"],
        "scale_out_to_all_18": "allowed" if scale_allowed else "blocked",
        "next_actions": next_actions,
        "human_participation_required": False,
    }


def write_immutable(proof: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(proof, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--lettering", default="lettering/fixtures-v1.json")
    parser.add_argument("--validation", default="releases/golden-validation-v1.json")
    parser.add_argument("--identity", default="identity-style/retrieval-v2.json")
    args = parser.parse_args()
    proof = build_proof(args.production_root, lettering_relative=args.lettering,
                        validation_relative=args.validation, identity_relative=args.identity)
    write_immutable(proof, args.output)
    print(json.dumps({"golden_release_state": proof["golden_release_state"], "scale_out": proof["scale_out_to_all_18"]}))


if __name__ == "__main__":
    main()
