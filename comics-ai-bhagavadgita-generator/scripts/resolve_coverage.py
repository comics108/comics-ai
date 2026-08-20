"""Resolve story-beat coverage in provenance order without guessed canonical mappings."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True)
class CoverageCandidate:
    asset_version_id: str
    semantic_work: Literal["bhagavad_gita", "gita_dhyanam", "unclassified"]
    chapter_orders: tuple[int, ...]
    mapping_state: Literal["confirmed", "inferred", "unmapped", "not_applicable"]
    review_state: Literal["accepted", "proposed", "rejected"]
    use_mode: Literal["direct", "reuse", "transform"]
    evidence: tuple[str, ...]


def resolve_beat(beat: dict, candidates: tuple[CoverageCandidate, ...]) -> dict:
    eligible = [
        item for item in candidates
        if item.semantic_work == "bhagavad_gita"
        and beat["chapter_order"] in item.chapter_orders
        and item.mapping_state == "confirmed"
        and item.review_state == "accepted"
    ]
    priority = {"direct": 0, "reuse": 1, "transform": 2}
    eligible.sort(key=lambda item: (priority[item.use_mode], item.asset_version_id))
    if eligible:
        chosen = eligible[0]
        state = {"direct": "accepted_source", "reuse": "reusable", "transform": "transformable"}[
            chosen.use_mode
        ]
        return {
            "beat_id": beat["id"], "requirement": "source-grounded visual composition",
            "state": state, "asset_version_ids": [chosen.asset_version_id],
            "proposed_action_ids": [], "paid_generation_suppressed": True,
            "evidence": list(chosen.evidence), "blockers": [],
        }
    considered = [
        {"asset_version_id": item.asset_version_id, "semantic_work": item.semantic_work,
         "mapping_state": item.mapping_state, "review_state": item.review_state,
         "excluded": (
             "noncanonical_gita_dhyanam" if item.semantic_work == "gita_dhyanam"
             else "mapping_or_review_not_accepted"
         )}
        for item in candidates
        if beat["chapter_order"] in item.chapter_orders or item.semantic_work == "gita_dhyanam"
    ]
    return {
        "beat_id": beat["id"], "requirement": "source-grounded visual composition",
        "state": "generation_required", "asset_version_ids": [],
        "proposed_action_ids": [f"local:grounded-visual:{beat['id']}:v1"],
        "paid_generation_suppressed": True,
        "evidence": ["no confirmed+accepted canonical source asset satisfies this beat"],
        "blockers": ["canonical_identity_unresolved", "production_segmenter_not_promoted"],
        "considered_candidates": considered,
    }


def build_coverage(beats_manifest: Path) -> dict:
    beats_payload = json.loads(beats_manifest.read_text(encoding="utf-8"))
    # These visually plausible mappings remain inferred/proposed under the approved source scope;
    # they are disclosed to the resolver but deliberately cannot satisfy coverage yet.
    candidates = (
        CoverageCandidate("panorama-bw-page-02:v1", "bhagavad_gita", (1,), "inferred", "proposed",
                          "direct", ("visual hypothesis: armies and central chariot",)),
        CoverageCandidate("panorama-bw-page-12:v1", "bhagavad_gita", (11,), "inferred", "proposed",
                          "direct", ("visual hypothesis: cosmic multi-faced figure",)),
        CoverageCandidate("lottie-gita-dhyanam:v1", "gita_dhyanam", (), "not_applicable", "accepted",
                          "reuse", ("reviewed standalone nine-stanza prologue",)),
    )
    chapters = []
    for chapter in beats_payload["chapters"]:
        coverage = [resolve_beat(beat, candidates) for beat in chapter["beats"]]
        chapters.append({
            "chapter_order": chapter["chapter_order"], "beat_count": len(chapter["beats"]),
            "coverage": coverage,
            "accepted_coverage_count": sum(item["state"] != "generation_required" for item in coverage),
            "generation_required_count": sum(item["state"] == "generation_required" for item in coverage),
        })
    return {
        "schema_version": 1,
        "beats_manifest_sha256": hashlib.sha256(beats_manifest.read_bytes()).hexdigest(),
        "resolution_order": ["accepted_source", "reusable", "transformable", "generation_required"],
        "candidates": [asdict(item) for item in candidates],
        "chapters": chapters,
        "paid_external_generation_authorized": False,
        "release_state": "blocked_pending_verified_assets",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("beats_manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build_coverage(args.beats_manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({
        "output": str(args.out),
        "chapters": len(report["chapters"]),
        "generation_required": sum(item["generation_required_count"] for item in report["chapters"]),
    }))


if __name__ == "__main__":
    main()
