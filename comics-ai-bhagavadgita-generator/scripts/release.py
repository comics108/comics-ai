#!/usr/bin/env python3
"""Immutable, fail-closed production release compiler and six-dimension QA registry."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


GateState = Literal["approved", "rejected", "abstained", "stale"]
DIMENSIONS = (
    "technical", "identity_style", "art_direction", "lettering",
    "cultural_editorial", "runtime",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GateDecision:
    dimension: str
    state: GateState
    reviewer: str
    rationale: str
    evidence: tuple[str, ...]
    dependency_hashes: tuple[str, ...]

    def validate(self) -> None:
        if self.dimension not in DIMENSIONS:
            raise ValueError(f"unknown review dimension: {self.dimension}")
        if not self.reviewer or not self.rationale or not self.evidence:
            raise ValueError("gate reviewer, rationale, and evidence are mandatory")
        if any(len(value) != 64 for value in self.dependency_hashes):
            raise ValueError("dependency hashes must be SHA-256")


def evaluate_release(gates: list[GateDecision], artifacts: list[Path]) -> dict:
    for gate in gates:
        gate.validate()
    by_dimension = {gate.dimension: gate for gate in gates}
    if len(by_dimension) != len(gates):
        raise ValueError("each review dimension must occur exactly once")
    missing = sorted(set(DIMENSIONS) - set(by_dimension))
    artifact_records = []
    for artifact in artifacts:
        if not artifact.is_file():
            raise ValueError(f"release artifact is missing: {artifact}")
        artifact_records.append({"path": artifact.as_posix(), "sha256": file_sha256(artifact)})
    blockers = [f"missing_dimension:{item}" for item in missing]
    blockers += [f"{name}:{gate.state}" for name, gate in sorted(by_dimension.items()) if gate.state != "approved"]
    if not artifacts:
        blockers.append("release_artifacts_missing")
    return {
        "schema_version": 1,
        "release_state": "accepted" if not blockers else "blocked",
        "gates": [asdict(by_dimension[name]) for name in DIMENSIONS if name in by_dimension],
        "artifacts": artifact_records,
        "blockers": blockers,
    }


def invalidate_changed_dependencies(gates: list[GateDecision], current_hashes: set[str]) -> list[GateDecision]:
    result = []
    for gate in gates:
        stale = gate.state == "approved" and any(value not in current_hashes for value in gate.dependency_hashes)
        result.append(GateDecision(
            gate.dimension, "stale" if stale else gate.state, gate.reviewer,
            f"{gate.rationale}; upstream dependency changed" if stale else gate.rationale,
            gate.evidence + (("dependency_hash_mismatch",) if stale else ()), gate.dependency_hashes,
        ))
    return result


def publish_immutable(report: dict, report_path: Path, release_root: Path | None = None) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    if report["release_state"] != "accepted":
        return
    if release_root is None:
        raise ValueError("accepted release requires a release destination")
    staging = release_root.with_name(release_root.name + ".staging")
    if staging.exists() or release_root.exists():
        raise FileExistsError("release publication is immutable")
    staging.mkdir(parents=True)
    (staging / "manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    os.replace(staging, release_root)


def real_golden_gates(
    root: Path,
    *,
    lettering_relative: str = "lettering/fixtures-v1.json",
    identity_relative: str = "identity-style/retrieval-v2.json",
) -> list[GateDecision]:
    paths = {
        "segmenter": root / "segmenter-competition/summary-v1.json",
        "identity": root / identity_relative,
        "colour_d": root / "colourization/deterministic-v2.json",
        "colour_l": root / "colourization/learned-v1.json",
        "coverage": root / "story-coverage/coverage-v1.json",
        "lettering": root / lettering_relative,
        "composition": root / "compositions/golden-summary-v1.json",
    }
    payload = {name: json.loads(path.read_text(encoding="utf-8")) for name, path in paths.items()}
    hashes = {name: file_sha256(path) for name, path in paths.items()}
    no_segmenter = payload["segmenter"]["decision"] != "candidate_promoted"
    identity_abstained = any(item["identity_action"] == "abstained" for item in payload["identity"]["results"])
    colour_rejected = all(payload[name]["aggregate"]["decision"] != "accepted" for name in ("colour_d", "colour_l"))
    coverage_open = any(item["accepted_coverage_count"] < item["beat_count"] for item in payload["coverage"]["chapters"])
    return [
        GateDecision("technical", "rejected" if no_segmenter else "approved", "auto:technical-v1",
                     "production segmenter promotion is mandatory", (payload["segmenter"]["decision"],), (hashes["segmenter"],)),
        GateDecision("identity_style", "abstained" if identity_abstained or colour_rejected else "approved", "auto:identity-style-v1",
                     "identity and palette must both be resolved", ("identity_abstained" if identity_abstained else "identity_resolved", "colourizers_rejected" if colour_rejected else "colourizer_accepted"), (hashes["identity"], hashes["colour_d"], hashes["colour_l"])),
        GateDecision("art_direction", "rejected" if payload["composition"]["release_state"] != "accepted" else "approved", "auto:art-direction-v1",
                     "accepted complete vertical composition is mandatory", (payload["composition"]["release_state"],), (hashes["composition"],)),
        GateDecision("lettering", "rejected" if payload["lettering"]["release_state"] != "accepted" else "approved", "auto:lettering-v1",
                     "every authoritative fixture requires exact OCR", (f"{payload['lettering']['accepted_count']}/{payload['lettering']['fixture_count']}",), (hashes["lettering"],)),
        GateDecision("cultural_editorial", "abstained" if coverage_open else "approved", "auto:cultural-editorial-v1",
                     "canonical source-grounded beat coverage must close", ("coverage_open" if coverage_open else "coverage_closed",), (hashes["coverage"],)),
        GateDecision("runtime", "abstained", "auto:runtime-v1",
                     "device/editor/viewer checks require a release candidate archive", ("no_release_candidate_archive",), (hashes["composition"],)),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--production-root", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--lettering", default="lettering/fixtures-v1.json")
    parser.add_argument("--identity", default="identity-style/retrieval-v2.json")
    args = parser.parse_args()
    report = evaluate_release(real_golden_gates(
        args.production_root, lettering_relative=args.lettering, identity_relative=args.identity,
    ), [])
    publish_immutable(report, args.report)
    print(json.dumps({"release_state": report["release_state"], "blockers": len(report["blockers"])}))


if __name__ == "__main__":
    main()
