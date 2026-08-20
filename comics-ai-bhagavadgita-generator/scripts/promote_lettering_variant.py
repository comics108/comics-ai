#!/usr/bin/env python3
"""Materialize an audited lettering variant and publish a fail-closed aggregate decision."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image

from lettering import AuthoritativeLettering, render_lettering


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def promote(authoritative_path: Path, fixtures_path: Path, audit_path: Path,
            output_root: Path, candidate_id: str, weight: int, size: int) -> dict:
    authoritative = json.loads(authoritative_path.read_text(encoding="utf-8"))
    fixtures = json.loads(fixtures_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audited = [row for row in audit["rows"] if row["id"] == candidate_id
               and row["font_weight"] == weight and row["font_size"] == size and row["exact"]]
    if not audited:
        raise ValueError("variant has no exact audit evidence")
    raw = next(item for item in authoritative["entries"] if item["id"] == candidate_id)
    prior = next(item for item in fixtures["results"] if item["id"] == candidate_id)
    entry = AuthoritativeLettering(**raw)
    region = Image.open(prior["files"]["region_mask"])
    candidate = render_lettering(entry, region, output_root, min_font_size=size,
                                 max_font_size=size, font_weight=weight)
    accepted = candidate["decision"] == "accepted"
    previous_accepted = fixtures["accepted_count"]
    total_accepted = previous_accepted + (1 if accepted and prior["decision"] != "accepted" else 0)
    return {
        "schema_version": 1,
        "authoritative_manifest_sha256": _sha256(authoritative_path),
        "fixture_manifest_sha256": _sha256(fixtures_path),
        "variant_audit_sha256": _sha256(audit_path),
        "candidate": candidate,
        "audit_exact_evidence_count": len(audited),
        "aggregate_accepted_count": total_accepted,
        "fixture_count": fixtures["fixture_count"],
        "release_state": "accepted" if total_accepted == fixtures["fixture_count"] else "blocked",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--authoritative", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--candidate-id", required=True)
    parser.add_argument("--weight", type=int, required=True)
    parser.add_argument("--size", type=int, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = promote(args.authoritative, args.fixtures, args.audit, args.output_root,
                     args.candidate_id, args.weight, args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"candidate": report["candidate"]["decision"],
                      "accepted": report["aggregate_accepted_count"],
                      "release_state": report["release_state"]}))


if __name__ == "__main__":
    main()
