#!/usr/bin/env python3
"""Build an immutable lettering aggregate from v1 plus verified promotion reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(base_path: Path, promotions: list[Path]) -> dict:
    base = json.loads(base_path.read_text(encoding="utf-8"))
    results = {item["id"]: item for item in base["results"]}
    lineage = [{"path": str(base_path), "sha256": _sha256(base_path), "role": "base"}]
    for path in promotions:
        report = json.loads(path.read_text(encoding="utf-8"))
        candidate = report["candidate"]
        if candidate["id"] not in results:
            raise ValueError(f"promotion candidate is outside fixture corpus: {candidate['id']}")
        if candidate["decision"] != "accepted" or not candidate["exact_readback"]:
            raise ValueError("only an accepted exact-readback candidate may replace a fixture")
        for file_path in candidate["files"].values():
            if not Path(file_path).is_file():
                raise ValueError(f"promoted lettering artifact missing: {file_path}")
        results[candidate["id"]] = candidate
        lineage.append({"path": str(path), "sha256": _sha256(path), "role": "promotion"})
    ordered = [results[item["id"]] for item in base["results"]]
    accepted = sum(item["decision"] == "accepted" for item in ordered)
    return {
        "schema_version": 2,
        "manifest_sha256": base["manifest_sha256"],
        "fixture_count": len(ordered),
        "accepted_count": accepted,
        "release_state": "accepted" if accepted == len(ordered) else "blocked",
        "lineage": lineage,
        "results": ordered,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--promotion", type=Path, action="append", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.base, args.promotion)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({"accepted": report["accepted_count"], "total": report["fixture_count"],
                      "release_state": report["release_state"]}))


if __name__ == "__main__":
    main()
