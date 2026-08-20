#!/usr/bin/env python3
"""Overlay source-explicit Gold identities onto retrieval without similarity promotion."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _asset_lineage_key(asset_version_id: str) -> str:
    head, separator, version = asset_version_id.rpartition(":v")
    if not separator or not version.isdigit():
        raise ValueError(f"invalid asset version id: {asset_version_id}")
    return head


def build(retrieval_path: Path, gold_path: Path) -> dict:
    retrieval = json.loads(retrieval_path.read_text(encoding="utf-8"))
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    explicit = {}
    for annotation in gold["annotations"]:
        identity = annotation.get("canonical_entity_id")
        if not identity:
            continue
        evidence = annotation.get("review_evidence", [])
        if not any(item.startswith("canonical_identity:explicit_psd_parent_group:") for item in evidence):
            raise ValueError("canonical identity lacks explicit source-hierarchy provenance")
        lineage_key = _asset_lineage_key(annotation["asset_version_id"])
        if lineage_key in explicit:
            raise ValueError("multiple explicit identities exist for one asset lineage")
        explicit[lineage_key] = {
            "canonical_entity_id": identity,
            "resolved_asset_version_id": annotation["asset_version_id"],
            "gold_annotation_id": annotation["id"],
            "gold_mask_sha256": annotation["mask_sha256"],
            "review_evidence": evidence,
        }
    results = []
    resolved = 0
    for item in retrieval["results"]:
        result = dict(item)
        evidence = explicit.get(_asset_lineage_key(item["query_asset_version_id"]))
        if evidence:
            result.update({
                "identity_action": "source_explicit",
                "identity_reason": "explicit_psd_parent_group_provenance",
                **evidence,
            })
            resolved += 1
        results.append(result)
    represented = {_asset_lineage_key(item["query_asset_version_id"]) for item in results}
    for lineage_key, evidence in sorted(explicit.items()):
        if lineage_key in represented:
            continue
        results.append({
            "query_asset_version_id": evidence["resolved_asset_version_id"],
            "neighbors": [],
            "identity_action": "source_explicit",
            "identity_reason": "explicit_psd_parent_group_provenance",
            **evidence,
        })
        resolved += 1
    return {
        "schema_version": 2,
        "dataset_version": gold.get("dataset_version", gold.get("version", "gold-v2.2")),
        "retrieval_sha256": _sha256(retrieval_path),
        "gold_sha256": _sha256(gold_path),
        "query_count": len(results),
        "source_explicit_count": resolved,
        "abstained_count": sum(item["identity_action"] == "abstained" for item in results),
        "identity_merges_from_similarity": 0,
        "decision": "partially_resolved" if resolved else "abstained",
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval", type=Path, required=True)
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    report = build(args.retrieval, args.gold)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({key: report[key] for key in
                      ("decision", "source_explicit_count", "abstained_count")}))


if __name__ == "__main__":
    main()
