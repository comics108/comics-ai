"""Rank visually similar assets without converting similarity into canonical identity."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def cosine_similarity(left: list[float], right: list[float]) -> float:
    left_vector, right_vector = np.asarray(left), np.asarray(right)
    if left_vector.shape != right_vector.shape or left_vector.ndim != 1:
        raise ValueError("retrieval descriptors must have one identical shape")
    denominator = float(np.linalg.norm(left_vector) * np.linalg.norm(right_vector))
    return float(np.dot(left_vector, right_vector) / max(denominator, 1e-12))


def build_retrieval_index(catalog: dict, *, limit: int = 5) -> dict:
    proposals = catalog["proposals"]
    results = []
    for query in proposals:
        ranked = []
        for candidate in proposals:
            if candidate["asset_version_id"] == query["asset_version_id"]:
                continue
            similarity = cosine_similarity(query["descriptor"], candidate["descriptor"])
            ranked.append({
                "asset_version_id": candidate["asset_version_id"],
                "similarity": similarity,
                "same_semantic_proposal": (
                    candidate["semantic_kind_proposal"] == query["semantic_kind_proposal"]
                ),
                "canonical_identity_match": None,
                "decision": "similarity_only",
            })
        ranked.sort(key=lambda item: (-item["similarity"], item["asset_version_id"]))
        results.append({
            "query_asset_version_id": query["asset_version_id"],
            "neighbors": ranked[:limit],
            "identity_action": "abstained",
            "identity_reason": "similarity_cannot_confirm_canonical_identity",
        })
    return {
        "schema_version": 1,
        "dataset_version": catalog["dataset_version"],
        "catalog_sha256": None,
        "retriever": "cosine-visual-descriptor-v1",
        "query_count": len(results),
        "top_k": limit,
        "identity_merges": 0,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("catalog", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()
    if args.limit < 1:
        raise ValueError("retrieval limit must be positive")
    catalog = json.loads(args.catalog.read_text(encoding="utf-8"))
    index = build_retrieval_index(catalog, limit=args.limit)
    index["catalog_sha256"] = hashlib.sha256(args.catalog.read_bytes()).hexdigest()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(index, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
    print(json.dumps({
        "index": str(args.out), "queries": index["query_count"],
        "identity_merges": index["identity_merges"],
    }))


if __name__ == "__main__":
    main()
