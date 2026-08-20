"""Independent review ledger with graph-propagated approval invalidation."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from production_models import Asset, ReviewDecision, ReviewDimension
from asset_graph import AssetGraph


_SEPARABLE_FOREGROUND_KINDS = {
    "character", "animal", "prop", "vehicle", "architecture", "fx", "ornament",
}


class ReviewLedger:
    def __init__(self) -> None:
        self.decisions: list[ReviewDecision] = []

    def append(self, decision: ReviewDecision) -> None:
        if any(existing.id == decision.id for existing in self.decisions):
            raise ValueError(f"review decision already exists: {decision.id}")
        self.decisions.append(decision)

    def approve_asset_version(
        self,
        asset: Asset,
        version: int,
        *,
        decision_id: str,
        dimension: ReviewDimension,
        reviewer: str,
        timestamp: str,
        rationale: str,
    ) -> ReviewDecision:
        candidate = next((item for item in asset.versions if item.version == version), None)
        if candidate is None:
            raise KeyError(f"unknown asset version: {asset.id}:v{version}")
        if asset.semantic_kind in _SEPARABLE_FOREGROUND_KINDS and (
            not candidate.rgba_file or not candidate.bitmap_mask_file
        ):
            raise ValueError("separable foreground approval requires RGBA and a bitmap mask")
        decision = ReviewDecision(
            id=decision_id,
            subject_id=f"{asset.id}:v{version}",
            subject_version=version,
            dimension=dimension,
            state="approved",
            reviewer=reviewer,
            timestamp=timestamp,
            rationale=rationale,
        )
        self.append(decision)
        return decision

    def invalidate_from(
        self,
        upstream_id: str,
        graph: AssetGraph,
        *,
        invalidated_by: str,
        timestamp: str,
    ) -> tuple[ReviewDecision, ...]:
        affected = graph.dependents_of(upstream_id) | {upstream_id}
        already_invalidated = {
            original_id
            for decision in self.decisions
            if decision.state == "invalidated"
            for original_id in decision.evidence
        }
        created: list[ReviewDecision] = []
        for decision in tuple(self.decisions):
            if (
                decision.state != "approved"
                or decision.subject_id not in affected
                or decision.id in already_invalidated
            ):
                continue
            invalidation = ReviewDecision(
                id=f"invalidate:{decision.id}:{invalidated_by}",
                subject_id=decision.subject_id,
                subject_version=decision.subject_version,
                dimension=decision.dimension,
                state="invalidated",
                reviewer="system",
                timestamp=timestamp,
                rationale=f"upstream dependency changed: {upstream_id}",
                evidence=(decision.id,),
                invalidated_by=invalidated_by,
            )
            self.append(invalidation)
            created.append(invalidation)
        return tuple(created)

    def write_snapshot(self, path: Path) -> None:
        """Publish an immutable review-history snapshot without rewriting earlier releases."""
        payload = {
            "schema_version": 1,
            "decisions": [asdict(decision) for decision in self.decisions],
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")

    @classmethod
    def read_snapshot(cls, path: Path) -> "ReviewLedger":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported review snapshot schema")
        ledger = cls()
        ledger.decisions = [
            ReviewDecision(**{**item, "evidence": tuple(item.get("evidence") or ())})
            for item in payload["decisions"]
        ]
        return ledger
