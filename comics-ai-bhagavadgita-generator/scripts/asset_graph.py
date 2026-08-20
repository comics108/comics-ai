"""Append-only production asset/entity graph and immutable graph snapshots."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from production_models import Asset


@dataclass(frozen=True)
class AssetEntityLink:
    id: str
    asset_id: str
    asset_version: int
    entity_id: str
    role: str
    confidence: float
    method: str
    review_state: Literal["proposed", "accepted", "rejected", "superseded"]
    supersedes_id: str | None

    def __post_init__(self) -> None:
        if not all((self.id, self.asset_id, self.entity_id, self.role, self.method)):
            raise ValueError("identity link fields are required")
        if self.asset_version < 1 or not 0 <= self.confidence <= 1:
            raise ValueError("identity link version/confidence is invalid")


@dataclass(frozen=True)
class EntityRevision:
    id: str
    entity_id: str
    revision: int
    operation: Literal["create", "merge", "split"]
    source_entity_ids: tuple[str, ...]
    reviewer: str
    timestamp: str
    rationale: str

    def __post_init__(self) -> None:
        if not all((self.id, self.entity_id, self.reviewer, self.timestamp, self.rationale)):
            raise ValueError("entity revision provenance is required")
        if self.revision < 1 or not self.source_entity_ids:
            raise ValueError("entity revision and source history are required")


class AssetGraph:
    def __init__(self) -> None:
        self.assets: dict[str, Asset] = {}
        self.identity_links: list[AssetEntityLink] = []
        self.entity_revisions: list[EntityRevision] = []
        self._dependencies: dict[str, set[str]] = {}

    def add_asset(self, asset: Asset) -> None:
        if asset.id in self.assets:
            raise ValueError(f"asset already exists: {asset.id}")
        self.assets[asset.id] = asset

    def add_identity_proposal(self, link: AssetEntityLink) -> None:
        if any(existing.id == link.id for existing in self.identity_links):
            raise ValueError(f"identity link already exists: {link.id}")
        asset = self.assets.get(link.asset_id)
        if asset is None:
            raise KeyError(f"unknown asset: {link.asset_id}")
        if not any(version.version == link.asset_version for version in asset.versions):
            raise KeyError(f"unknown asset version: {link.asset_id}:v{link.asset_version}")
        # Deliberately never update Asset.canonical_entity_ids here. A model result is a
        # reviewable proposal, not canonical identity authority.
        self.identity_links.append(link)

    def record_entity_revision(self, revision: EntityRevision) -> None:
        if any(existing.id == revision.id for existing in self.entity_revisions):
            raise ValueError(f"entity revision already exists: {revision.id}")
        previous = [item.revision for item in self.entity_revisions if item.entity_id == revision.entity_id]
        if previous and revision.revision <= max(previous):
            raise ValueError("entity revisions must append with increasing revision numbers")
        self.entity_revisions.append(revision)

    def add_dependency(self, upstream_id: str, downstream_id: str) -> None:
        if not upstream_id or not downstream_id or upstream_id == downstream_id:
            raise ValueError("dependency endpoints must be distinct and non-empty")
        if upstream_id in self.dependents_of(downstream_id):
            raise ValueError("asset graph dependencies cannot contain a cycle")
        self._dependencies.setdefault(upstream_id, set()).add(downstream_id)

    def dependents_of(self, subject_id: str) -> frozenset[str]:
        found: set[str] = set()
        pending = list(self._dependencies.get(subject_id, ()))
        while pending:
            dependent = pending.pop()
            if dependent in found:
                continue
            found.add(dependent)
            pending.extend(self._dependencies.get(dependent, ()))
        return frozenset(found)

    def write_snapshot(self, path: Path) -> None:
        """Publish once; an existing historical snapshot is never overwritten."""
        payload = {
            "schema_version": 1,
            "identity_links": [asdict(item) for item in self.identity_links],
            "entity_revisions": [asdict(item) for item in self.entity_revisions],
            "dependencies": {
                key: sorted(values) for key, values in sorted(self._dependencies.items())
            },
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
            stream.write("\n")

    @classmethod
    def read_snapshot(cls, path: Path) -> "AssetGraph":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("unsupported asset graph snapshot schema")
        graph = cls()
        graph.identity_links = [AssetEntityLink(**item) for item in payload["identity_links"]]
        graph.entity_revisions = [
            EntityRevision(**{**item, "source_entity_ids": tuple(item["source_entity_ids"])})
            for item in payload["entity_revisions"]
        ]
        graph._dependencies = {
            key: set(values) for key, values in payload["dependencies"].items()
        }
        return graph
