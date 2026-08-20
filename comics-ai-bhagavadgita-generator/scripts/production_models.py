#!/usr/bin/env python3
"""Canonical production asset/provenance models from Specifications v0.9."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal, TypeAlias


JsonValue: TypeAlias = None | bool | int | float | str | list["JsonValue"] | dict[str, "JsonValue"]
_SHA256 = re.compile(r"^[0-9a-f]{64}$")

SourceKind: TypeAlias = Literal[
    "structured_text", "manuscript", "psd", "pdf", "raster", "lottie",
    "audio", "comics", "font", "lettering_sample", "palette", "editorial_note",
]
SemanticWork: TypeAlias = Literal["bhagavad_gita", "gita_dhyanam", "unclassified"]
SemanticScopeKind: TypeAlias = Literal[
    "canonical_chapter", "canonical_verse_range", "standalone_prologue",
    "source_component", "unclassified",
]
MappingState: TypeAlias = Literal["confirmed", "inferred", "unmapped", "not_applicable"]
AssetKind: TypeAlias = Literal[
    "background", "environment", "character", "animal", "prop", "vehicle",
    "architecture", "fx", "ornament", "balloon", "caption", "lettering", "art",
]
ArtStage: TypeAlias = Literal["thumbnail", "sketch", "ink", "flat", "shaded", "final"]
ReviewState: TypeAlias = Literal["proposed", "accepted", "rejected", "superseded"]
ReviewDimension: TypeAlias = Literal[
    "technical", "identity_style", "art_direction", "lettering",
    "cultural_editorial", "runtime",
]


@dataclass(frozen=True)
class SourceRecord:
    id: str
    kind: SourceKind
    relative_path: str
    sha256: str
    byte_size: int
    media_type: str
    metadata: dict[str, JsonValue]
    semantic_scope_id: str
    parent_source_id: str | None = None

    def __post_init__(self) -> None:
        if not self.id or not self.semantic_scope_id:
            raise ValueError("source id and semantic_scope_id are required")
        if self.relative_path.startswith("/") or ".." in self.relative_path.split("/"):
            raise ValueError(f"source path must be safe and relative: {self.relative_path!r}")
        if not _SHA256.fullmatch(self.sha256):
            raise ValueError("source sha256 must be 64 lowercase hexadecimal characters")
        if self.byte_size < 0:
            raise ValueError("source byte_size cannot be negative")


@dataclass(frozen=True)
class SourceSemanticScope:
    id: str
    work: SemanticWork
    scope: SemanticScopeKind
    chapter_orders: tuple[int, ...]
    verse_ranges: tuple[tuple[int, int, int], ...]
    mapping_state: MappingState
    evidence: tuple[str, ...]
    reviewer: str | None

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("semantic scope id is required")
        if any(chapter < 1 for chapter in self.chapter_orders):
            raise ValueError("chapter orders must be positive")
        for chapter, first, last in self.verse_ranges:
            if chapter < 1 or first < 1 or last < first:
                raise ValueError(f"invalid verse range: {(chapter, first, last)!r}")
        if self.mapping_state == "confirmed" and (not self.evidence or not self.reviewer):
            raise ValueError("confirmed semantic scope requires evidence and reviewer")


@dataclass(frozen=True)
class Lineage:
    input_checksums: tuple[str, ...]
    action_id: str | None
    code_revision: str
    model_checkpoint: str | None
    configuration_hash: str
    environment: dict[str, JsonValue]
    timestamp: str
    cost_usage: dict[str, JsonValue]
    reviewer_decision_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.input_checksums or any(not _SHA256.fullmatch(value) for value in self.input_checksums):
            raise ValueError("lineage requires valid input sha256 checksums")
        if not _SHA256.fullmatch(self.configuration_hash):
            raise ValueError("lineage configuration_hash must be a sha256")
        if not self.code_revision or not self.timestamp:
            raise ValueError("lineage code_revision and timestamp are required")


@dataclass(frozen=True)
class AssetVersion:
    version: int
    source_id: str
    source_region: tuple[int, int, int, int] | None
    rgba_file: str | None
    bitmap_mask_file: str | None
    contour_file: str | None
    width: int
    height: int
    art_stage: ArtStage
    style_tags: tuple[str, ...]
    palette: tuple[str, ...]
    pose: str | None
    expression: str | None
    costume: str | None
    view: str | None
    allowed_transformations: tuple[str, ...]
    lineage: Lineage
    metrics: dict[str, float]
    review_state: ReviewState

    def __post_init__(self) -> None:
        if self.version < 1 or self.width < 1 or self.height < 1:
            raise ValueError("asset version and dimensions must be positive")
        if not self.source_id:
            raise ValueError("asset version source_id is required")
        if self.source_region is not None:
            x, y, width, height = self.source_region
            if x < 0 or y < 0 or width < 1 or height < 1:
                raise ValueError(f"invalid source_region: {self.source_region!r}")


@dataclass
class Asset:
    id: str
    canonical_entity_ids: list[str]
    semantic_kind: AssetKind
    versions: list[AssetVersion]

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("asset id is required")
        numbers = [version.version for version in self.versions]
        if len(numbers) != len(set(numbers)):
            raise ValueError("asset versions must have unique version numbers")


@dataclass(frozen=True)
class ReviewDecision:
    id: str
    subject_id: str
    subject_version: int | str
    dimension: ReviewDimension
    state: Literal["approved", "rejected", "changes_requested", "invalidated"]
    reviewer: str
    timestamp: str
    rationale: str
    evidence: tuple[str, ...] = ()
    invalidated_by: str | None = None

    def __post_init__(self) -> None:
        if not all((self.id, self.subject_id, self.reviewer, self.timestamp, self.rationale)):
            raise ValueError("review decision identity, reviewer, timestamp, and rationale are required")
