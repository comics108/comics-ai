"""Gold v1 annotation manifest contract and release-gate validation."""

from __future__ import annotations

import json
import hashlib
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class GoldAnnotation:
    id: str
    asset_version_id: str
    source_composition_id: str
    source_kind: Literal["psd", "panorama"]
    split: Literal["train", "validation", "test"]
    semantic_kind: str
    canonical_entity_id: str | None
    principal_character: bool
    bitmap_mask_file: str
    mask_sha256: str
    source_region: tuple[int, int, int, int]
    review_resolution: tuple[int, int]
    source_to_review_scale: tuple[float, float]
    label_origin: Literal[
        "human_corrected", "psd_alpha_reviewed", "automated_consensus", "bbox_bootstrap"
    ]
    reviewer: str
    accepted_at: str
    accepted: bool
    review_mode: Literal["human", "automated"] = "automated"
    review_evidence: tuple[str, ...] = ()


@dataclass(frozen=True)
class GoldDataset:
    version: str
    annotations: tuple[GoldAnnotation, ...]


@dataclass(frozen=True)
class GoldValidationResult:
    accepted_count: int
    held_out_count: int
    source_composition_count: int


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_gold_dataset(dataset: GoldDataset) -> GoldValidationResult:
    if not dataset.version:
        raise ValueError("gold dataset version is required")
    accepted = [annotation for annotation in dataset.annotations if annotation.accepted]
    if len({annotation.id for annotation in accepted}) != len(accepted):
        raise ValueError("gold annotation IDs must be unique")
    if len(accepted) < 120:
        raise ValueError("Gold v1 requires at least 120 accepted foreground instances")

    for annotation in accepted:
        if annotation.label_origin == "bbox_bootstrap":
            raise ValueError("bbox bootstrap labels cannot enter Gold v1")
        if not annotation.bitmap_mask_file or not _SHA256.fullmatch(annotation.mask_sha256):
            raise ValueError("every gold annotation requires a checksummed bitmap mask")
        if not annotation.reviewer or not annotation.accepted_at:
            raise ValueError("every accepted gold annotation requires reviewer provenance")
        if annotation.review_mode == "automated":
            if not annotation.reviewer.startswith("auto:"):
                raise ValueError("automated gold reviewer must use a versioned auto: identity")
            if "automated-mask-review-v1" not in annotation.review_evidence:
                raise ValueError("automated gold requires validated mask-review evidence")
            if annotation.label_origin not in {"psd_alpha_reviewed", "automated_consensus"}:
                raise ValueError("automated gold label origin is not independently verifiable")
        if annotation.principal_character and not annotation.canonical_entity_id:
            raise ValueError("principal-character gold requires canonical identity")
        x, y, width, height = annotation.source_region
        review_width, review_height = annotation.review_resolution
        scale_x, scale_y = annotation.source_to_review_scale
        # Native PSD layers may legitimately extend above/left of the document canvas. Preserve
        # signed origins; reversibility depends on positive extents and scale, not clipped origins.
        if min(width, height, review_width, review_height) < 1:
            raise ValueError("gold source/review extents must be positive and reversible")
        if scale_x <= 0 or scale_y <= 0:
            raise ValueError("gold source-to-review scale must be positive")

    compositions = {annotation.source_composition_id for annotation in accepted}
    psd_compositions = {
        annotation.source_composition_id for annotation in accepted if annotation.source_kind == "psd"
    }
    panorama_compositions = {
        annotation.source_composition_id
        for annotation in accepted
        if annotation.source_kind == "panorama"
    }
    if len(compositions) < 4 or len(psd_compositions) < 2 or len(panorama_compositions) < 2:
        raise ValueError("Gold v1 requires 2 PSD and 2 panorama source-disjoint compositions")

    train_sources = {
        annotation.source_composition_id for annotation in accepted if annotation.split == "train"
    }
    held_out_sources = {
        annotation.source_composition_id
        for annotation in accepted
        if annotation.split in {"validation", "test"}
    }
    overlap = train_sources & held_out_sources
    if overlap:
        raise ValueError(f"source composition split leakage: {sorted(overlap)}")
    held_out_count = sum(annotation.split == "test" for annotation in accepted)
    if held_out_count < 30:
        raise ValueError("Gold v1 requires at least 30 held-out test instances")
    return GoldValidationResult(len(accepted), held_out_count, len(compositions))


def verify_gold_artifacts(dataset: GoldDataset, repository_root: Path) -> GoldValidationResult:
    """Recheck every accepted mask from the published path and immutable checksum."""
    result = validate_gold_dataset(dataset)
    for annotation in dataset.annotations:
        if not annotation.accepted:
            continue
        path = Path(annotation.bitmap_mask_file)
        if not path.is_absolute():
            path = repository_root / path
        if not path.is_file():
            raise ValueError(f"gold mask is missing: {annotation.bitmap_mask_file}")
        if _sha256_file(path) != annotation.mask_sha256:
            raise ValueError(f"gold mask checksum mismatch: {annotation.id}")
    return result


def write_gold_manifest(dataset: GoldDataset, path: Path) -> None:
    """Validate, then publish a version once; never overwrite reviewed gold history."""
    validate_gold_dataset(dataset)
    payload = {
        "schema_version": 1,
        "dataset_version": dataset.version,
        "annotations": [asdict(annotation) for annotation in dataset.annotations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, ensure_ascii=False, sort_keys=True, indent=2)
        stream.write("\n")
