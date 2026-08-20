#!/usr/bin/env python3
"""Fail-closed vertical composition candidates for production golden chapters."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Literal


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass(frozen=True)
class AcceptedAsset:
    version_id: str
    beat_id: str
    beat_order: int
    rgba_file: str
    bitmap_mask_file: str
    width: int
    height: int
    semantic_kind: str
    checksum: str
    review_state: Literal["accepted", "proposed", "rejected"]

    def validate(self) -> None:
        if self.review_state != "accepted":
            raise ValueError(f"composition input is not accepted: {self.version_id}")
        if self.width < 1 or self.height < 1 or self.beat_order < 1:
            raise ValueError("asset dimensions and beat order must be positive")
        if len(self.checksum) != 64:
            raise ValueError("asset checksum must be SHA-256")
        if not self.rgba_file or not self.bitmap_mask_file:
            raise ValueError("editable RGBA and bitmap mask are mandatory")


@dataclass(frozen=True)
class Placement:
    asset_version_id: str
    beat_id: str
    beat_order: int
    x: int
    y: int
    width: int
    height: int
    scale: float
    z_depth: float
    rgba_file: str
    bitmap_mask_file: str


@dataclass(frozen=True)
class CompositionCandidate:
    id: str
    chapter_order: int
    method: str
    review_state: Literal["proposed"]
    scroll_type: Literal["vertical"]
    preferred_orientation: Literal["portrait"]
    viewport_width: int
    viewport_height: int
    canvas_width: int
    canvas_height: int
    placements: tuple[Placement, ...]
    camera_path: tuple[dict, ...]
    animation_proposals: tuple[dict, ...]
    lineage: dict
    quality: dict


_DEPTH = {
    "background": -0.75, "environment": -0.5, "architecture": -0.35,
    "art": 0.0, "character": 0.25, "animal": 0.25, "vehicle": 0.2,
    "prop": 0.35, "fx": 0.5, "ornament": 0.55, "balloon": 0.7,
    "caption": 0.7, "lettering": 0.8,
}


def _validate_placements(placements: list[Placement], canvas_width: int) -> dict:
    bounds = all(
        item.x >= 0 and item.y >= 0 and item.x + item.width <= canvas_width
        and item.width > 0 and item.height > 0 and math.isfinite(item.z_depth)
        and item.z_depth > -1
        for item in placements
    )
    reading_order = [item.beat_order for item in placements]
    reading_order_valid = reading_order == sorted(reading_order) and len(reading_order) == len(set(reading_order))
    overlap_pixels = 0
    for previous, current in zip(placements, placements[1:]):
        overlap_pixels += max(0, previous.y + previous.height - current.y) * max(
            0, min(previous.x + previous.width, current.x + current.width) - max(previous.x, current.x)
        )
    return {
        "bounds_valid": bounds,
        "reading_order_valid": reading_order_valid,
        "unapproved_overlap_pixels": overlap_pixels,
        "editable_masks_retained": all(item.bitmap_mask_file for item in placements),
    }


def _camera_path(placements: list[Placement], viewport_height: int) -> tuple[dict, ...]:
    if not placements:
        return ()
    centres = [max(0.0, item.y + item.height / 2 - viewport_height / 2) for item in placements]
    denominator = max(1, len(centres) - 1)
    return tuple(
        {"position": round(index * 1000 / denominator), "x": 0.0, "y": round(y, 3)}
        for index, y in enumerate(centres)
    )


def _place(
    assets: list[AcceptedAsset], *, canvas_width: int, margin: int, gap: int,
    learned_offsets: dict[str, tuple[int, int]] | None = None,
) -> list[Placement]:
    placements, cursor = [], margin
    for asset in sorted(assets, key=lambda item: item.beat_order):
        asset.validate()
        usable_width = canvas_width - 2 * margin
        scale = min(1.0, usable_width / asset.width)
        width, height = round(asset.width * scale), round(asset.height * scale)
        x = (canvas_width - width) // 2
        y = cursor
        if learned_offsets:
            dx, dy = learned_offsets.get(asset.version_id, (0, 0))
            x = max(0, min(canvas_width - width, x + round(dx)))
            y = max(cursor, y + round(dy))
        placements.append(Placement(
            asset.version_id, asset.beat_id, asset.beat_order, x, y, width, height, scale,
            _DEPTH.get(asset.semantic_kind, 0.0), asset.rgba_file, asset.bitmap_mask_file,
        ))
        cursor = y + height + gap
    return placements


def compose_candidates(
    chapter_order: int, assets: list[AcceptedAsset], *, canvas_width: int = 1080,
    viewport_height: int = 1440, learned_positioner: Callable[[list[AcceptedAsset]], dict[str, tuple[int, int]]] | None = None,
) -> tuple[CompositionCandidate, ...]:
    if not assets:
        return ()
    beat_orders = [asset.beat_order for asset in assets]
    if len(beat_orders) != len(set(beat_orders)):
        raise ValueError("exactly one accepted composition asset is required per beat")
    methods: list[tuple[str, dict[str, tuple[int, int]] | None, str | None]] = [
        ("deterministic_vertical_stack_v1", None, None)
    ]
    if learned_positioner is not None:
        methods.append(("learned_positioner_candidate_v1", learned_positioner(assets), "external_adapter"))
    candidates = []
    for method, offsets, checkpoint in methods:
        placements = _place(assets, canvas_width=canvas_width, margin=64, gap=96, learned_offsets=offsets)
        quality = _validate_placements(placements, canvas_width)
        if not all((quality["bounds_valid"], quality["reading_order_valid"], quality["editable_masks_retained"])):
            raise ValueError(f"invalid composition proposal from {method}")
        canvas_height = placements[-1].y + placements[-1].height + 64
        lineage = {
            "input_checksums": [asset.checksum for asset in sorted(assets, key=lambda item: item.beat_order)],
            "method": method, "model_checkpoint": checkpoint,
            "intent_claim": "candidate_only_not_artist_intent",
        }
        identifier = f"ch{chapter_order:02d}:{method}:{canonical_sha256(lineage)[:12]}"
        animations = tuple({
            "asset_version_id": item.asset_version_id, "kind": "alpha_reveal",
            "start": (index * 1000) // len(placements), "end": ((index + 1) * 1000) // len(placements),
            "state": "proposed_not_packaged",
        } for index, item in enumerate(placements))
        candidates.append(CompositionCandidate(
            identifier, chapter_order, method, "proposed", "vertical", "portrait",
            canvas_width, viewport_height, canvas_width, canvas_height, tuple(placements),
            _camera_path(placements, viewport_height), animations, lineage, quality,
        ))
    return tuple(candidates)


def build_fail_closed_summary(coverage: dict) -> dict:
    chapters = []
    for chapter in coverage["chapters"]:
        unresolved = [item["beat_id"] for item in chapter["coverage"] if not item["asset_version_ids"]]
        chapters.append({
            "chapter_order": chapter["chapter_order"], "candidate_count": 0,
            "accepted_asset_count": chapter["accepted_coverage_count"],
            "missing_beat_ids": unresolved, "state": "blocked" if unresolved else "ready",
        })
    return {
        "schema_version": 1, "scroll_type": "vertical", "preferred_orientation": "portrait",
        "candidate_state": "proposed_only", "chapters": chapters,
        "release_state": "blocked" if any(item["state"] == "blocked" for item in chapters) else "proposed",
        "blockers": ["accepted_assets_missing", "composition_candidates_not_reviewed"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build_fail_closed_summary(json.loads(args.coverage.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"release_state": summary["release_state"], "chapters": len(summary["chapters"])}))


if __name__ == "__main__":
    main()
