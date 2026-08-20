#!/usr/bin/env python3
"""Task 8.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): `manifest.json`
(schema-versioned, per-chapter provenance/validation/hash records) and `report.md`
(human-readable coverage summary), per 02-specifications.md's Manifest Contract.

Scope note: "a stale file from an earlier run never counts when its dataset/config fingerprints
differ" (Specifications' Manifest Contract) requires comparing a *previous* manifest.json's stored
fingerprint against a freshly computed one before deciding whether to reuse a chapter -- that's a
resumability decision belonging to Task 8.2's `pipeline.py`, not this module. This module only
ever builds one fresh manifest for the chapters it's given; it doesn't read or compare against a
prior run's manifest.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from build_storyboard import ChapterStoryboard
from models import CanonicalChapter
from validate_output import ValidationResult

SCHEMA_VERSION = 1
CONFIG_VERSION = 1  # bump when renderer/layout/packager behavior changes in a way that should
                     # invalidate cached outputs from an earlier run

APP_DIR = Path(__file__).resolve().parents[1]
FONTS_DIR = APP_DIR / "fonts" / "Noto"

# The exact canvas constants this fingerprint must track if they ever change -- duplicated here
# (not imported from layout_chapter/render_cards) so this module has no import-order dependency on
# them; a mismatch would only matter if someone changes one file without the other, which the
# real-value cross-check test below guards against.
_CANVAS_CONSTANTS_SNAPSHOT = (1080, 72, 32, 72)  # (CANVAS_WIDTH, CONTENT_MARGIN, LAYER_GAP, SAFE_AREA)


def compute_file_sha256(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def compute_dataset_fingerprint(dataset_dir: Path) -> str:
    """Hashes the real CSV files this generator reads, by relative name and content -- changes if
    any of them is edited, added, or removed."""
    hasher = hashlib.sha256()
    for name in sorted(("db_books.csv", "db_chapters.csv", "Gita_Slokas.csv")):
        path = dataset_dir / name
        hasher.update(name.encode("utf-8"))
        hasher.update(path.read_bytes())
    return f"sha256:{hasher.hexdigest()}"


def compute_config_fingerprint() -> str:
    """Hashes the real vendored font files plus the fixed canvas constants and CONFIG_VERSION --
    changes if fonts are swapped or rendering/layout constants change."""
    hasher = hashlib.sha256()
    hasher.update(str(CONFIG_VERSION).encode("utf-8"))
    for font_path in sorted(FONTS_DIR.glob("*.ttf")):
        hasher.update(font_path.name.encode("utf-8"))
        hasher.update(font_path.read_bytes())
    hasher.update(str(_CANVAS_CONSTANTS_SNAPSHOT).encode("utf-8"))
    return f"sha256:{hasher.hexdigest()}"


def _storyboard_prompt_hash(storyboard: ChapterStoryboard) -> str | None:
    if storyboard.mode == "deterministic":
        return None
    return f"sha256:{hashlib.sha256(storyboard.prompt_version.encode('utf-8')).hexdigest()}"


@dataclass(frozen=True)
class ChapterManifestEntry:
    order: int
    chapter_id: int
    title: str
    source_sloka_count: int
    source_id_min: int
    source_id_max: int
    output_path: str
    byte_size: int
    sha256: str
    layer_count: int
    width: int
    height: int
    storyboard_mode: str
    storyboard_model: str | None
    storyboard_prompt_hash: str | None
    psd_inputs: tuple[str, ...]
    audio_refs_omitted: int
    structural_valid: bool
    validation_issues: tuple[str, ...]
    warnings: tuple[str, ...]
    status: Literal["valid", "failed"]

    @staticmethod
    def from_dict(raw: dict) -> "ChapterManifestEntry":
        """Reconstructs an entry from a previously-written manifest.json's chapter dict -- used by
        pipeline.py (Task 8.2) to carry forward chapters not touched by the current run."""
        source_id_min, source_id_max = raw["source_id_range"]
        return ChapterManifestEntry(
            order=raw["order"],
            chapter_id=raw["chapter_id"],
            title=raw["title"],
            source_sloka_count=raw["source_sloka_count"],
            source_id_min=source_id_min,
            source_id_max=source_id_max,
            output_path=raw["output_path"],
            byte_size=raw["byte_size"],
            sha256=raw["sha256"],
            layer_count=raw["layer_count"],
            width=raw["width"],
            height=raw["height"],
            storyboard_mode=raw["storyboard_mode"],
            storyboard_model=raw["storyboard_model"],
            storyboard_prompt_hash=raw["storyboard_prompt_hash"],
            psd_inputs=tuple(raw["psd_inputs"]),
            audio_refs_omitted=raw["audio_refs_omitted"],
            structural_valid=raw["structural_valid"],
            validation_issues=tuple(raw["validation_issues"]),
            warnings=tuple(raw["warnings"]),
            status=raw["status"],
        )

    def to_dict(self) -> dict:
        return {
            "order": self.order,
            "chapter_id": self.chapter_id,
            "title": self.title,
            "source_sloka_count": self.source_sloka_count,
            "source_id_range": [self.source_id_min, self.source_id_max],
            "output_path": self.output_path,
            "byte_size": self.byte_size,
            "sha256": self.sha256,
            "layer_count": self.layer_count,
            "width": self.width,
            "height": self.height,
            "storyboard_mode": self.storyboard_mode,
            "storyboard_model": self.storyboard_model,
            "storyboard_prompt_hash": self.storyboard_prompt_hash,
            "psd_inputs": list(self.psd_inputs),
            "audio_refs_omitted": self.audio_refs_omitted,
            "structural_valid": self.structural_valid,
            "validation_issues": list(self.validation_issues),
            "warnings": list(self.warnings),
            "status": self.status,
        }


def build_chapter_entry(
    chapter: CanonicalChapter,
    output_path: Path,
    layer_count: int,
    width: int,
    height: int,
    storyboard: ChapterStoryboard,
    validation_result: ValidationResult,
    psd_inputs: tuple[str, ...] = (),
    audio_refs_omitted: int = 0,
) -> ChapterManifestEntry:
    orders = [sloka.id for sloka in chapter.slokas]
    status: Literal["valid", "failed"] = "valid" if validation_result.ok else "failed"
    return ChapterManifestEntry(
        order=chapter.order,
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        source_sloka_count=len(chapter.slokas),
        source_id_min=min(orders) if orders else 0,
        source_id_max=max(orders) if orders else 0,
        output_path=str(output_path),
        byte_size=output_path.stat().st_size,
        sha256=compute_file_sha256(output_path),
        layer_count=layer_count,
        width=width,
        height=height,
        storyboard_mode=storyboard.mode,
        storyboard_model=storyboard.model,
        storyboard_prompt_hash=_storyboard_prompt_hash(storyboard),
        psd_inputs=psd_inputs,
        audio_refs_omitted=audio_refs_omitted,
        structural_valid=validation_result.ok,
        validation_issues=tuple(f"{i.check}: {i.message}" for i in validation_result.issues),
        warnings=storyboard.warnings,
        status=status,
    )


def build_failed_chapter_entry(chapter: CanonicalChapter, error_message: str) -> ChapterManifestEntry:
    """For a chapter that failed before producing any output file (e.g. a rendering exception) --
    `build_chapter_entry` can't be used since it requires a real file to hash/stat. Used by
    pipeline.py's (Task 8.2) continue-after-failure batch semantics."""
    return ChapterManifestEntry(
        order=chapter.order,
        chapter_id=chapter.chapter_id,
        title=chapter.title,
        source_sloka_count=len(chapter.slokas),
        source_id_min=min((s.id for s in chapter.slokas), default=0),
        source_id_max=max((s.id for s in chapter.slokas), default=0),
        output_path="",
        byte_size=0,
        sha256="",
        layer_count=0,
        width=0,
        height=0,
        storyboard_mode="n/a",
        storyboard_model=None,
        storyboard_prompt_hash=None,
        psd_inputs=(),
        audio_refs_omitted=0,
        structural_valid=False,
        validation_issues=(f"pipeline_error: {error_message}",),
        warnings=(),
        status="failed",
    )


def build_manifest(
    dataset_dir: Path,
    book_id: int,
    language: str,
    expected_chapters: int,
    expected_slokas: int,
    chapter_entries: list[ChapterManifestEntry],
) -> dict:
    return {
        "schema_version": SCHEMA_VERSION,
        "dataset_fingerprint": compute_dataset_fingerprint(dataset_dir),
        "config_fingerprint": compute_config_fingerprint(),
        "book_id": book_id,
        "language": language,
        "expected_chapters": expected_chapters,
        "expected_slokas": expected_slokas,
        "chapters": [entry.to_dict() for entry in chapter_entries],
    }


def coverage_count(manifest: dict) -> int:
    return sum(1 for chapter in manifest["chapters"] if chapter["status"] == "valid")


def write_manifest(path: Path, manifest: dict) -> None:
    path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def render_report_md(manifest: dict) -> str:
    valid = coverage_count(manifest)
    expected = manifest["expected_chapters"]
    lines = [
        "# Bhagavad Gita Generator -- Coverage Report",
        "",
        f"Coverage: **{valid}/{expected}** chapters valid.",
        "",
        "| Order | Title | Slokas | Status | Issues |",
        "|-------|-------|--------|--------|--------|",
    ]
    for chapter in sorted(manifest["chapters"], key=lambda c: c["order"]):
        issues = "; ".join(chapter["validation_issues"]) or "-"
        lines.append(
            f"| {chapter['order']} | {chapter['title']} | {chapter['source_sloka_count']} | "
            f"{chapter['status']} | {issues} |"
        )
    return "\n".join(lines) + "\n"


def render_lottie_report_md(manifest: dict) -> str:
    """Dedicated report for the standalone Lottie-derived document (not one of 18 chapters)."""
    references = "<br>".join(manifest["camera_reference_layers"])
    return "\n".join([
        "# Bhagavad Gita Lottie Import Report",
        "",
        f"Output: `{manifest['output_file']}`",
        f"Scenes: **{manifest['scene_count']}**; image layers: **{manifest['image_layer_count']}**; "
        f"animated layers: **{manifest['animated_layer_count']}**.",
        f"Camera points: **{manifest['camera_point_count']}**; distinct non-zero z-depth values: "
        f"**{manifest['distinct_nonzero_z_depth_count']}**.",
        "",
        f"Camera references: {references}",
        "",
        "## Parallax limitation",
        "",
        "The archive contains canonical `cameraPath` and per-layer `zDepth` data. These fields are "
        "additive and safe for legacy readers to ignore, but current `.comics` viewers do not yet "
        "render the resulting parallax effect. This output preserves the data needed by the approved "
        "future shared-library/viewer implementation; it does not claim visible parallax today.",
        "",
        f"SHA-256: `{manifest['sha256']}`",
        "",
    ])
