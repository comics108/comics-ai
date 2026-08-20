#!/usr/bin/env python3
"""Task 8.2 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): the real
end-to-end pipeline CLI wiring every prior phase together, per 02-specifications.md's Pipeline
CLI and Idempotency/Resumability sections.

`--no-ai` currently has no *choice* to make: Task 2.2 (Ollama-backed storyboard mode) hasn't been
built yet, so every chapter always uses Task 2.1's deterministic storyboard regardless of this
flag. Passing `--no-ai` is still meaningful (it's the documented, explicit way to request the
deterministic path once Task 2.2 exists) and is accepted rather than rejected; *not* passing it
records a warning in that chapter's manifest entry rather than silently pretending AI mode ran.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

import render_cards
from build_storyboard import build_deterministic_storyboard
from import_psd import import_psd_panel
from layout_chapter import CANVAS_WIDTH, layout_chapter, layout_chapter_content
from load_dataset import DATASET_DIR, DEFAULT_BOOK_ID, EXPECTED_CHAPTER_COUNT, EXPECTED_SLOKA_COUNT, load_book_one, verify_dataset_integrity
from models import CanonicalChapter
from import_lottie import import_lottie_file
from package_comics import PackagingAsset, write_comics_archive
from report import (
    ChapterManifestEntry,
    build_chapter_entry,
    build_failed_chapter_entry,
    build_manifest,
    compute_config_fingerprint,
    compute_dataset_fingerprint,
    compute_file_sha256,
    coverage_count,
    render_report_md,
    render_lottie_report_md,
    write_manifest,
)
from validate_output import validate_archive_structure

DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parents[4] / "work" / "bhagavadgita"
PSD_DIR = DATASET_DIR.parent / "vaishnav" / "drawing"
PSD_FILENAMES = ("5_1.psd", "5_2.psd", "app_BG._chiba5.psd")
PSD_CHAPTER_ORDER = 5
LOTTIE_SOURCE = (
    DATASET_DIR.parent
    / "vaishnav/bhagavadgita/lottie_unzip/Mediation of the Bhagavat Gita"
    / "Mediation of the Bhagavat Gita_content"
    / "Mediation of the Bhagavat Gita.json"
)
LOTTIE_OUTPUT_NAME = "mediation_of_the_bhagavat_gita.comics"


class ChapterLockedError(RuntimeError):
    pass


@dataclass(frozen=True)
class PipelineConfig:
    use_ai: bool
    use_psd: bool
    force: bool


def _output_path(output_dir: Path, chapter: CanonicalChapter) -> Path:
    return output_dir / f"chapter_{chapter.order:02d}.comics"


def _lock_path(output_dir: Path, chapter: CanonicalChapter) -> Path:
    return output_dir / f".chapter_{chapter.order:02d}.lock"


@contextlib.contextmanager
def chapter_lock(output_dir: Path, chapter: CanonicalChapter):
    """A simple, real cross-process lock via exclusive file creation (`O_CREAT|O_EXCL`), per
    Specifications' "Chapter-level locks prevent two processes from publishing the same chapter
    concurrently." A leftover lock from a crashed process must be removed manually -- this is a
    real limitation of a plain lock file, not silently auto-broken, since auto-breaking a lock is
    exactly what would defeat its purpose if the other process were merely slow, not dead."""
    lock_path = _lock_path(output_dir, chapter)
    try:
        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.close(fd)
    except FileExistsError:
        raise ChapterLockedError(
            f"chapter {chapter.order} is locked by another process (stale lock? remove "
            f"{lock_path} manually if you're sure no other run is in progress)"
        )
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def load_previous_manifest(manifest_path: Path) -> dict | None:
    if not manifest_path.exists():
        return None
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None


def can_reuse_chapter(
    previous_manifest: dict | None,
    chapter: CanonicalChapter,
    output_path: Path,
    dataset_fingerprint: str,
    config_fingerprint: str,
) -> bool:
    if previous_manifest is None:
        return False
    if (
        previous_manifest.get("dataset_fingerprint") != dataset_fingerprint
        or previous_manifest.get("config_fingerprint") != config_fingerprint
    ):
        return False
    entry = next(
        (c for c in previous_manifest["chapters"] if c["order"] == chapter.order), None
    )
    if entry is None or entry["status"] != "valid":
        return False
    if not output_path.exists():
        return False
    return compute_file_sha256(output_path) == entry["sha256"]


def generate_chapter_assets(chapter: CanonicalChapter, config: PipelineConfig):
    """Renders every real asset for `chapter` and returns (layout, packaging_assets, storyboard,
    warnings). Real per-asset language-slot tracking: a title card bakes real Russian text
    (Russian slot), a PSD panel is wordless visual art (language-neutral slot) -- both are
    `kind="art"`, so this distinction can only be made here, at render time, not reconstructed
    later from data.json alone (see validate_output.py's own documented scope limit)."""
    storyboard = build_deterministic_storyboard(chapter)
    warnings = list(storyboard.warnings)
    if config.use_ai:
        warnings.append(
            "AI storyboard mode (--no-ai not passed) requested but Task 2.2 (Ollama-backed "
            "storyboard) is not yet implemented; used the deterministic storyboard instead"
        )

    # (kind, image, contains_russian_text)
    content: list[tuple[str, object, bool]] = [
        ("art", render_cards.render_title_card(chapter.order, chapter.title, book_id=DEFAULT_BOOK_ID), True)
    ]

    if config.use_psd and chapter.order == PSD_CHAPTER_ORDER:
        from layout_chapter import CONTENT_WIDTH

        for psd_name in PSD_FILENAMES:
            result = import_psd_panel(PSD_DIR / psd_name, content_width=CONTENT_WIDTH)
            if result.image is not None:
                content.append(("art", result.image, False))
            else:
                warnings.append(result.warning)

    for sloka in chapter.slokas:
        content.append(("balloon", render_cards.render_verse_card(sloka, chapter.order, book_id=DEFAULT_BOOK_ID), True))

    geometry = [(kind, image) for kind, image, _ in content]
    _, total_height = layout_chapter_content(geometry)
    background = render_cards.render_chapter_background(
        render_cards.theme_for_chapter(chapter.order), CANVAS_WIDTH, total_height
    )
    layout = layout_chapter(chapter, geometry, background)

    russian_flags = [False] + [flag for _, _, flag in content]  # background is language-neutral
    assets = [
        PackagingAsset(
            kind=asset.kind, image=asset.image, x=asset.x, y=asset.y,
            stem=f"{asset.kind}_{index:03d}", contains_russian_text=russian_flags[index],
        )
        for index, asset in enumerate(layout.assets)
    ]
    return layout, assets, storyboard, tuple(warnings)


def process_chapter(
    chapter: CanonicalChapter,
    output_dir: Path,
    dataset_fingerprint: str,
    config_fingerprint: str,
    config: PipelineConfig,
    previous_manifest: dict | None,
) -> tuple[ChapterManifestEntry, bool]:
    """Returns (entry, was_reused). Never raises: any failure becomes a status="failed" entry so
    a batch (--all) run can continue to the next chapter, per Specifications' "the batch continues
    after an individual chapter failure" rule."""
    output_path = _output_path(output_dir, chapter)

    if not config.force and can_reuse_chapter(previous_manifest, chapter, output_path, dataset_fingerprint, config_fingerprint):
        entry = ChapterManifestEntry.from_dict(
            next(c for c in previous_manifest["chapters"] if c["order"] == chapter.order)
        )
        return entry, True

    try:
        with chapter_lock(output_dir, chapter):
            layout, assets, storyboard, warnings = generate_chapter_assets(chapter, config)
            render_cards.shutdown_browser()  # release Chromium before the (memory-heavy) PSD/packaging step settles
            write_comics_archive(output_path, layout.width, layout.height, assets)
            validation = validate_archive_structure(output_path, expected_verse_count=len(chapter.slokas))
            entry = build_chapter_entry(
                chapter, output_path, layer_count=len(assets), width=layout.width, height=layout.height,
                storyboard=storyboard, validation_result=validation,
                psd_inputs=tuple(PSD_FILENAMES) if (config.use_psd and chapter.order == PSD_CHAPTER_ORDER) else (),
            )
            # merge storyboard/pipeline-level warnings into the entry actually written
            entry = ChapterManifestEntry(**{**entry.__dict__, "warnings": entry.warnings + warnings})
            return entry, False
    except Exception as exc:  # noqa: BLE001 -- intentionally broad: one chapter's crash must not abort the batch
        return build_failed_chapter_entry(chapter, str(exc)), False


def run(chapters_to_run: list[CanonicalChapter], output_dir: Path, config: PipelineConfig) -> dict:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    previous_manifest = load_previous_manifest(manifest_path)

    dataset_fingerprint = compute_dataset_fingerprint(DATASET_DIR)
    config_fingerprint = compute_config_fingerprint()

    entries_by_order: dict[int, ChapterManifestEntry] = {}
    if previous_manifest is not None:
        for raw in previous_manifest["chapters"]:
            entries_by_order[raw["order"]] = ChapterManifestEntry.from_dict(raw)

    for chapter in chapters_to_run:
        entry, reused = process_chapter(chapter, output_dir, dataset_fingerprint, config_fingerprint, config, previous_manifest)
        entries_by_order[chapter.order] = entry
        # Logs report IDs/counts/status only -- never full source/comment text, per Specifications.
        tag = "reused" if reused else entry.status
        print(f"chapter {chapter.order:02d} ({chapter.chapter_id}): {tag}", file=sys.stderr)

    final_entries = [entries_by_order[order] for order in sorted(entries_by_order)]
    manifest = build_manifest(
        DATASET_DIR, DEFAULT_BOOK_ID, "ru", EXPECTED_CHAPTER_COUNT, EXPECTED_SLOKA_COUNT, final_entries
    )
    write_manifest(manifest_path, manifest)
    (output_dir / "report.md").write_text(render_report_md(manifest), encoding="utf-8")
    return manifest


def run_lottie_source(output_dir: Path, source_path: Path = LOTTIE_SOURCE) -> dict:
    """Produces the standalone Lottie-derived document; never enters/counts toward the 18 chapters."""
    output_dir.mkdir(parents=True, exist_ok=True)
    imported = import_lottie_file(source_path)
    output_path = output_dir / LOTTIE_OUTPUT_NAME
    write_comics_archive(
        output_path,
        imported.width,
        imported.height,
        list(imported.assets),
        camera_path=list(imported.camera_path),
        preferred_viewport_width=imported.width,
        preferred_viewport_height=1600,
    )
    nonzero_depths = {asset.z_depth for asset in imported.assets if asset.z_depth != 0}
    manifest = {
        "schema_version": 1,
        "source": str(source_path),
        "output_file": output_path.name,
        "standalone": True,
        "counts_toward_chapters": False,
        "scene_count": imported.scene_count,
        "image_layer_count": imported.image_layer_count,
        "animated_layer_count": imported.animated_layer_count,
        "camera_point_count": len(imported.camera_path),
        "camera_reference_layers": list(imported.reference_layers),
        "distinct_nonzero_z_depth_count": len(nonzero_depths),
        "parallax_rendered_by_current_viewers": False,
        "sha256": compute_file_sha256(output_path),
    }
    write_manifest(output_dir / "lottie_manifest.json", manifest)
    (output_dir / "lottie_report.md").write_text(
        render_lottie_report_md(manifest), encoding="utf-8"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument("--chapter", type=int, metavar="N")
    target_group.add_argument("--all", action="store_true")
    target_group.add_argument("--lottie-source", action="store_true")
    parser.add_argument("--no-ai", action="store_true")
    parser.add_argument("--no-psd", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args(argv)

    if args.lottie_source:
        manifest = run_lottie_source(args.output_dir)
        print(f"lottie output: {args.output_dir / manifest['output_file']}", file=sys.stderr)
        print(f"lottie manifest: {args.output_dir / 'lottie_manifest.json'}", file=sys.stderr)
        print(f"lottie report: {args.output_dir / 'lottie_report.md'}", file=sys.stderr)
        return 0

    all_chapters = load_book_one(dataset_dir=DATASET_DIR, book_id=DEFAULT_BOOK_ID)
    verify_dataset_integrity(all_chapters)

    if args.all:
        targets = list(all_chapters)
    else:
        targets = [c for c in all_chapters if c.order == args.chapter]
        if not targets:
            print(f"error: no chapter with order={args.chapter} (expected 1-{EXPECTED_CHAPTER_COUNT})", file=sys.stderr)
            return 2

    config = PipelineConfig(use_ai=not args.no_ai, use_psd=not args.no_psd, force=args.force)
    manifest = run(targets, args.output_dir, config)

    valid = coverage_count(manifest)
    failed_orders = [c["order"] for c in manifest["chapters"] if c["status"] != "valid"]
    print(f"valid/expected: {valid}/{EXPECTED_CHAPTER_COUNT}", file=sys.stderr)
    if failed_orders:
        print(f"failed chapters: {failed_orders}", file=sys.stderr)
    print(f"manifest: {args.output_dir / 'manifest.json'}", file=sys.stderr)
    print(f"report: {args.output_dir / 'report.md'}", file=sys.stderr)

    if args.all:
        return 0 if valid >= EXPECTED_CHAPTER_COUNT else 1
    return 0 if manifest["chapters"] and any(
        c["order"] == args.chapter and c["status"] == "valid" for c in manifest["chapters"]
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
