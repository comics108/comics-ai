#!/usr/bin/env python3
"""Task 7.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): structural and
source-fidelity validation per 02-specifications.md's "Structural archive validation" and "Source
fidelity validation" sections.

Scope note on the Russian-slot regression check: Specifications asks generically for "every
layer's images[] has its Russian content at index 1, not index 0." This module can only enforce
that unambiguously for `kind="balloon"` layers, since every balloon layer in this generator *is*
a verse card (real Russian text, by construction -- see package_comics.py). `kind="art"` layers
are ambiguous from data.json alone: a rendered title card carries real Russian text (should be
slot 1), but a chapter-5 PSD panel is wordless visual art (correctly slot 0, matching the real
`Images.FirstOrDefault()` fallback target) -- validate_output.py cannot tell these apart without
extra provenance the archive format doesn't carry, so it does not flag art-layer slot placement.

Scope note on source-fidelity validation: full CSV-string round-tripping needs a companion
checkpoint/manifest recording exactly which source record produced each layer -- that's Task
8.1's `manifest.json`, not yet built. What *is* checkable here without it, per Specifications'
own "source fidelity is ... guaranteed ... by renderer snapshot/hash tests" framing (already
covered by Tasks 3.1/6.1's real-data tests) plus one additional real check this module adds:
`validate_storyboard_citations` catches a storyboard scene citing a sloka order that doesn't
exist in its own chapter -- the concrete failure mode an AI-generated (Task 2.2) storyboard could
introduce that Task 2.1's deterministic storyboard structurally cannot.
"""

from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

from build_storyboard import ChapterStoryboard
from models import CanonicalChapter
from tile_assets import stitch_tiles, tile_grid

REQUIRED_ROOT_KEYS = {"width", "height", "layers", "sounds"}
TRANSLATE_ANIM_TYPE_PREFIX = "Comics.Editor.Models.TranslateAnim"
_TILE_FILE_SUFFIX = "_{0}_{1}_{2}.png"


@dataclass(frozen=True)
class ValidationIssue:
    check: str
    message: str


@dataclass(frozen=True)
class ValidationResult:
    issues: tuple[ValidationIssue, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.issues


def _issue(check: str, message: str) -> ValidationIssue:
    return ValidationIssue(check=check, message=message)


def validate_archive_structure(path: Path, expected_verse_count: int) -> ValidationResult:
    issues: list[ValidationIssue] = []

    try:
        zf = zipfile.ZipFile(path)
    except zipfile.BadZipFile as exc:
        return ValidationResult((_issue("zip_open", f"{path}: not a valid ZIP: {exc}"),))

    if zf.testzip() is not None:
        issues.append(_issue("zip_corrupt", f"corrupt member: {zf.testzip()}"))

    names = zf.namelist()
    if names.count("data.json") != 1:
        issues.append(_issue("data_json_count", f"expected exactly 1 data.json, found {names.count('data.json')}"))
        return ValidationResult(tuple(issues))

    lower_names = [n.lower() for n in names]
    if len(lower_names) != len(set(lower_names)):
        issues.append(_issue("case_collision", "duplicate or case-colliding ZIP member names found"))
    if len(names) != len(set(names)):
        issues.append(_issue("duplicate_member", "duplicate ZIP member names found"))
    for name in names:
        if name.startswith("/") or ".." in Path(name).parts:
            issues.append(_issue("unsafe_path", f"unsafe path in archive: {name!r}"))

    try:
        data = json.loads(zf.read("data.json").decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        issues.append(_issue("data_json_parse", f"data.json is not valid UTF-8 JSON: {exc}"))
        return ValidationResult(tuple(issues))

    missing_keys = REQUIRED_ROOT_KEYS - data.keys()
    if missing_keys:
        issues.append(_issue("root_keys", f"missing required root keys: {missing_keys}"))
        return ValidationResult(tuple(issues))

    width, height, layers, sounds = data["width"], data["height"], data["layers"], data["sounds"]
    if not isinstance(width, int) or width <= 0:
        issues.append(_issue("root_types", f"width must be a positive int, got {width!r}"))
    if not isinstance(height, int) or height <= 0:
        issues.append(_issue("root_types", f"height must be a positive int, got {height!r}"))
    if not isinstance(layers, list):
        issues.append(_issue("root_types", "layers must be a list"))
        return ValidationResult(tuple(issues))
    if sounds != []:
        issues.append(_issue("sounds_not_empty", f"sounds must be empty, got {sounds!r}"))

    background_count = sum(1 for layer in layers if layer.get("kind") == "background")
    art_count = sum(1 for layer in layers if layer.get("kind") == "art")
    balloon_layers = [layer for layer in layers if layer.get("kind") == "balloon"]

    if background_count < 1:
        issues.append(_issue("layer_counts", "expected at least 1 background layer"))
    if art_count < 1:
        issues.append(_issue("layer_counts", "expected at least 1 title/art layer"))
    if len(balloon_layers) != expected_verse_count:
        issues.append(_issue(
            "verse_count",
            f"expected {expected_verse_count} verse (balloon) layers, found {len(balloon_layers)}",
        ))

    for layer in balloon_layers:
        images = layer.get("images", [])
        if len(images) != 3:
            issues.append(_issue("images_shape", f"balloon layer images[] must have length 3, got {len(images)}"))
            continue
        if images[0]:
            issues.append(_issue(
                "russian_slot_regression",
                "balloon layer has content at images[0] (En) -- Russian content must be at images[1] (Ru)",
            ))
        if not images[1]:
            issues.append(_issue("russian_slot_missing", "balloon layer has no content at images[1] (Ru)"))

    for index, layer in enumerate(layers):
        images = layer.get("images", [])
        populated = [(slot, image) for slot, image in enumerate(images) if image]
        if not populated:
            issues.append(_issue("empty_layer", f"layer {index} has no populated image slot"))
            continue

        translate = next(
            (a for a in layer.get("animations", []) if a.get("$type", "").startswith(TRANSLATE_ANIM_TYPE_PREFIX)),
            None,
        )
        if translate is None:
            issues.append(_issue("missing_translate_anim", f"layer {index} has no TranslateAnim"))
            continue
        x, y = translate.get("x"), translate.get("y")

        for slot, image in populated:
            w, h = image.get("width"), image.get("height")
            if not isinstance(w, int) or not isinstance(h, int) or w <= 0 or h <= 0:
                issues.append(_issue("image_dims", f"layer {index} slot {slot} has invalid width/height"))
                continue
            if x is None or y is None or x < 0 or y < 0 or x + w > width or y + h > height:
                issues.append(_issue(
                    "out_of_bounds",
                    f"layer {index} slot {slot} rect (x={x}, y={y}, w={w}, h={h}) doesn't fit canvas ({width}x{height})",
                ))

            file_template = image.get("file", "")
            if not file_template.endswith(_TILE_FILE_SUFFIX):
                issues.append(_issue("tile_template", f"unexpected file template: {file_template!r}"))
                continue
            stem = file_template[: -len(_TILE_FILE_SUFFIX)]

            expected_tile_names = {f"layers/{stem}_1000_{col}_{row}.png" for col, row, _ in tile_grid(w, h)}
            missing = expected_tile_names - set(names)
            if missing:
                issues.append(_issue(
                    "missing_tile", f"layer {index} slot {slot} missing tiles: {sorted(missing)[:3]}"
                ))
                continue

            tile_bytes = {name.split("/", 1)[1]: zf.read(name) for name in expected_tile_names}
            stitched = stitch_tiles(tile_bytes, stem, w, h)
            if stitched.size != (w, h):
                issues.append(_issue(
                    "tile_size_mismatch", f"layer {index} slot {slot} stitched size {stitched.size} != declared ({w},{h})"
                ))
            if stitched.getchannel("A").getextrema()[1] == 0:
                issues.append(_issue("fully_transparent", f"layer {index} slot {slot} image is fully transparent"))

    return ValidationResult(tuple(issues))


def validate_storyboard_citations(chapter: CanonicalChapter, storyboard: ChapterStoryboard) -> ValidationResult:
    real_orders = {sloka.order for sloka in chapter.slokas}
    issues: list[ValidationIssue] = []
    for scene in storyboard.scenes:
        bad = [order for order in scene.source_sloka_orders if order not in real_orders]
        if bad:
            issues.append(_issue(
                "citation_out_of_chapter",
                f"scene {scene.scene_id} cites orders not in chapter {chapter.order}: {bad}",
            ))
    return ValidationResult(tuple(issues))
