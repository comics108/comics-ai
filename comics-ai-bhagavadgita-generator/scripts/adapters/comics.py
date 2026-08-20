"""Recover `.comics` runtime fixtures as reference evidence, without approval promotion."""

from __future__ import annotations

import json
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from PIL import Image

from tile_assets import stitch_tiles, tile_grid


_TILED_TEMPLATE = re.compile(r"^(?P<stem>.+)_\{0\}_\{1\}_\{2\}\.png$")


@dataclass(frozen=True)
class RecoveredComicsLayer:
    index: int
    kind: str | None
    layer_id: str | None
    parent_id: str | None
    visible: bool
    populated_slots: tuple[int, ...]
    image_slots: tuple[dict[str, Any], ...]
    animations: tuple[dict[str, Any], ...]
    translations: tuple[tuple[str, str], ...]
    z_depth: float


@dataclass(frozen=True)
class RecoveredComicsDocument:
    source_path: Path
    width: int
    height: int
    layers: tuple[RecoveredComicsLayer, ...]
    sounds: tuple[dict[str, Any], ...]
    camera_path: tuple[dict[str, Any], ...]
    preferred_viewport: tuple[int | None, int | None]
    evidence_class: str


def _safe_archive_names(archive: zipfile.ZipFile) -> frozenset[str]:
    names = archive.namelist()
    if len(names) != len(set(names)):
        raise ValueError("duplicate ZIP members are not valid recovery evidence")
    for name in names:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe archive path: {name!r}")
    return frozenset(names)


def _document(path: Path) -> tuple[Path, dict[str, Any]]:
    source = path.resolve(strict=True)
    try:
        with zipfile.ZipFile(source) as archive:
            names = _safe_archive_names(archive)
            if "data.json" not in names:
                raise ValueError(f".comics archive has no data.json: {source}")
            document = json.loads(archive.read("data.json").decode("utf-8-sig"))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"invalid .comics ZIP: {source}") from exc
    if not isinstance(document, dict):
        raise ValueError(".comics data.json root must be an object")
    return source, document


def recover_comics_structure(path: Path) -> RecoveredComicsDocument:
    """Recover slots, transforms, animation/text/audio provenance as unapproved evidence."""
    source, document = _document(path)
    width = document.get("width")
    height = document.get("height")
    raw_layers = document.get("layers")
    if not isinstance(width, int) or width <= 0 or not isinstance(height, int) or height <= 0:
        raise ValueError(".comics width and height must be positive integers")
    if not isinstance(raw_layers, list):
        raise ValueError(".comics layers must be an array")

    layers: list[RecoveredComicsLayer] = []
    for index, raw in enumerate(raw_layers):
        if not isinstance(raw, dict):
            raise ValueError(f"layer {index} must be an object")
        images = raw.get("images") or []
        animations = raw.get("animations") or raw.get("anims") or []
        translations = raw.get("translations") or {}
        if not isinstance(images, list) or not all(isinstance(item, dict) for item in images):
            raise ValueError(f"layer {index} images must be an array of objects")
        if not isinstance(animations, list) or not all(isinstance(item, dict) for item in animations):
            raise ValueError(f"layer {index} animations must be an array of objects")
        if isinstance(translations, dict):
            translation_items = tuple(
                sorted((str(language), str(text)) for language, text in translations.items())
            )
        else:
            translation_items = ()
        depth = raw.get("zDepth", 0)
        if not isinstance(depth, (int, float)) or isinstance(depth, bool):
            raise ValueError(f"layer {index} zDepth must be numeric")
        layers.append(
            RecoveredComicsLayer(
                index=index,
                kind=str(raw["kind"]) if raw.get("kind") is not None else None,
                layer_id=str(raw["id"]) if raw.get("id") is not None else None,
                parent_id=str(raw["parentId"]) if raw.get("parentId") is not None else None,
                visible=bool(raw.get("visible", True)),
                populated_slots=tuple(slot for slot, image in enumerate(images) if image),
                image_slots=tuple(dict(image) for image in images),
                animations=tuple(dict(animation) for animation in animations),
                translations=translation_items,
                z_depth=float(depth),
            )
        )

    sounds = document.get("sounds") or []
    camera_path = document.get("cameraPath") or []
    if not isinstance(sounds, list) or not all(isinstance(item, dict) for item in sounds):
        raise ValueError(".comics sounds must be an array of objects")
    if not isinstance(camera_path, list) or not all(isinstance(item, dict) for item in camera_path):
        raise ValueError(".comics cameraPath must be an array of objects")
    return RecoveredComicsDocument(
        source_path=source,
        width=width,
        height=height,
        layers=tuple(layers),
        sounds=tuple(dict(sound) for sound in sounds),
        camera_path=tuple(dict(point) for point in camera_path),
        preferred_viewport=(
            document.get("preferredViewportWidth"),
            document.get("preferredViewportHeight"),
        ),
        evidence_class="runtime_reference_unapproved",
    )


def recover_comics_layer(path: Path, *, layer_index: int, slot: int) -> Image.Image:
    """Lazily stitch one selected language/image slot into source-resolution RGBA."""
    document = recover_comics_structure(path)
    if layer_index < 0 or layer_index >= len(document.layers):
        raise IndexError(f"layer index out of range: {layer_index}")
    layer = document.layers[layer_index]
    if slot < 0 or slot >= len(layer.image_slots):
        raise IndexError(f"image slot out of range: {slot}")
    image = layer.image_slots[slot]
    if not image:
        raise ValueError(f"layer {layer_index} image slot {slot} is empty")
    file_template = image.get("file")
    width = image.get("width")
    height = image.get("height")
    if not isinstance(file_template, str) or not isinstance(width, int) or not isinstance(height, int):
        raise ValueError(f"layer {layer_index} slot {slot} has invalid image metadata")
    if width <= 0 or height <= 0:
        raise ValueError(f"layer {layer_index} slot {slot} has non-positive dimensions")
    match = _TILED_TEMPLATE.fullmatch(file_template)
    if match is None:
        raise ValueError(f"unsupported .comics image template: {file_template!r}")
    stem = match.group("stem")

    with zipfile.ZipFile(document.source_path) as archive:
        names = _safe_archive_names(archive)
        tile_names = {
            f"layers/{stem}_1000_{column}_{row}.png"
            for column, row, _ in tile_grid(width, height)
        }
        missing = tile_names - names
        if missing:
            raise FileNotFoundError(
                f"layer {layer_index} slot {slot} is missing tiles: {sorted(missing)[:3]}"
            )
        tiles = {name.removeprefix("layers/"): archive.read(name) for name in tile_names}
    return stitch_tiles(tiles, stem, width, height)
