#!/usr/bin/env python3
"""Task 6.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): build a `.comics`
ZIP archive (`data.json` + `layers/*.png`) per 02-specifications.md's corrected v0.2 Packaging
Contract.

Real, verified contract (checked directly against a real production archive this session --
dataset/mahabharata/boranko/mahabharata-dot-comics_v2012/zip_by_uid/
8a89f7d689fb441ea280cd782276bd7a.comics's own English+Russian balloon layer): root keys
`{width, height, layers, sounds}`; each layer's `images` is a 3-slot array indexed by the real
`Cultures` enum `{En=0, Ru=1, Hi=2}`. That real layer's own `images` array is
`[{"file": "b10_eng_..."}, {"file": "b10_ru_..."}, {}]` -- English at slot 0, Russian at slot 1 --
directly confirming Specifications' corrected contract against real production data, not just
source code. This module never places Russian content at slot 0.

Judgment call, extending that correction to a case Specifications doesn't separately examine:
this generator's rendered title cards bake real Russian text into their pixels (unlike the plain
color-field background or the wordless PSD art panels), so `contains_russian_text=True` puts them
in the same Russian slot (1) as verse cards. Pure-visual layers (background, PSD panels) use the
language-neutral slot 0, which is also the real `Images.FirstOrDefault()` fallback target, so they
render regardless of the active culture. Flagged here for Anton to redirect if a different
treatment was intended.
"""

from __future__ import annotations

import json
import math
import zipfile
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from tile_assets import retile_image

RUSSIAN_SLOT_INDEX = 1  # real Ru index in Cultures.cs's {En=0, Ru=1, Hi=2}
LANGUAGE_NEUTRAL_SLOT_INDEX = 0  # also the real Images.FirstOrDefault() fallback target
TRANSLATE_ANIM_TYPE = "Comics.Editor.Models.TranslateAnim, Comics.Editor"

# Fixed, non-real-time timestamp so identical inputs produce byte-identical archives/hashes, per
# Specifications' "ZIP entry order is deterministic ... and timestamps are normalized" rule.
_ZIP_FIXED_TIMESTAMP = (2026, 1, 1, 0, 0, 0)


@dataclass(frozen=True)
class PackagingAsset:
    kind: str  # "background" | "art" | "balloon"
    image: Image.Image
    x: int
    y: int
    stem: str  # unique tile-filename stem for this layer, e.g. "verse_003"
    contains_russian_text: bool  # True -> Ru slot (1); False -> language-neutral slot (0)
    animations: tuple[dict, ...] = ()
    z_depth: float = 0.0


def _layer_json(asset: PackagingAsset) -> dict:
    slot = RUSSIAN_SLOT_INDEX if asset.contains_russian_text else LANGUAGE_NEUTRAL_SLOT_INDEX
    images: list[dict] = [{}, {}, {}]
    images[slot] = {
        "file": f"{asset.stem}_{{0}}_{{1}}_{{2}}.png",
        "width": asset.image.width,
        "height": asset.image.height,
    }
    layer = {
        "images": images,
        "animations": list(asset.animations) or [
            {"$type": TRANSLATE_ANIM_TYPE, "x": asset.x, "y": asset.y}
        ],
        "kind": asset.kind,
    }
    if not math.isfinite(asset.z_depth) or asset.z_depth <= -1:
        raise ValueError(f"invalid zDepth for {asset.stem!r}: {asset.z_depth!r}")
    if asset.z_depth != 0:
        layer["zDepth"] = round(asset.z_depth, 3)
    return layer


def _canonical_camera_path(camera_path: list[dict] | None) -> list[dict] | None:
    if camera_path is None:
        return None
    canonical: list[dict] = []
    previous: int | None = None
    for point in camera_path:
        position = point.get("position")
        x, y = point.get("x"), point.get("y")
        if not isinstance(position, int) or isinstance(position, bool):
            raise ValueError(f"cameraPath position must be an int: {position!r}")
        if previous is not None and position <= previous:
            raise ValueError("cameraPath positions must be strictly increasing")
        if not isinstance(x, (int, float)) or not isinstance(y, (int, float)):
            raise ValueError("cameraPath x/y must be numeric")
        if not math.isfinite(float(x)) or not math.isfinite(float(y)):
            raise ValueError("cameraPath x/y must be finite")
        canonical.append({"position": position, "x": round(float(x), 3), "y": round(float(y), 3)})
        previous = position
    return canonical


def build_data_json(
    width: int,
    height: int,
    assets: list[PackagingAsset],
    *,
    camera_path: list[dict] | None = None,
    preferred_viewport_width: int | None = None,
    preferred_viewport_height: int | None = None,
) -> dict:
    stems = [a.stem for a in assets]
    if len(stems) != len(set(stems)):
        raise ValueError(f"duplicate asset stems (would collide as tile filenames): {stems}")
    for asset in assets:
        if "/" in asset.stem or ".." in asset.stem:
            raise ValueError(f"unsafe asset stem (path traversal risk): {asset.stem!r}")
    data = {
        "width": width,
        "height": height,
        "layers": [_layer_json(a) for a in assets],
        "sounds": [],
    }
    canonical_path = _canonical_camera_path(camera_path)
    if canonical_path:
        data["cameraPath"] = canonical_path
    if preferred_viewport_width is not None:
        data["preferredViewportWidth"] = preferred_viewport_width
    if preferred_viewport_height is not None:
        data["preferredViewportHeight"] = preferred_viewport_height
    return data


def build_tiles(assets: list[PackagingAsset]) -> dict[str, bytes]:
    tiles: dict[str, bytes] = {}
    for asset in assets:
        asset_tiles = retile_image(asset.image, asset.stem)
        collision = set(asset_tiles) & set(tiles)
        if collision:
            raise ValueError(f"tile name collision across assets: {collision}")
        tiles.update(asset_tiles)
    return tiles


def _write_deterministic_entry(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, date_time=_ZIP_FIXED_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def build_archive_bytes(
    width: int,
    height: int,
    assets: list[PackagingAsset],
    **data_options,
) -> bytes:
    """Builds the full archive in memory: deterministic entry order (`data.json` first, then
    lexically sorted tiles) and fixed timestamps, so identical inputs produce byte-identical
    output (and therefore identical SHA-256 hashes)."""
    import io

    data = build_data_json(width, height, assets, **data_options)
    tiles = build_tiles(assets)

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        _write_deterministic_entry(
            zf, "data.json", json.dumps(data, ensure_ascii=False, indent=2).encode("utf-8")
        )
        for name in sorted(tiles):
            _write_deterministic_entry(zf, f"layers/{name}", tiles[name])
    return buf.getvalue()


def write_comics_archive(
    path: Path,
    width: int,
    height: int,
    assets: list[PackagingAsset],
    **data_options,
) -> None:
    """Staging-then-atomic-replace write: builds the full archive at `path.name + '.staging'`
    first, then atomically renames it into place, so a failed/interrupted write never leaves a
    corrupt or partial file at `path`."""
    archive_bytes = build_archive_bytes(width, height, assets, **data_options)
    staging_path = path.with_name(path.name + ".staging")
    staging_path.parent.mkdir(parents=True, exist_ok=True)
    staging_path.write_bytes(archive_bytes)
    staging_path.replace(path)  # atomic on POSIX
