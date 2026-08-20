#!/usr/bin/env python3
"""Task 5.2 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): split a rendered
layer image into 512x512 tiles named `<stem>_1000_<col>_<row>.png`, per
02-specifications.md's `.comics` Packaging Contract.

Fresh, minimal, dependency-free (Pillow only) implementation of the same tiling contract already
proven in `apps/comics-ai/comics-ai-baloons/scripts/tiling.py` -- per this app's Plan, "reuse by
contract, not import accident": that module's `tile_grid`/`tile_filename`/`retile_image` algorithm
(ceil-based grid, edge tiles clipped not padded, `scale_1000` always literally 1000 for non-puzzle
comics layers) is reproduced here rather than imported across apps/venvs, since it's a small,
stable, independently-verifiable contract rather than a shared library this repo has any
mechanism to depend on cleanly.
"""

from __future__ import annotations

import io
import math

from PIL import Image

TILE_SIZE = 512
SCALE_1000 = 1000  # comics layers always use scale 1.0 -> literal "1000" placeholder


def tile_grid(width: int, height: int, tile_size: int = TILE_SIZE) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """List of (col, row, box) covering a width x height canvas, edge tiles clipped (not padded)."""
    cols = math.ceil(width / tile_size)
    rows = math.ceil(height / tile_size)
    out = []
    for row in range(rows):
        for col in range(cols):
            box = (
                col * tile_size,
                row * tile_size,
                min((col + 1) * tile_size, width),
                min((row + 1) * tile_size, height),
            )
            out.append((col, row, box))
    return out


def tile_filename(stem: str, col: int, row: int, scale_1000: int = SCALE_1000) -> str:
    return f"{stem}_{scale_1000}_{col}_{row}.png"


def retile_image(image: Image.Image, stem: str) -> dict[str, bytes]:
    """Splits `image` into 512px tiles, 32-bit PNG encoded. Returns
    {tile filename (no "layers/" prefix): png bytes}."""
    width, height = image.size
    rgba = image.convert("RGBA")
    out: dict[str, bytes] = {}
    for col, row, box in tile_grid(width, height):
        tile = rgba.crop(box)
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        out[tile_filename(stem, col, row)] = buf.getvalue()
    return out


def stitch_tiles(tiles: dict[str, bytes], stem: str, width: int, height: int) -> Image.Image:
    """Reconstructs a full-size image from a {filename: png bytes} tile set. Used here for
    round-trip verification; `package_comics.py`/an editor-side reader would do the equivalent
    against a real archive."""
    canvas = Image.new("RGBA", (width, height))
    for col, row, box in tile_grid(width, height):
        name = tile_filename(stem, col, row)
        if name not in tiles:
            raise FileNotFoundError(f"missing tile {name} for stem={stem!r} ({width}x{height})")
        tile_img = Image.open(io.BytesIO(tiles[name])).convert("RGBA")
        canvas.paste(tile_img, (box[0], box[1]))
    return canvas
