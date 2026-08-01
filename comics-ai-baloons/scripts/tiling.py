"""Stitch/re-tile balloon images per the reverse-engineered editor tiling algorithm.

Ground truth (apps/comics-editor-v2.9/native/Comics.Editor/Utils/FileManager.cs +
IWS/Utils/ImageMagick.cs): 512x512px tiles, filename template
"<basename>_<scale*1000>_<col>_<row>.<ext>", col = floor(x/512), row = floor(y/512), edge tiles
clipped (not padded) to the declared image bounds. Comics layers (not puzzle assets) always use
scale 1.0, so the scale placeholder is always literally "1000".
"""

from __future__ import annotations

import io
import math

from PIL import Image

import comics_io

TILE_SIZE = 512
SCALE_1000 = 1000  # ComicsScales = [1.0] for non-puzzle comics layers


def tile_grid(width: int, height: int, tile_size: int = TILE_SIZE) -> list[tuple[int, int, tuple[int, int, int, int]]]:
    """List of (col, row, box) covering a width x height canvas, edge tiles clipped."""
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


def tile_filename(file_template: str, col: int, row: int, scale_1000: int = SCALE_1000) -> str:
    return file_template.format(scale_1000, col, row)


def stitch_image(
    archive: comics_io.ComicsArchive,
    file_template: str,
    width: int,
    height: int,
    *,
    layer_dir: str = "layers/",
) -> Image.Image:
    """Reconstruct a full-size image from its 512px tiles inside `archive`."""
    canvas = Image.new("RGBA", (width, height))
    available = set(archive.names())
    for col, row, box in tile_grid(width, height):
        name = layer_dir + tile_filename(file_template, col, row)
        if name not in available:
            raise FileNotFoundError(f"missing tile {name} for {file_template} ({width}x{height})")
        tile_bytes = archive.read_bytes(name)
        tile_img = Image.open(io.BytesIO(tile_bytes)).convert("RGBA")
        canvas.paste(tile_img, (box[0], box[1]))
    return canvas


def retile_image(image: Image.Image, file_template: str) -> dict[str, bytes]:
    """Split `image` into 512px tiles named per `file_template`, 32-bit PNG encoded.

    Returns {relative tile filename (no "layers/" prefix): png bytes}.
    """
    width, height = image.size
    rgba = image.convert("RGBA")
    out: dict[str, bytes] = {}
    for col, row, box in tile_grid(width, height):
        tile = rgba.crop(box)
        buf = io.BytesIO()
        tile.save(buf, format="PNG")
        out[tile_filename(file_template, col, row)] = buf.getvalue()
    return out
