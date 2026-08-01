#!/usr/bin/env python3
"""Stage 6.2: layout -- find a balloon's text-safe interior rectangle and fit text into it.

Two independent pieces:
1. `find_interior_rect`: locate the largest axis-aligned rectangle that fits entirely inside the
   balloon, at least `margin` pixels away from the drawn outline stroke (classic "maximal
   rectangle in a binary matrix" problem, via a per-row largest-rectangle-in-histogram scan). This
   is script-agnostic -- it only looks at the balloon's shape, not the text.
2. `fit_text_to_rect`: given that rectangle and a target string, greedy word-wrap + binary-search
   the largest font size where the wrapped lines fit. This assumes space-delimited word wrapping
   (fine for Latin/Cyrillic and most Specifications-listed languages); CJK line-breaking rules
   differ and are handled separately in the headless-browser path (stage 6.4), which can still
   reuse `find_interior_rect` for the target box.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

DEFAULT_MARGIN = 6  # px, minimum distance from the drawn outline stroke


def find_safe_interior_mask(img: Image.Image, margin: int = DEFAULT_MARGIN) -> np.ndarray:
    """Boolean mask: True where a pixel is inside the balloon and >= margin px from the outline
    ink. Assumes `img` is an *empty* balloon (erase.py output) -- no text ink to confuse this.
    """
    arr = np.array(img.convert("RGBA"))
    alpha = arr[:, :, 3]
    inside = alpha > 128

    gray = cv2.cvtColor(arr[:, :, :3], cv2.COLOR_RGB2GRAY)
    ink = (gray < 128) & inside

    not_ink = (~ink).astype(np.uint8)
    dist = cv2.distanceTransform(not_ink, cv2.DIST_L2, 5)
    return inside & (dist >= margin)


def largest_inscribed_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Largest axis-aligned all-True rectangle in a 2D boolean array. Returns (x, y, w, h).
    Classic per-row "largest rectangle in histogram" scan, O(rows * cols).
    """
    h, w = mask.shape
    heights = np.zeros(w, dtype=int)
    best = (0, 0, 0, 0)
    best_area = 0

    for row in range(h):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[tuple[int, int]] = []  # (start_col, height)
        for col in range(w + 1):
            cur_h = int(heights[col]) if col < w else 0
            start = col
            while stack and stack[-1][1] >= cur_h:
                idx, ht = stack.pop()
                width = col - idx
                area = ht * width
                if area > best_area:
                    best_area = area
                    best = (idx, row - ht + 1, width, ht)
                start = idx
            stack.append((start, cur_h))
    return best


def find_interior_rect(
    empty_balloon_img: Image.Image, margin: int = DEFAULT_MARGIN
) -> tuple[int, int, int, int]:
    mask = find_safe_interior_mask(empty_balloon_img, margin)
    return largest_inscribed_rectangle(mask)


def _wrap_text(text: str, font: ImageFont.FreeTypeFont, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
    words = text.split()
    if not words:
        return []
    lines: list[str] = []
    current = words[0]
    for word in words[1:]:
        candidate = f"{current} {word}"
        if draw.textlength(candidate, font=font) <= max_width:
            current = candidate
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return lines


def fit_text_to_rect(
    text: str,
    rect_w: int,
    rect_h: int,
    font_path: str,
    min_size: int = 10,
    max_size: int = 96,
) -> tuple[int, list[str]]:
    """Binary-search the largest font size where `text`, greedily word-wrapped to `rect_w`,
    fits within `rect_h`. Returns (font_size, wrapped_lines). Falls back to `min_size` (possibly
    overflowing) if even the smallest size doesn't fit -- caller decides how to handle overflow
    (Specifications' "text_overflow" edge case).
    """
    scratch = Image.new("RGB", (max(rect_w, 1), max(rect_h, 1)))
    draw = ImageDraw.Draw(scratch)

    best_size = min_size
    best_lines = _wrap_text(text, ImageFont.truetype(font_path, min_size), rect_w, draw)

    lo, hi = min_size, max_size
    while lo <= hi:
        mid = (lo + hi) // 2
        font = ImageFont.truetype(font_path, mid)
        lines = _wrap_text(text, font, rect_w, draw)
        line_height = font.getbbox("Ag")[3] - font.getbbox("Ag")[1]
        line_spacing = int(line_height * 1.25)
        total_height = line_spacing * len(lines)
        widths_ok = all(draw.textlength(line, font=font) <= rect_w for line in lines)
        if total_height <= rect_h and widths_ok:
            best_size, best_lines = mid, lines
            lo = mid + 1
        else:
            hi = mid - 1

    return best_size, best_lines


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("empty_balloon_image")
    ap.add_argument("--margin", type=int, default=DEFAULT_MARGIN)
    args = ap.parse_args()

    img = Image.open(args.empty_balloon_image)
    x, y, w, h = find_interior_rect(img, args.margin)
    print(f"Interior rect: x={x} y={y} w={w} h={h} (image size {img.size})")


if __name__ == "__main__":
    main()
