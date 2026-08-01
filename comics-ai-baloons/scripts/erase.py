#!/usr/bin/env python3
"""Stage 6.1: erase -- synthesize an empty balloon by removing its text ink.

Originally implemented as an en/ru pixel-agreement diff (Specifications' first proposal), but
that assumes the two language renders share the same canvas size. Checking the real dataset found
that's false for 179 of 825 balloons (21.7%) -- the artist resized the balloon shape per language
to better fit each translation's length, sometimes substantially (e.g. one balloon is 757x439 in
en and 524x400 in ru). Naively resizing one to match the other before diffing shifts the outline
by enough pixels to make the *entire* border register as "disagreement" and get inpainted away --
found and fixed during Task 6.3 by testing against a real matched balloon, not a cherry-picked one.

Replaced with a single-image approach that doesn't need a second language at all: the balloon's
drawn outline+tail is one large connected component of dark ink spanning most of the image; every
individual text glyph is a much smaller, separate component. This is the same discriminator
already validated in ocr.py's strip_outline_component (used there to *exclude* the outline and
isolate text for OCR) -- here it's inverted to *keep* the outline and erase everything else. This
also means "erase" now works as a true standalone capability on a single balloon render, matching
the Requirements acceptance criterion, rather than only as an en/ru data-prep trick.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from imaging import flatten_to_white

OUTLINE_SPAN_FRACTION = 0.85  # component spanning >= this fraction of width/height = outline/tail
SPECKLE_MAX_AREA = 60  # px; small dark blobs below this, away from the outline, are noise


def _connected_dark_components(rgb: np.ndarray) -> tuple[int, np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    dark = (gray < 128).astype(np.uint8) * 255
    n, labels, stats, _centroids = cv2.connectedComponentsWithStats(dark, connectivity=8)
    return n, labels, stats


def erase_text_single_image(img: Image.Image) -> Image.Image:
    """Returns an RGBA image with text ink removed, outline/tail ink kept, alpha unchanged."""
    flat = flatten_to_white(img)
    rgb = np.array(flat)
    h, w = rgb.shape[:2]

    n, labels, stats = _connected_dark_components(rgb)
    outline_labels = {
        i
        for i in range(1, n)
        if stats[i][2] > OUTLINE_SPAN_FRACTION * w or stats[i][3] > OUTLINE_SPAN_FRACTION * h
    }

    cleaned = rgb.copy()
    if outline_labels:
        # Erase every component that ISN'T part of the outline/tail.
        text_mask = np.isin(labels, list(outline_labels), invert=True) & (labels != 0)
        kernel = np.ones((3, 3), np.uint8)
        text_mask = cv2.dilate(text_mask.astype(np.uint8) * 255, kernel, iterations=1)
        cleaned[text_mask > 0] = (255, 255, 255)
    # else: no component spans enough of the image to be recognized as an outline (shouldn't
    # happen for a real balloon) -- leave `cleaned` as the untouched flattened image rather than
    # guess, so a caller can detect this via comparing output to input if needed.

    alpha = (
        np.array(img.convert("RGBA"))[:, :, 3]
        if img.mode == "RGBA"
        else np.full((h, w), 255, dtype=np.uint8)
    )
    rgba = np.dstack([cleaned, alpha]).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA")


def remove_speckles(rgb: np.ndarray) -> np.ndarray:
    """Paint over small residual dark specks (e.g. antialiasing crumbs left after erase)."""
    n, labels, stats = _connected_dark_components(rgb)
    h, w = rgb.shape[:2]
    remove_mask = np.zeros((h, w), dtype=np.uint8)
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw > OUTLINE_SPAN_FRACTION * w or ch > OUTLINE_SPAN_FRACTION * h:
            continue  # outline/tail -- keep
        if area > SPECKLE_MAX_AREA:
            continue  # large enough to plausibly be real content -- keep, don't guess
        remove_mask[labels == i] = 255

    if not remove_mask.any():
        return rgb

    kernel = np.ones((3, 3), np.uint8)
    remove_mask = cv2.dilate(remove_mask, kernel, iterations=2)
    cleaned = rgb.copy()
    cleaned[remove_mask > 0] = (255, 255, 255)
    return cleaned


def erase_text(en_img: Image.Image, ru_img: Image.Image | None = None) -> Image.Image:
    """Erase text from `en_img`. `ru_img` is accepted for API/CLI compatibility but unused --
    kept as a parameter (rather than removed outright) since the pipeline still has an en/ru pair
    on hand at this stage and a future revision may want it again as a cross-check; today it adds
    nothing the single-image approach doesn't already handle more robustly.
    """
    return erase_text_single_image(en_img)


def erase_balloon_images(en_path: Path, ru_path: Path | None = None) -> Image.Image:
    return erase_text_single_image(Image.open(en_path))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("balloon_image")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    result = erase_text_single_image(Image.open(args.balloon_image))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
