#!/usr/bin/env python3
"""Task 7.1 (flows/sdd-comics-ai-positioning/03-plan.md): extract printed page numbers from
comics_book_lowcamera photos. New capability -- nothing in comics-multimodal does this today
(confirmed in Specifications' correction: Checkpoint A found these numbers by manual visual
inspection, not via any script).

Real investigation before writing this (Read tool, not assumed): viewed 3 real photos.
20260731_153604.jpg is a clean, straight two-page interior spread with small folio numbers at the
outer-bottom corner of each page ("66" bottom-left, "67" bottom-right) -- confirms Checkpoint A's
finding and gives a real crop target. But photo framing varies significantly across the 80-photo
set: 20260731_153236.jpg ("AMBA'S CURSE" title card) is physically rotated ~90 degrees in its raw
pixel data (the photographer held the phone sideways -- EXIF orientation tag alone doesn't capture
this, it reads 1/"normal" for that file same as the straight shot); 20260731_153252.jpg is front-
matter (an essay/credits page) with no folio number at all. A fixed bottom-corner crop assumes the
common straight-landscape-spread case; this module targets that case and returns None (not a wrong
guess) for anything else -- consistent with this whole project's skip+log-don't-guess discipline.

**Real bug found and fixed before this module worked at all**: a first version cropped a fixed
fraction of the *whole photo frame*'s bottom corners. Failed even on the known-good 20260731_153604
example -- saved and viewed the actual crop (Read tool) and found it was mostly photographed table
surface below the book, not the page itself; photos vary too much in how much background surrounds
the book for a frame-relative crop to land on the folio number. Fixed by reusing
`comics-multimodal/scripts/detect_panels.py`'s already-working `detect_pages` (page-boundary
detection, built and verified in that flow) to find the real page box(es) first, then cropping
corners *relative to each detected page's own bounding box*, not the raw photo frame.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image

import positioning_bridge as pb

if str(pb.MULTIMODAL_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(pb.MULTIMODAL_SCRIPTS_DIR))
from detect_panels import detect_pages  # noqa: E402  (comics-multimodal's own page-boundary detector, reused)

CORNER_HEIGHT_FRAC = 0.06  # bottom slice of the *detected page box*, not the whole photo
CORNER_WIDTH_FRAC = 0.18  # left/right slice width, from each outer edge of the page box
DIGIT_RE = re.compile(r"\d{1,3}")


@dataclass
class PageNumberResult:
    photo_file: str
    left_page_number: int | None
    right_page_number: int | None
    status: str  # "found" | "partial" | "not_found"
    reason: str


def _ocr_digits(crop: Image.Image) -> int | None:
    text = pytesseract.image_to_string(
        crop, config="--psm 7 -c tessedit_char_whitelist=0123456789"
    )
    match = DIGIT_RE.search(text)
    if not match:
        return None
    value = int(match.group())
    return value if 1 <= value <= 999 else None  # a real book, not a garbage OCR artifact


def _bottom_corner_crop(img: Image.Image, box: tuple[int, int, int, int], side: str) -> Image.Image:
    x0, y0, x1, y1 = box
    page_w, page_h = x1 - x0, y1 - y0
    corner_h = int(page_h * CORNER_HEIGHT_FRAC)
    corner_w = int(page_w * CORNER_WIDTH_FRAC)
    top = y1 - corner_h
    if side == "left":
        return img.crop((x0, top, x0 + corner_w, y1))
    return img.crop((x1 - corner_w, top, x1, y1))


def extract_page_numbers(image_path: Path) -> PageNumberResult:
    pil_img = Image.open(image_path).convert("L")  # grayscale -- sharper OCR contrast
    rgb = np.array(Image.open(image_path).convert("RGB"))
    bgr = rgb[:, :, ::-1]  # detect_pages expects a cv2-style BGR array

    pages = detect_pages(bgr)
    if not pages:
        return PageNumberResult(image_path.name, None, None, "not_found", "no page box detected")

    left_box = pages[0].bbox
    right_box = pages[-1].bbox  # same box as left_box if only one page detected

    left_num = _ocr_digits(_bottom_corner_crop(pil_img, left_box, "left"))
    right_num = _ocr_digits(_bottom_corner_crop(pil_img, right_box, "right"))

    # Real, tested finding (flows/sdd-comics-ai-positioning/04-implementation-log.md, Task 7.1):
    # Tesseract frequently mis-drops a digit from this book's small, stylized folio numbers (e.g.
    # reads "67" as bare "7") rather than failing cleanly -- a single OCR'd digit is NOT trustworthy
    # on its own. A left/right two-page spread's numbers are always consecutive integers; this is a
    # real, strong, book-specific validation check, not an arbitrary heuristic -- only accept a
    # "found" result when it holds, so a half-misread number is rejected rather than silently kept.
    if left_num is not None and right_num is not None and right_num == left_num + 1:
        status, reason = "found", ""
    elif left_num is not None or right_num is not None:
        status, reason = (
            "partial",
            "a digit sequence was found but failed the left+1==right consistency check "
            "(or only one side had any result) -- not trusted as a real page number",
        )
    else:
        status, reason = "not_found", "no plausible page number in either bottom corner"

    return PageNumberResult(
        photo_file=image_path.name,
        left_page_number=left_num,
        right_page_number=right_num,
        status=status,
        reason=reason,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("photos", nargs="+", type=Path)
    args = parser.parse_args()
    for photo in args.photos:
        result = extract_page_numbers(photo)
        print(f"{result.photo_file}: left={result.left_page_number} right={result.right_page_number} "
              f"status={result.status} ({result.reason})")


if __name__ == "__main__":
    main()
