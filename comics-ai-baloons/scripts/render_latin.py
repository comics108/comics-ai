#!/usr/bin/env python3
"""Stage 6.3: render text into an empty balloon for Latin/Cyrillic-script languages.

Covers: en, ru, uk, es, fr, pt, tr, vi (everything in languages.LANGUAGES except
languages.COMPLEX_SCRIPT_LANGUAGES). Plain PIL word-wrap + draw is sufficient for these --
space-delimited word wrapping and simple left-to-right shaping both hold.

Font: Shantell Sans (OFL-licensed, Google Fonts), vendored at
apps/comics-ai-baloons/fonts/ShantellSans/. Chosen after a side-by-side comparison against the
dataset's actual lettering (flows/sdd-comics-ai-baloons/04-implementation-log.md, Task 6.3):
Comic Sans MS was visually closest to the original but is not freely licensed; Comic Neue (its
usual OFL substitute) has no Cyrillic glyphs at all; Shantell Sans covers both Latin and Cyrillic
and is stylistically very close to the comic's existing rounded-caps lettering.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

import layout

FONT_PATH = str(
    Path(__file__).resolve().parents[1] / "fonts" / "ShantellSans" / "ShantellSans-VariableFont.ttf"
)
TEXT_COLOR = (0, 0, 0)
LINE_SPACING_MULT = 1.25


def render_text_on_balloon(
    empty_balloon_img: Image.Image,
    text: str,
    font_path: str = FONT_PATH,
    margin: int = layout.DEFAULT_MARGIN,
) -> tuple[Image.Image, bool]:
    """Returns (rendered_image, fit_ok). fit_ok is False if even the smallest font size didn't
    fit within the interior rect (Specifications' "text_overflow" edge case) -- the image is
    still returned (best effort at the minimum size) so the caller can decide whether to accept
    an overflowing render or skip it, rather than this function silently guessing.
    """
    x, y, w, h = layout.find_interior_rect(empty_balloon_img, margin)
    if w <= 0 or h <= 0:
        return empty_balloon_img.convert("RGBA"), False

    size, lines = layout.fit_text_to_rect(text, w, h, font_path)

    result = empty_balloon_img.convert("RGBA")
    draw = ImageDraw.Draw(result)
    font = ImageFont.truetype(font_path, size)
    bbox = font.getbbox("Ag")
    line_height = bbox[3] - bbox[1]
    spacing = int(line_height * LINE_SPACING_MULT)
    total_height = spacing * len(lines)
    fit_ok = total_height <= h and all(
        draw.textlength(line, font=font) <= w for line in lines
    )

    start_y = y + max(0, (h - total_height) // 2)
    for i, line in enumerate(lines):
        line_w = draw.textlength(line, font=font)
        line_x = x + max(0, (w - line_w) // 2)
        draw.text((line_x, start_y + i * spacing), line, font=font, fill=TEXT_COLOR)

    return result, fit_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("empty_balloon_image")
    ap.add_argument("text")
    ap.add_argument("--out", required=True)
    ap.add_argument("--font", default=FONT_PATH)
    args = ap.parse_args()

    img = Image.open(args.empty_balloon_image)
    result, fit_ok = render_text_on_balloon(img, args.text, args.font)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"Wrote {out_path} (fit_ok={fit_ok})")


if __name__ == "__main__":
    main()
