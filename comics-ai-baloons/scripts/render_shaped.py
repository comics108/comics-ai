#!/usr/bin/env python3
"""Stage 6.4: render text into an empty balloon for complex-script languages, via headless Chromium.

Covers everything in languages.COMPLEX_SCRIPT_LANGUAGES: th, zh, ko, kn, ja, hi, ta, mr, bn, ne,
he, ar. Plain PIL word-wrap (render_latin.py's approach) assumes space-delimited words and
left-to-right layout, neither of which holds here (CJK/Thai don't reliably space-delimit; Hebrew/
Arabic are RTL and Arabic additionally needs contextual glyph joining). Rather than reimplement
per-script line-breaking and shaping rules, this renders an HTML snippet in headless Chromium and
lets the browser's own (ICU-backed) text engine handle wrapping/shaping/direction, then screenshots
just the text and composites it onto the balloon. `find_interior_rect` (layout.py) is reused
unchanged -- interior-box detection is script-agnostic.

Font size is fit by binary search in-browser (grow/shrink until the text div stops overflowing its
box), since we don't pre-wrap ourselves here -- the browser reflows live as font-size changes.
"""

from __future__ import annotations

import argparse
import html
import io
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

import languages
import layout

FONTS_DIR = Path(__file__).resolve().parents[1] / "fonts" / "Noto"

FONT_FILES: dict[str, str] = {
    "th": "NotoSansThai[wdth,wght].ttf",
    "zh": "NotoSansSC[wght].ttf",
    "ko": "NotoSansKR[wght].ttf",
    "kn": "NotoSansKannada[wdth,wght].ttf",
    "ja": "NotoSansJP[wght].ttf",
    "hi": "NotoSansDevanagari[wdth,wght].ttf",
    "mr": "NotoSansDevanagari[wdth,wght].ttf",
    "ne": "NotoSansDevanagari[wdth,wght].ttf",
    "ta": "NotoSansTamil[wdth,wght].ttf",
    "bn": "NotoSansBengali[wdth,wght].ttf",
    "he": "NotoSansHebrew[wdth,wght].ttf",
    "ar": "NotoSansArabic[wdth,wght].ttf",
}

MIN_FONT_SIZE = 10
MAX_FONT_SIZE = 96

# Lazily-started, process-wide browser singleton -- launching Chromium per render (as an earlier
# version of this module did) is expensive enough to dominate a full-dataset batch run. Reused
# across calls; call shutdown_browser() when done (package.py's CLI does this at exit).
_playwright_ctx = None
_browser = None


def _get_browser():
    global _playwright_ctx, _browser
    if _browser is None:
        _playwright_ctx = sync_playwright().start()
        _browser = _playwright_ctx.chromium.launch()
    return _browser


def shutdown_browser() -> None:
    global _playwright_ctx, _browser
    if _browser is not None:
        _browser.close()
        _browser = None
    if _playwright_ctx is not None:
        _playwright_ctx.stop()
        _playwright_ctx = None


def _font_path_for(lang_code: str) -> Path:
    if lang_code not in FONT_FILES:
        raise ValueError(f"no Noto font mapped for language {lang_code!r}")
    return FONTS_DIR / FONT_FILES[lang_code]


def _build_html(text: str, width: int, height: int, font_url: str, direction: str) -> str:
    escaped = html.escape(text)
    return f"""
<!doctype html>
<html lang="">
<head>
<style>
  @font-face {{
    font-family: "BalloonFont";
    src: url("{font_url}");
  }}
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #box {{
    width: {width}px;
    height: {height}px;
    box-sizing: border-box;
    display: flex;
    align-items: center;
    justify-content: center;
    overflow: hidden;
  }}
  #text {{
    font-family: "BalloonFont", sans-serif;
    direction: {direction};
    text-align: center;
    color: #000;
    width: 100%;
  }}
</style>
</head>
<body>
  <div id="box"><div id="text">{escaped}</div></div>
</body>
</html>
"""


def render_text_on_balloon(
    empty_balloon_img: Image.Image,
    text: str,
    lang_code: str,
    margin: int = layout.DEFAULT_MARGIN,
) -> tuple[Image.Image, bool]:
    """Returns (rendered_image, fit_ok), matching render_latin.render_text_on_balloon's contract."""
    x, y, w, h = layout.find_interior_rect(empty_balloon_img, margin)
    result = empty_balloon_img.convert("RGBA")
    if w <= 0 or h <= 0:
        return result, False

    direction = "rtl" if lang_code in languages.RTL_LANGUAGES else "ltr"
    font_path = _font_path_for(lang_code)
    font_url = font_path.resolve().as_uri()

    browser = _get_browser()
    page = browser.new_page(viewport={"width": w, "height": h})
    try:
        page.set_content(_build_html(text, w, h, font_url, direction))
        page.wait_for_timeout(50)  # let the @font-face load

        fit_ok = _fit_font_size(page, w, h)
        text_box = page.locator("#text")
        screenshot_bytes = text_box.screenshot(omit_background=True)
    finally:
        page.close()

    text_img = Image.open(io.BytesIO(screenshot_bytes)).convert("RGBA")
    # Center the (possibly smaller, if fit shrank the box) text image within the interior rect.
    paste_x = x + max(0, (w - text_img.width) // 2)
    paste_y = y + max(0, (h - text_img.height) // 2)
    result.alpha_composite(text_img, dest=(paste_x, paste_y))
    return result, fit_ok


def _fit_font_size(page, box_w: int, box_h: int) -> bool:
    """Binary search the largest font-size (px) where #text doesn't overflow #box. Returns
    whether MIN_FONT_SIZE already fits (False means even the smallest size overflows).
    """

    def overflows(size: int) -> bool:
        page.evaluate(
            "(size) => { document.getElementById('text').style.fontSize = size + 'px'; }", size
        )
        return page.evaluate(
            """() => {
                const box = document.getElementById('box');
                const text = document.getElementById('text');
                return text.scrollWidth > box.clientWidth + 1
                    || text.scrollHeight > box.clientHeight + 1;
            }"""
        )

    if overflows(MIN_FONT_SIZE):
        return False

    lo, hi = MIN_FONT_SIZE, MAX_FONT_SIZE
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        if overflows(mid):
            hi = mid - 1
        else:
            best = mid
            lo = mid + 1
    overflows(best)  # leave the DOM at the chosen size for the screenshot
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("empty_balloon_image")
    ap.add_argument("text")
    ap.add_argument("lang_code")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    img = Image.open(args.empty_balloon_image)
    result, fit_ok = render_text_on_balloon(img, args.text, args.lang_code)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.save(out_path)
    print(f"Wrote {out_path} (fit_ok={fit_ok})")
    shutdown_browser()


if __name__ == "__main__":
    main()
