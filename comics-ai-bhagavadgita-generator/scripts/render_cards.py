#!/usr/bin/env python3
"""Task 3.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): render one verse
card per SlokaSource to a transparent PNG via headless Chromium/Playwright, following
`comics-ai-baloons`'s `render_shaped.py` pattern (real, proven precedent in this repo) -- the
browser's own text engine handles Devanagari shaping, Cyrillic, and line wrapping, which a plain
PIL word-wrap cannot do correctly.

Real difference from `render_shaped.py`'s balloon renderer: that renderer has a *fixed* box and
binary-searches font-size down to fit. Per 02-specifications.md ("Font sizes have documented
minimums; cards grow vertically rather than shrinking below the minimum or clipping content."),
verse cards do the opposite: font sizes below are fixed constants (never shrunk), and the card's
*height* grows to fit whatever the browser wraps to at a fixed CARD_WIDTH.

Judgment call (Specifications says the source marker is "book:chapter:sloka derived from IDs"
without pinning down order-vs-id): this renders it as `{book_id}:{chapter_id}:{sloka.id}` -- the
real database identifiers, not the human-facing Chapter.Order/Sloka.Order used in the label --
since "derived from IDs" (plural, capitalized as a field name in the surrounding CSV/dataclass
vocabulary) most naturally means the Id columns, and the label already covers the order-based
reference. Flagged here for Anton to redirect if a different marker shape was intended.
"""

from __future__ import annotations

import colorsys
import html
import io
import random
from dataclasses import dataclass
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

from models import SlokaSource

FONTS_DIR = Path(__file__).resolve().parents[1] / "fonts" / "Noto"
LATIN_CYRILLIC_FONT = FONTS_DIR / "NotoSans-Regular.ttf"
DEVANAGARI_FONT = FONTS_DIR / "NotoSansDevanagari[wdth,wght].ttf"

CARD_WIDTH = 936  # Specifications' "content width" canvas constant
CARD_PADDING = 32

LABEL_FONT_SIZE = 28
SANSKRIT_FONT_SIZE = 34
TRANSCRIPTION_FONT_SIZE = 22
TRANSLATION_FONT_SIZE = 28
SOURCE_MARKER_FONT_SIZE = 16

_INITIAL_VIEWPORT_HEIGHT = 100  # real height is measured after render, then the viewport is grown

# Lazily-started, process-wide browser singleton -- matches render_shaped.py's rationale: launching
# Chromium per card is expensive enough to dominate an 18-chapter/663-card batch run.
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


def source_marker(book_id: int, sloka: SlokaSource) -> str:
    return f"{book_id}:{sloka.chapter_id}:{sloka.id}"


def build_verse_card_html(sloka: SlokaSource, chapter_order: int, book_id: int) -> str:
    """Pure string-building, no browser involved -- kept separate from render_verse_card so
    HTML-escaping/structure can be unit-tested without launching Chromium."""
    label = html.escape(f"{chapter_order}.{sloka.order}")
    sanskrit = html.escape(sloka.sanskrit)
    transcription = html.escape(sloka.transcription)
    translation = html.escape(sloka.translation_ru)
    marker = html.escape(source_marker(book_id, sloka))
    latin_url = LATIN_CYRILLIC_FONT.resolve().as_uri()
    devanagari_url = DEVANAGARI_FONT.resolve().as_uri()

    return f"""
<!doctype html>
<html lang="">
<head>
<style>
  @font-face {{ font-family: "CardLatin"; src: url("{latin_url}"); }}
  @font-face {{ font-family: "CardDevanagari"; src: url("{devanagari_url}"); }}
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #card {{
    width: {CARD_WIDTH}px;
    box-sizing: border-box;
    padding: {CARD_PADDING}px;
    font-family: "CardLatin", "CardDevanagari", sans-serif;
    color: #1a1a1a;
  }}
  .label {{ font-size: {LABEL_FONT_SIZE}px; font-weight: bold; margin-bottom: 12px; }}
  .sanskrit {{
    font-family: "CardDevanagari", "CardLatin", sans-serif;
    font-size: {SANSKRIT_FONT_SIZE}px;
    margin-bottom: 12px;
  }}
  .transcription {{
    font-size: {TRANSCRIPTION_FONT_SIZE}px;
    font-style: italic;
    margin-bottom: 12px;
  }}
  .translation {{ font-size: {TRANSLATION_FONT_SIZE}px; margin-bottom: 16px; }}
  .source {{ font-size: {SOURCE_MARKER_FONT_SIZE}px; color: #666; }}
</style>
</head>
<body>
  <div id="card">
    <div class="label">{label}</div>
    <div class="sanskrit">{sanskrit}</div>
    <div class="transcription">{transcription}</div>
    <div class="translation">{translation}</div>
    <div class="source">{marker}</div>
  </div>
</body>
</html>
"""


def _screenshot_html_element(html_doc: str, element_id: str) -> Image.Image:
    """Shared render path for both verse and title cards: fixed CARD_WIDTH viewport, content
    height measured after layout, then re-screenshot at that exact height -- never shrinks font,
    never clips (per Specifications), and never guesses a height in advance."""
    browser = _get_browser()
    page = browser.new_page(viewport={"width": CARD_WIDTH, "height": _INITIAL_VIEWPORT_HEIGHT})
    try:
        page.set_content(html_doc)
        page.wait_for_timeout(50)  # let the @font-face files load

        content_height = page.evaluate(f"document.getElementById('{element_id}').scrollHeight")
        page.set_viewport_size({"width": CARD_WIDTH, "height": content_height})
        screenshot_bytes = page.locator(f"#{element_id}").screenshot(omit_background=True)
    finally:
        page.close()

    return Image.open(io.BytesIO(screenshot_bytes)).convert("RGBA")


def render_verse_card(sloka: SlokaSource, chapter_order: int, book_id: int) -> Image.Image:
    """Renders one verse card at CARD_WIDTH, height grown to fit content. Returns an RGBA PIL
    image with a transparent background."""
    return _screenshot_html_element(build_verse_card_html(sloka, chapter_order, book_id), "card")


# --- Task 3.2: deterministic visual theme -----------------------------------------------------
#
# Specifications: "The theme derives colors and ornaments from the chapter order using a fixed
# palette and fixed seed. It never needs network access or generated artwork." Implemented here as
# a `random.Random(chapter_order)` seeded purely by the chapter's own order -- the "fixed seed" is
# the chapter order itself, so the same order always reproduces the same theme with no external
# state, matching "never needs network access or generated artwork" literally (no image assets,
# just deterministic color math).

ORNAMENT_COUNT = 6  # small fixed set of ornament variants (index only; no art assets in this task)

TITLE_FONT_SIZE = 56
CHAPTER_NUMBER_FONT_SIZE = 24
TITLE_CARD_MIN_HEIGHT = 200


@dataclass(frozen=True)
class ChapterTheme:
    chapter_order: int
    background_hex: str
    accent_hex: str
    ornament_index: int


def _rgb_to_hex(r: float, g: float, b: float) -> str:
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))


def theme_for_chapter(chapter_order: int) -> ChapterTheme:
    """Deterministic: calling this twice with the same chapter_order always returns an identical
    ChapterTheme (same seed -> same random.Random draw sequence -> same values)."""
    rng = random.Random(chapter_order)
    hue = rng.random()
    accent_rgb = colorsys.hsv_to_rgb(hue, 0.55, 0.75)
    background_rgb = colorsys.hsv_to_rgb(hue, 0.15, 0.97)
    return ChapterTheme(
        chapter_order=chapter_order,
        background_hex=_rgb_to_hex(*background_rgb),
        accent_hex=_rgb_to_hex(*accent_rgb),
        ornament_index=rng.randrange(ORNAMENT_COUNT),
    )


def render_chapter_background(theme: ChapterTheme, width: int, height: int) -> Image.Image:
    """A solid deterministic color field -- no Playwright needed, pure PIL. Per Specifications,
    the background is "a deterministic generated texture/color field," and a flat fill is the
    minimal honest implementation of that (no invented texture/ornament artwork exists yet)."""
    return Image.new("RGBA", (width, height), theme.background_hex)


def build_title_card_html(chapter_order: int, chapter_title: str, book_id: int) -> str:
    theme = theme_for_chapter(chapter_order)
    number_label = html.escape(f"Глава {chapter_order}")
    title = html.escape(chapter_title)
    latin_url = LATIN_CYRILLIC_FONT.resolve().as_uri()

    return f"""
<!doctype html>
<html lang="">
<head>
<style>
  @font-face {{ font-family: "CardLatin"; src: url("{latin_url}"); }}
  html, body {{ margin: 0; padding: 0; background: transparent; }}
  #title-card {{
    width: {CARD_WIDTH}px;
    min-height: {TITLE_CARD_MIN_HEIGHT}px;
    box-sizing: border-box;
    padding: {CARD_PADDING}px;
    font-family: "CardLatin", sans-serif;
    background-color: {theme.background_hex};
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    text-align: center;
  }}
  .chapter-number {{
    font-size: {CHAPTER_NUMBER_FONT_SIZE}px;
    color: {theme.accent_hex};
    text-transform: uppercase;
    letter-spacing: 2px;
    margin-bottom: 12px;
  }}
  .chapter-title {{ font-size: {TITLE_FONT_SIZE}px; font-weight: bold; color: #1a1a1a; }}
</style>
</head>
<body>
  <div id="title-card">
    <div class="chapter-number">{number_label}</div>
    <div class="chapter-title">{title}</div>
  </div>
</body>
</html>
"""


def render_title_card(chapter_order: int, chapter_title: str, book_id: int) -> Image.Image:
    return _screenshot_html_element(
        build_title_card_html(chapter_order, chapter_title, book_id), "title-card"
    )
