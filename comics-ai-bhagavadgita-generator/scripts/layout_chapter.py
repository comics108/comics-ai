#!/usr/bin/env python3
"""Task 5.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): vertical
continuous-strip layout per 02-specifications.md's exact Canvas constants.

Deliberately separates pure layout math (`layout_chapter_content`, `layout_chapter`) from
rendering (`render_cards.py`, `import_psd.py`): callers render each asset first, then hand the
finished images here for positioning. This keeps the safety-guard/gap-accumulation math testable
against synthetic, arbitrarily large "images" (duck-typed objects with only `.width`/`.height`)
without needing to actually allocate or render gigapixel content.
"""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image

from models import CanonicalChapter

CANVAS_WIDTH = 1080
CONTENT_MARGIN = 72
CONTENT_WIDTH = CANVAS_WIDTH - 2 * CONTENT_MARGIN  # 936, matches render_cards.CARD_WIDTH
LAYER_GAP = 32
SAFE_AREA = 72
MAX_COORDINATE = 2**31 - 1  # 32-bit signed safety guard, per Specifications


class ChapterTooTallError(ValueError):
    """Raised before packaging when the computed height exceeds the safe coordinate range."""


@dataclass(frozen=True)
class LayoutAsset:
    kind: str  # "background" | "art" | "balloon"
    image: Image.Image
    x: int
    y: int


@dataclass(frozen=True)
class ChapterLayout:
    width: int
    height: int
    assets: tuple[LayoutAsset, ...]  # background first, then content assets in layer-sequence order


def layout_chapter_content(content_images: list[tuple[str, object]]) -> tuple[tuple[LayoutAsset, ...], int]:
    """Stacks `content_images` (each `(kind, image)`, image needing only `.width`/`.height`)
    vertically starting at SAFE_AREA, gapped by LAYER_GAP, all left-aligned at CONTENT_MARGIN.
    Returns (positioned assets, total chapter height including top+bottom safe areas)."""
    assets = []
    y = SAFE_AREA
    for kind, image in content_images:
        if image.width != CONTENT_WIDTH:
            raise ValueError(
                f"asset width {image.width} != CONTENT_WIDTH {CONTENT_WIDTH} for kind={kind!r}"
            )
        assets.append(LayoutAsset(kind=kind, image=image, x=CONTENT_MARGIN, y=y))
        y += image.height + LAYER_GAP

    total_height = (y - LAYER_GAP + SAFE_AREA) if content_images else SAFE_AREA * 2
    if total_height > MAX_COORDINATE:
        raise ChapterTooTallError(
            f"computed chapter height {total_height} exceeds the 32-bit coordinate safety guard "
            f"({MAX_COORDINATE})"
        )
    return tuple(assets), total_height


def layout_chapter(
    chapter: CanonicalChapter,
    content_images: list[tuple[str, object]],
    background_image: Image.Image,
) -> ChapterLayout:
    """Assembles the full layer sequence: background (full canvas width, pre-rendered by the
    caller at the height this function reports back via a two-pass call, see pipeline.py) followed
    by the positioned content assets."""
    content_assets, total_height = layout_chapter_content(content_images)
    if background_image.size != (CANVAS_WIDTH, total_height):
        raise ValueError(
            f"background_image size {background_image.size} != expected "
            f"({CANVAS_WIDTH}, {total_height}) -- render it only after computing total_height "
            f"(e.g. via layout_chapter_content first)"
        )
    background_asset = LayoutAsset(kind="background", image=background_image, x=0, y=0)
    return ChapterLayout(width=CANVAS_WIDTH, height=total_height, assets=(background_asset, *content_assets))
