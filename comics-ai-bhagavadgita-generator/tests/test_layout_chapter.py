import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from PIL import Image

from layout_chapter import (
    CANVAS_WIDTH,
    CONTENT_WIDTH,
    ChapterTooTallError,
    layout_chapter,
    layout_chapter_content,
)
from models import CanonicalChapter


@dataclass(frozen=True)
class FakeImage:
    """Duck-typed stand-in for a PIL Image exposing only what layout math needs -- lets the
    safety-guard test declare a multi-billion-pixel-tall "image" without allocating real memory."""

    width: int
    height: int


def _real_image(height: int) -> Image.Image:
    return Image.new("RGBA", (CONTENT_WIDTH, height))


def test_layout_stacks_assets_with_correct_gaps_and_positions():
    content = [("art", _real_image(100)), ("balloon", _real_image(200)), ("balloon", _real_image(150))]
    assets, total_height = layout_chapter_content(content)
    assert [a.y for a in assets] == [72, 204, 436]
    assert all(a.x == 72 for a in assets)
    assert total_height == 658  # 436 + 150 + 72 (bottom safe area)


def test_layout_with_no_content_images_gives_minimal_height():
    assets, total_height = layout_chapter_content([])
    assert assets == ()
    assert total_height == 144  # SAFE_AREA * 2


def test_layout_rejects_an_asset_not_at_content_width():
    with pytest.raises(ValueError, match="CONTENT_WIDTH"):
        layout_chapter_content([("art", _real_image(100).resize((500, 100)))])


def test_safety_guard_triggers_on_an_artificially_huge_chapter():
    huge = FakeImage(width=CONTENT_WIDTH, height=3_000_000_000)
    with pytest.raises(ChapterTooTallError):
        layout_chapter_content([("art", huge)])


def test_layout_chapter_assembles_background_first_then_content_in_order():
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=())
    content = [("art", _real_image(100)), ("balloon", _real_image(50))]
    _, total_height = layout_chapter_content(content)
    background = Image.new("RGBA", (CANVAS_WIDTH, total_height))
    layout = layout_chapter(chapter, content, background)
    assert layout.width == CANVAS_WIDTH
    assert layout.height == total_height
    assert [a.kind for a in layout.assets] == ["background", "art", "balloon"]
    assert layout.assets[0].x == 0 and layout.assets[0].y == 0


def test_layout_chapter_rejects_a_background_of_the_wrong_size():
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=())
    content = [("art", _real_image(100))]
    wrong_background = Image.new("RGBA", (CANVAS_WIDTH, 1))
    with pytest.raises(ValueError, match="background_image size"):
        layout_chapter(chapter, content, wrong_background)


def test_real_chapter_one_layout_produces_a_plausible_total_height():
    """Real integration test: renders chapter 1's actual title card and all 37 real verse cards
    via Playwright, then lays them out, asserting the resulting structure/height is real and
    plausible (not a fixture stand-in)."""
    import render_cards
    from load_dataset import DATASET_DIR, load_book_one

    chapters = load_book_one(dataset_dir=DATASET_DIR, book_id=1)
    chapter_one = next(c for c in chapters if c.order == 1)
    assert len(chapter_one.slokas) == 37  # real, previously-verified fact about this chapter

    try:
        content = [("art", render_cards.render_title_card(chapter_one.order, chapter_one.title, book_id=1))]
        for sloka in chapter_one.slokas:
            content.append(("balloon", render_cards.render_verse_card(sloka, chapter_one.order, book_id=1)))

        _, total_height = layout_chapter_content(content)
        background = render_cards.render_chapter_background(
            render_cards.theme_for_chapter(chapter_one.order), CANVAS_WIDTH, total_height
        )
        layout = layout_chapter(chapter_one, content, background)
    finally:
        render_cards.shutdown_browser()

    assert len(layout.assets) == 1 + 1 + 37  # background + title + 37 real verse cards
    assert layout.assets[1].kind == "art"  # title card
    assert all(a.kind == "balloon" for a in layout.assets[2:])
    # plausible bounds: each card is at least the min font's line height and at most a generous cap
    assert 5_000 < layout.height < 200_000
