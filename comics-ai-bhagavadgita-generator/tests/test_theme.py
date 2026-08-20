import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from render_cards import (
    build_title_card_html,
    render_chapter_background,
    render_title_card,
    shutdown_browser,
    theme_for_chapter,
)


def test_theme_for_chapter_is_deterministic_across_repeated_calls():
    theme_a = theme_for_chapter(5)
    theme_b = theme_for_chapter(5)
    assert theme_a == theme_b


def test_theme_for_chapter_differs_across_chapter_orders():
    themes = {theme_for_chapter(order) for order in range(1, 19)}
    # 18 distinct real chapter orders should not collide onto identical themes.
    assert len(themes) == 18


def test_render_chapter_background_is_a_solid_deterministic_fill():
    theme = theme_for_chapter(3)
    img_a = render_chapter_background(theme, width=1080, height=500)
    img_b = render_chapter_background(theme, width=1080, height=500)
    assert img_a.tobytes() == img_b.tobytes()
    assert img_a.size == (1080, 500)
    # every pixel equals the theme's background color (fully opaque)
    expected_rgb = tuple(int(theme.background_hex[i : i + 2], 16) for i in (1, 3, 5))
    assert img_a.getpixel((0, 0)) == (*expected_rgb, 255)


def test_title_card_html_contains_the_escaped_real_chapter_title():
    doc = build_title_card_html(1, "<b>Осмотр Армий</b>", book_id=1)
    assert "&lt;b&gt;Осмотр Армий&lt;/b&gt;" in doc
    assert "Глава 1" in doc


def test_real_title_card_rendering_produces_a_non_transparent_png():
    try:
        img = render_title_card(1, "Осмотр Армий", book_id=1)
        assert img.mode == "RGBA"
        assert img.width == 936
        alpha = img.getchannel("A")
        assert alpha.getextrema()[1] > 0
    finally:
        shutdown_browser()
