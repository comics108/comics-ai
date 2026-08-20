import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from models import SlokaSource
from render_cards import build_verse_card_html, render_verse_card, shutdown_browser, source_marker


def _sloka(**overrides) -> SlokaSource:
    fields = dict(
        id=42,
        chapter_id=7,
        order=3,
        name="7.3",
        sanskrit="धृतराष्ट्र उवाच",
        transcription="dhṛtarāṣṭra uvāca",
        translation_ru="Дхритараштра сказал",
        comment_ru="",
        audio_ref="",
        sanskrit_audio_ref="",
    )
    fields.update(overrides)
    return SlokaSource(**fields)


def test_source_marker_is_derived_from_real_ids_not_orders():
    sloka = _sloka(id=42, chapter_id=7)
    assert source_marker(book_id=1, sloka=sloka) == "1:7:42"


def test_html_escapes_all_source_text_fields():
    sloka = _sloka(
        sanskrit="<script>alert(1)</script>",
        transcription="a & b",
        translation_ru='"quoted" <b>bold</b>',
    )
    doc = build_verse_card_html(sloka, chapter_order=7, book_id=1)
    assert "<script>alert(1)</script>" not in doc
    assert "&lt;script&gt;" in doc
    assert "a &amp; b" in doc
    assert "&lt;b&gt;bold&lt;/b&gt;" in doc


def test_html_contains_the_chapter_order_sloka_order_label_not_the_raw_name_field():
    sloka = _sloka(order=3, name="totally different display name")
    doc = build_verse_card_html(sloka, chapter_order=7, book_id=1)
    assert ">7.3<" in doc


def test_real_rendering_produces_a_non_transparent_png_at_the_content_width():
    sloka = _sloka()
    try:
        img = render_verse_card(sloka, chapter_order=7, book_id=1)
        assert img.mode == "RGBA"
        assert img.width == 936
        assert img.height > 0
        alpha = img.getchannel("A")
        assert alpha.getextrema()[1] > 0  # at least some non-transparent pixel exists
    finally:
        shutdown_browser()
