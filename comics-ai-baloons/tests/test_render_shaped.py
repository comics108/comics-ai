import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import discover
import erase
import extract
import render_shaped

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_FILE = "8a89f7d689fb441ea280cd782276bd7a.comics"


def _empty_balloon(source_file, layer_index, tmp_path):
    balloons = discover.discover_all(DATASET_DIR)
    balloon = next(
        b for b in balloons if b.source_file == source_file and b.layer_index == layer_index
    )
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)
    return erase.erase_balloon_images(tmp_path / paths[0])


def test_all_mapped_fonts_exist():
    for lang_code in render_shaped.FONT_FILES:
        assert render_shaped._font_path_for(lang_code).exists(), lang_code


def test_unmapped_language_raises():
    import pytest

    with pytest.raises(ValueError):
        render_shaped._font_path_for("en")


def test_render_thai(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    result, fit_ok = render_shaped.render_text_on_balloon(empty, "สงครามจบลง", "th")
    assert fit_ok
    assert result.mode == "RGBA"
    assert result.size == empty.size


def test_render_cjk_no_spaces_still_wraps(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    text = "战争结束了。这是一个关于强大的圣人杜瓦萨·穆尼的故事，他诅咒了强大的国王。"
    result, fit_ok = render_shaped.render_text_on_balloon(empty, text, "zh")
    assert fit_ok
    assert result.size == empty.size


def test_render_devanagari_conjuncts(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    result, fit_ok = render_shaped.render_text_on_balloon(
        empty, "युद्ध समाप्त हो गया है।", "hi"
    )
    assert fit_ok


def test_render_rtl_hebrew(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    result, fit_ok = render_shaped.render_text_on_balloon(
        empty, "בסוף הקרב הלכתי לבד בשדה הקרב הענק.", "he"
    )
    assert fit_ok
    assert result.mode == "RGBA"


def test_render_rtl_arabic(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    result, fit_ok = render_shaped.render_text_on_balloon(
        empty, "في نهاية المعركة سرت وحدي.", "ar"
    )
    assert fit_ok


def test_very_long_text_reports_overflow(tmp_path):
    empty = _empty_balloon("6c690c679511407cb558a0dc347fdebf.comics", 102, tmp_path)
    text = "战" * 300
    _result, fit_ok = render_shaped.render_text_on_balloon(empty, text, "zh")
    assert fit_ok is False
