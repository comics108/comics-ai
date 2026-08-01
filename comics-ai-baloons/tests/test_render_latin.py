import sys
from pathlib import Path

import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import discover
import erase
import extract
import render_latin

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


def test_font_file_exists():
    assert Path(render_latin.FONT_PATH).exists()


def test_render_english_text_is_ocr_recoverable(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    text = "A COMPLETELY DIFFERENT SENTENCE ABOUT SOMETHING ELSE."
    result, fit_ok = render_latin.render_text_on_balloon(empty, text)
    assert fit_ok

    flat = Image.new("RGB", result.size, (255, 255, 255))
    flat.paste(result, mask=result.split()[3])
    ocr_text = pytesseract.image_to_string(flat, lang="eng", config="--psm 6").strip().upper()
    assert "DIFFERENT" in ocr_text
    assert "SOMETHING" in ocr_text


def test_render_cyrillic_text(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    text = "СОВЕРШЕННО ДРУГОЕ ПРЕДЛОЖЕНИЕ О ЧЕМ-ТО ИНОМ."
    result, fit_ok = render_latin.render_text_on_balloon(empty, text)
    assert fit_ok
    assert result.mode == "RGBA"

    flat = Image.new("RGB", result.size, (255, 255, 255))
    flat.paste(result, mask=result.split()[3])
    ocr_text = pytesseract.image_to_string(flat, lang="rus", config="--psm 6").strip()
    assert "ДРУГОЕ" in ocr_text or "СОВЕРШЕННО" in ocr_text


def test_very_long_text_reports_overflow(tmp_path):
    # small balloon, absurdly long text -- must not silently claim a good fit
    empty = _empty_balloon("6c690c679511407cb558a0dc347fdebf.comics", 102, tmp_path)
    text = " ".join(["WORD"] * 200)
    _result, fit_ok = render_latin.render_text_on_balloon(empty, text)
    assert fit_ok is False


def test_preserves_balloon_alpha_envelope(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    result, _fit_ok = render_latin.render_text_on_balloon(empty, "SHORT TEXT")
    assert result.size == empty.size
    assert result.mode == "RGBA"
