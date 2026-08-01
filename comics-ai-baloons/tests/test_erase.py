import sys
from pathlib import Path

import numpy as np
import pytesseract
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import discover
import erase
import extract

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_FILE = "8a89f7d689fb441ea280cd782276bd7a.comics"


def _extract_pair(source_file, layer_index, tmp_path):
    balloons = discover.discover_all(DATASET_DIR)
    balloon = next(
        b for b in balloons if b.source_file == source_file and b.layer_index == layer_index
    )
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)
    return tmp_path / paths[0], tmp_path / paths[1]


def test_erase_removes_ocr_detectable_text(tmp_path):
    en_path, ru_path = _extract_pair(SAMPLE_FILE, 174, tmp_path)
    result = erase.erase_balloon_images(en_path, ru_path)

    flat = Image.new("RGB", result.size, (255, 255, 255))
    flat.paste(result, mask=result.split()[3])
    text = pytesseract.image_to_string(flat, lang="eng", config="--psm 6").strip()
    assert len(text) < 10  # only faint OCR noise, if anything, should survive -- no real words


def test_erase_preserves_outline_shape(tmp_path):
    en_path, ru_path = _extract_pair(SAMPLE_FILE, 174, tmp_path)
    original = Image.open(en_path)
    result = erase.erase_balloon_images(en_path, ru_path)

    assert result.size == original.size
    assert result.mode == "RGBA"
    # alpha (silhouette) must be preserved exactly -- it carries no text information
    assert np.array_equal(
        np.array(result)[:, :, 3], np.array(original.convert("RGBA"))[:, :, 3]
    )
    # a pixel known to be on the border in the original should stay dark in the result
    orig_rgb = np.array(original.convert("RGB"))
    result_rgb = np.array(result.convert("RGB"))
    border_y, border_x = 5, 300
    assert orig_rgb[border_y, border_x].sum() < 100
    assert result_rgb[border_y, border_x].sum() < 100


def test_erase_on_round_tailed_balloon(tmp_path):
    en_path, ru_path = _extract_pair(SAMPLE_FILE, 176, tmp_path)
    result = erase.erase_balloon_images(en_path, ru_path)
    flat = Image.new("RGB", result.size, (255, 255, 255))
    flat.paste(result, mask=result.split()[3])
    text = pytesseract.image_to_string(flat, lang="eng", config="--psm 6").strip()
    assert len(text) < 10


def test_remove_speckles_keeps_large_components():
    rgb = np.full((50, 50, 3), 255, dtype=np.uint8)
    rgb[5:45, 5:45] = (0, 0, 0)  # large solid black square -- should survive
    cleaned = erase.remove_speckles(rgb)
    assert (cleaned == 0).any()


def test_remove_speckles_removes_small_dots():
    rgb = np.full((50, 50, 3), 255, dtype=np.uint8)
    rgb[10:13, 10:13] = (0, 0, 0)  # tiny 3x3 speck
    cleaned = erase.remove_speckles(rgb)
    assert (cleaned == 255).all()


def test_erase_works_without_second_image(tmp_path):
    # Regression: erase must not require a same-sized ru image at all -- 21.7% of balloons have
    # mismatched en/ru dimensions in the real dataset (the artist resized the balloon shape per
    # language to fit translated text length), which broke an earlier two-image-diff approach.
    en_path, _ru_path = _extract_pair(SAMPLE_FILE, 174, tmp_path)
    result = erase.erase_balloon_images(en_path)
    flat = Image.new("RGB", result.size, (255, 255, 255))
    flat.paste(result, mask=result.split()[3])
    text = pytesseract.image_to_string(flat, lang="eng", config="--psm 6").strip()
    assert len(text) < 10


def test_erase_on_mismatched_size_balloon(tmp_path):
    # The specific real balloon that surfaced the size-mismatch bug: en is 416x481, ru is
    # 397x477. Erase must still cleanly remove text and keep the full outline+tail using en
    # alone, not distort anything by trying to align against ru.
    balloons = discover.discover_all(DATASET_DIR)
    balloon = next(
        b
        for b in balloons
        if b.source_file == "25a9a65e8a9f4421b4b8be62b2bf477f.comics" and b.layer_index == 166
    )
    assert balloon.slots[0].width != balloon.slots[1].width  # confirm the mismatch premise holds
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)
    en_path = tmp_path / paths[0]

    result = erase.erase_balloon_images(en_path)
    assert result.size == Image.open(en_path).size

    # outline must survive: alpha-opaque, dark pixel fraction should still be a sane amount
    # (previously this regressed to nearly nothing -- almost the whole outline vanished)
    rgb = np.array(result.convert("RGB"))
    gray = rgb.mean(axis=2)
    dark_fraction = (gray < 128).mean()
    assert dark_fraction > 0.02  # a real outline+tail covers a clearly non-trivial fraction

    flat = Image.new("RGB", result.size, (255, 255, 255))
    flat.paste(result, mask=result.split()[3])
    text = pytesseract.image_to_string(flat, lang="eng", config="--psm 6").strip()
    assert len(text) < 10


def test_identical_images_short_circuit():
    img = Image.new("RGBA", (20, 20), (255, 255, 255, 255))
    result = erase.erase_text(img, img)
    assert result.size == (20, 20)
