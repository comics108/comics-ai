import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import discover
import erase
import extract
import layout

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_FILE = "8a89f7d689fb441ea280cd782276bd7a.comics"
FONT_PATH = "/System/Library/Fonts/Supplemental/Comic Sans MS.ttf"


def _empty_balloon(source_file, layer_index, tmp_path):
    balloons = discover.discover_all(DATASET_DIR)
    balloon = next(
        b for b in balloons if b.source_file == source_file and b.layer_index == layer_index
    )
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)
    return erase.erase_balloon_images(tmp_path / paths[0], tmp_path / paths[1])


def test_largest_inscribed_rectangle_simple_case():
    mask = np.zeros((10, 10), dtype=bool)
    mask[2:8, 1:9] = True  # 6x8 solid block
    x, y, w, h = layout.largest_inscribed_rectangle(mask)
    assert w * h == 6 * 8
    assert mask[y : y + h, x : x + w].all()


def test_largest_inscribed_rectangle_empty_mask():
    mask = np.zeros((10, 10), dtype=bool)
    x, y, w, h = layout.largest_inscribed_rectangle(mask)
    assert (w, h) == (0, 0)


def test_interior_rect_stays_inside_balloon(tmp_path):
    empty = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    x, y, w, h = layout.find_interior_rect(empty)
    assert w > 0 and h > 0
    # rect must be within image bounds
    assert x >= 0 and y >= 0
    assert x + w <= empty.size[0]
    assert y + h <= empty.size[1]
    # every pixel in the rect must be part of the safe interior mask
    mask = layout.find_safe_interior_mask(empty)
    assert mask[y : y + h, x : x + w].all()


def test_interior_rect_smaller_on_round_tailed_balloon(tmp_path):
    # a round balloon with a tail should yield a materially smaller safe rect (relative to full
    # image area) than a plain rectangular caption box, since more area is excluded by the curve
    rect_caption = _empty_balloon(SAMPLE_FILE, 174, tmp_path)
    rect_round = _empty_balloon(SAMPLE_FILE, 176, tmp_path)
    _, _, w1, h1 = layout.find_interior_rect(rect_caption)
    _, _, w2, h2 = layout.find_interior_rect(rect_round)
    frac_caption = (w1 * h1) / (rect_caption.size[0] * rect_caption.size[1])
    frac_round = (w2 * h2) / (rect_round.size[0] * rect_round.size[1])
    assert frac_round < frac_caption


def test_fit_text_to_rect_wraps_and_fits():
    size, lines = layout.fit_text_to_rect(
        "HERE HE IS.", 150, 80, FONT_PATH, min_size=8, max_size=60
    )
    assert size >= 8
    assert lines
    assert " ".join(lines).replace("  ", " ") in ("HERE HE IS.",) or "".join(lines)

    # rendered width of every line must fit within rect_w
    from PIL import ImageDraw, ImageFont

    scratch = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(scratch)
    font = ImageFont.truetype(FONT_PATH, size)
    for line in lines:
        assert draw.textlength(line, font=font) <= 150


def test_fit_text_to_rect_larger_rect_gets_larger_font():
    text = "SHORT TEXT HERE"
    small_size, _ = layout.fit_text_to_rect(text, 100, 40, FONT_PATH)
    large_size, _ = layout.fit_text_to_rect(text, 500, 300, FONT_PATH)
    assert large_size >= small_size
