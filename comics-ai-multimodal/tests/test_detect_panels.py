import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baloons_bridge  # noqa: E402
from detect_panels import detect_pages  # noqa: E402

LOWCAMERA_DIR = (
    baloons_bridge.REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_book_lowcamera"
)


def _busy_region(h: int, w: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3), dtype=np.uint8)


def _make_two_page_synthetic() -> np.ndarray:
    # Two busy (high-frequency) regions separated by a wide blank (flat) gutter, on a flat
    # background -- mirrors the real structure: page content vs. low-detail spine/gutter.
    img = np.full((600, 1000, 3), 200, dtype=np.uint8)  # flat gray background
    img[50:550, 50:450] = _busy_region(500, 400, seed=1)  # left page
    img[50:550, 550:950] = _busy_region(500, 400, seed=2)  # right page
    return img


def _make_single_page_synthetic() -> np.ndarray:
    img = np.full((600, 1000, 3), 200, dtype=np.uint8)
    img[50:550, 200:800] = _busy_region(500, 600, seed=3)
    return img


def test_detects_two_pages_on_synthetic_spread():
    img = _make_two_page_synthetic()
    boxes = detect_pages(img)
    assert len(boxes) == 2
    # left-to-right order
    assert boxes[0].bbox[0] < boxes[1].bbox[0]
    # each box should roughly cover its own busy region, not merge into one
    left_w = boxes[0].bbox[2] - boxes[0].bbox[0]
    assert left_w < 500  # must not span the whole image (i.e. not merged across the gutter)


def test_detects_single_page_when_only_one_busy_region():
    img = _make_single_page_synthetic()
    boxes = detect_pages(img)
    assert len(boxes) == 1


def test_empty_flat_image_finds_no_pages():
    img = np.full((400, 400, 3), 128, dtype=np.uint8)
    boxes = detect_pages(img)
    assert boxes == []


def test_real_photo_sample_runs_without_crashing_and_reports_box_count_distribution():
    from collections import Counter

    files = sorted(LOWCAMERA_DIR.glob("*.jpg"))[:20]
    counts = Counter()
    for f in files:
        img = cv2.imread(str(f))
        boxes = detect_pages(img)
        counts[len(boxes)] += 1
        for b in boxes:
            x0, y0, x1, y1 = b.bbox
            assert x1 > x0 and y1 > y0

    print("page-count distribution across 20 real photos:", dict(counts))
    assert sum(counts.values()) == 20
