import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from page_number import extract_page_numbers

LOWCAMERA_DIR = (
    Path(__file__).resolve().parents[4]
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_book_lowcamera"
)


def test_consecutive_check_rejects_a_lone_misread_digit():
    """Real, tested finding (04-implementation-log.md Task 7.1): Tesseract reliably drops the '6'
    from this book's '67' folio number, reading bare '7'. A lone digit must never be reported as
    'found' -- the left+1==right consistency check exists specifically to catch this.
    """
    photo = LOWCAMERA_DIR / "20260731_153604.jpg"
    if not photo.is_file():
        import pytest

        pytest.skip("dataset not present in this checkout")
    result = extract_page_numbers(photo)
    # Real result as of this test: right corner alone yields '7' (missing the '6'), which must NOT
    # be accepted as "found" -- it fails the consistency check (no matching left value = 6).
    assert result.status != "found" or (result.right_page_number == (result.left_page_number or 0) + 1)


def test_extract_page_numbers_never_crashes_on_real_photos():
    if not LOWCAMERA_DIR.is_dir():
        import pytest

        pytest.skip("dataset not present in this checkout")
    photos = sorted(LOWCAMERA_DIR.glob("*.jpg"))[:5]
    for photo in photos:
        result = extract_page_numbers(photo)
        assert result.status in {"found", "partial", "not_found"}
        if result.status == "found":
            assert result.right_page_number == result.left_page_number + 1
