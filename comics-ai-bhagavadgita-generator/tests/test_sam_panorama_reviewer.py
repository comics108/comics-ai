import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from sam_panorama_reviewer import window_starts


def test_windows_cover_panorama_end_with_overlap():
    starts = window_starts(8192, 2048, 384)
    assert starts[0] == 0
    assert starts[-1] == 8192 - 2048
    assert all(right - left <= 2048 - 384 for left, right in zip(starts, starts[1:]))


def test_invalid_overlap_is_rejected():
    with pytest.raises(ValueError):
        window_starts(100, 50, 50)


def test_dense_configuration_is_explicit_in_source():
    source = Path(__file__).resolve().parent.parent / "scripts/sam_panorama_reviewer.py"
    text = source.read_text()
    assert "points_per_side=args.points_per_side" in text
    assert "crop_n_layers=args.crop_n_layers" in text
