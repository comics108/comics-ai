import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from PIL import Image

from import_psd import import_psd_panel, resize_to_content_width

REAL_PSD_DIR = (
    Path(__file__).resolve().parents[4] / "dataset" / "bhagavadgita" / "vaishnav" / "drawing"
)


def test_resize_to_content_width_preserves_aspect_ratio():
    img = Image.new("RGBA", (200, 100), "#ff0000")
    resized = resize_to_content_width(img, content_width=936)
    assert resized.width == 936
    assert resized.height == 468  # 200:100 == 936:468


def test_resize_to_content_width_is_a_noop_when_already_correct_width():
    img = Image.new("RGBA", (936, 500), "#00ff00")
    resized = resize_to_content_width(img, content_width=936)
    assert resized is img


def test_missing_psd_tools_package_degrades_to_a_warning_not_an_exception(monkeypatch):
    monkeypatch.setitem(sys.modules, "psd_tools", None)  # forces ImportError on import
    result = import_psd_panel(REAL_PSD_DIR / "5_1.psd", content_width=936)
    assert result.image is None
    assert result.warning is not None
    assert "5_1.psd" in result.warning


def test_nonexistent_file_degrades_to_a_warning_not_an_exception():
    result = import_psd_panel(Path("/no/such/file.psd"), content_width=936)
    assert result.image is None
    assert result.warning is not None


def test_real_psd_composite_produces_a_correctly_resized_rgba_image():
    """Real integration test against one of the three actual chapter-5 PSD files (the smallest,
    5_1.psd, to keep automated-suite runtime/memory bounded -- all three were manually verified
    to composite successfully this session; see 04-implementation-log.md)."""
    result = import_psd_panel(REAL_PSD_DIR / "5_1.psd", content_width=936)
    assert result.warning is None
    assert result.image is not None
    assert result.image.mode == "RGBA"
    assert result.image.width == 936
    # real source is 9449x7087 -> aspect ratio preserved at content width
    assert result.image.height == round(7087 * 936 / 9449)
