import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_lettering_render_variants import OCR_LANGUAGES, VARIANTS


def test_render_variant_space_is_bounded_and_has_independent_script_models():
    assert len(VARIANTS) == 73
    assert {weight for _, weight, _, _, _ in VARIANTS} == {400, 500, 600, 700, 800, 900}
    assert {size for _, _, size, _, _ in VARIANTS} == {44, 48, 52, 56, 60, 64}
    assert {align for _, _, _, _, align in VARIANTS} == {"center", "left"}
    assert "script/Latin" in OCR_LANGUAGES["en"]
    assert "script/Devanagari" in OCR_LANGUAGES["sa"]
