import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_lettering_ocr import CONFIGS


def test_ocr_matrix_has_script_models_without_authoritative_dictionary():
    assert ("script/Latin", 6) in CONFIGS["en"]
    assert ("script/Devanagari", 6) in CONFIGS["sa"]
    assert all(psm in {6, 11} for configs in CONFIGS.values() for _, psm in configs)
