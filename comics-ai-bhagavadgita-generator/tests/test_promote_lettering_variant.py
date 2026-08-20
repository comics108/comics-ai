import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from lettering import render_lettering


def test_render_lettering_records_and_applies_explicit_weight():
    source = inspect.getsource(render_lettering)
    assert "font_weight: int = 400" in source
    assert '"font_weight": font_weight' in source
    assert "font-weight:{font_weight}" in source
