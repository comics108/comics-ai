import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_cross_rendition_boundary_consensus


def test_cross_rendition_gate_is_strict_and_requires_context_qa():
    source = inspect.getsource(build_cross_rendition_boundary_consensus.build)
    assert "min_boundary_support: float = .8" in source
    assert "width_fraction >= .08" in source
    assert "height_fraction >= .15" in source
    assert '"proposed_requires_context_qa"' in source
    assert "not border_truncated" in source
    assert "page_width * .02" in source
    assert "page_height * .02" in source
