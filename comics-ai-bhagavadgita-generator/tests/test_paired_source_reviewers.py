import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import build_paired_source_reviewers


def test_paired_reviewers_use_different_registered_renditions():
    source = inspect.getsource(build_paired_source_reviewers.build)
    assert "aligned_colour_file" in source
    assert 'f"bw-{pair[\'bw_page\']:02}.jpg"' in source
    assert "sam_review(colour" in source
    assert "graph_review(bw" in source
