import inspect
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import graph_panorama_reviewer


def test_graph_reviewer_does_not_consume_sam_or_coco_outputs():
    source = inspect.getsource(graph_panorama_reviewer.review_page)
    assert "felzenszwalb" in source
    assert "checkpoint" not in source
    assert "segment_anything" not in source
