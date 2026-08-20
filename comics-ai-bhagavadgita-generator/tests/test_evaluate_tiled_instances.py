import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_tiled_instances import bbox_match, connected_bboxes, one_to_one_matches, tiled_instances


def test_connected_components_and_cross_tile_merge_are_not_duplicated():
    mask = np.zeros((30, 50), dtype=bool)
    mask[4:14, 3:13] = True
    mask[16:28, 30:47] = True
    assert len(connected_bboxes(mask, min_area=20)) == 2

    image = Image.new("RGB", (180, 80), "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((65, 15, 125, 65), fill="black")  # crosses the 100 px tile boundary
    raw, merged = tiled_instances(image, tile_width=100, overlap=40, min_area=20)
    assert len(raw) >= 2
    assert len(merged) == 1
    assert bbox_match((10, 10, 20, 20), (5, 5, 25, 25))
    assert len(one_to_one_matches([(0, 0, 100, 100)], [(5, 5, 20, 20), (30, 30, 50, 50)])) == 1
