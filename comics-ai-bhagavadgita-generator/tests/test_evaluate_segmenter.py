import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_segmenter import border_background_matting, mask_metrics


def test_mask_metrics_are_exact_for_identical_non_rectangular_masks():
    target = np.zeros((12, 12), dtype=bool)
    target[2:10, 4:8] = True
    target[5:8, 2:10] = True
    result = mask_metrics(target.copy(), target)
    assert result.iou == 1.0
    assert result.boundary_f1 == 1.0


def test_bbox_fill_exposes_shape_error():
    target = np.zeros((10, 10), dtype=bool)
    target[2:8, 3:7] = True
    result = mask_metrics(np.ones_like(target), target)
    assert result.iou == .24
    assert result.boundary_f1 < .7


def test_boundary_tolerance_can_match_documented_model_grid_scale():
    target = np.zeros((100, 100), dtype=bool)
    target[20:80, 20:80] = True
    shifted = np.zeros_like(target)
    shifted[25:85, 25:85] = True
    assert mask_metrics(shifted, target, boundary_radius=6).boundary_f1 > mask_metrics(
        shifted, target, boundary_radius=2
    ).boundary_f1


def test_border_matting_detects_dark_nonpaper_without_gold_mask_input():
    image = Image.new("RGB", (40, 40), "white")
    for x in range(10, 30):
        for y in range(8, 34):
            image.putpixel((x, y), (20, 30, 40))
    prediction = border_background_matting(image)
    assert prediction[20, 20]
    assert not prediction[0, 0]
    assert .25 < prediction.mean() < .5
