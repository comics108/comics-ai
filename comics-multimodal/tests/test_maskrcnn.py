import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from segmenter_models.maskrcnn import NUM_DETECTION_CLASSES, to_detection_target  # noqa: E402


def test_labels_are_shifted_by_one_to_reserve_class_zero():
    target = {
        "boxes": torch.tensor([[0.0, 0.0, 10.0, 10.0]]),
        "labels": torch.tensor([0]),  # our "art" = 0
    }
    det = to_detection_target(target, (20, 20))
    assert det["labels"].tolist() == [1]  # shifted: torchvision class 1, not reserved 0


def test_num_detection_classes_is_kind_count_plus_one():
    assert NUM_DETECTION_CLASSES == 4 + 1


def test_box_shaped_mask_matches_box_region():
    target = {
        "boxes": torch.tensor([[2.0, 3.0, 6.0, 7.0]]),
        "labels": torch.tensor([2]),
    }
    det = to_detection_target(target, (10, 10))
    mask = det["masks"][0]
    assert mask.shape == (10, 10)
    assert mask[3:7, 2:6].all()
    assert mask[0, 0] == 0
    assert mask[9, 9] == 0


def test_degenerate_boxes_are_dropped():
    target = {
        "boxes": torch.tensor([[5.0, 5.0, 5.0, 5.0], [1.0, 1.0, 4.0, 4.0]]),  # first is zero-area
        "labels": torch.tensor([1, 2]),
    }
    det = to_detection_target(target, (10, 10))
    assert len(det["boxes"]) == 1
    assert det["labels"].tolist() == [3]  # only the valid box's label (2+1), shifted


def test_out_of_bounds_box_is_clipped_to_crop():
    target = {
        "boxes": torch.tensor([[-5.0, -5.0, 100.0, 100.0]]),
        "labels": torch.tensor([1]),
    }
    det = to_detection_target(target, (10, 10))
    assert det["boxes"][0].tolist() == [0.0, 0.0, 10.0, 10.0]
    assert det["masks"][0].all()


def test_empty_boxes_produce_empty_but_correctly_shaped_tensors():
    target = {"boxes": torch.zeros((0, 4)), "labels": torch.zeros((0,), dtype=torch.int64)}
    det = to_detection_target(target, (10, 10))
    assert det["boxes"].shape == (0, 4)
    assert det["labels"].shape == (0,)
    assert det["masks"].shape == (0, 10, 10)
    assert det["area"].shape == (0,)
