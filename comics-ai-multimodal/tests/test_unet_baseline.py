import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from segmenter_models.unet_baseline import UNetBaseline, rasterize_label_map  # noqa: E402


def test_unet_forward_shape():
    model = UNetBaseline(num_classes=4)
    x = torch.randn(2, 3, 64, 64)  # divisible by 4
    out = model(x)
    assert out.shape == (2, 4, 64, 64)


def test_unet_rejects_odd_sizes_gracefully_or_works_when_divisible_by_4():
    model = UNetBaseline(num_classes=4)
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    assert out.shape == (1, 4, 32, 32)


def test_rasterize_label_map_paints_background_then_overwrites_in_order():
    boxes = torch.tensor([[0, 0, 10, 10], [2, 2, 6, 6]], dtype=torch.float32)
    labels = torch.tensor([1, 2], dtype=torch.int64)  # background=1 painted first, character=2 over it
    label_map = rasterize_label_map(boxes, labels, size=(10, 10), background_label=0)

    assert label_map.shape == (10, 10)
    assert label_map[0, 0].item() == 1  # only background box covers this corner
    assert label_map[4, 4].item() == 2  # inside both boxes -- later (character) wins
    assert label_map[9, 9].item() == 1  # only background box covers this corner (background box is 0..10 exclusive-ish)


def test_rasterize_label_map_defaults_to_background_label_outside_all_boxes():
    boxes = torch.tensor([[2, 2, 4, 4]], dtype=torch.float32)
    labels = torch.tensor([3], dtype=torch.int64)
    label_map = rasterize_label_map(boxes, labels, size=(8, 8), background_label=0)
    assert label_map[0, 0].item() == 0
    assert label_map[3, 3].item() == 3


def test_rasterize_label_map_clips_out_of_bounds_boxes():
    boxes = torch.tensor([[-5, -5, 100, 100]], dtype=torch.float32)
    labels = torch.tensor([1], dtype=torch.int64)
    label_map = rasterize_label_map(boxes, labels, size=(10, 10))
    assert label_map.shape == (10, 10)
    assert (label_map == 1).all()
