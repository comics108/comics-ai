import json
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import dataset as ds  # noqa: E402


def _write_fixture_manifest(tmp_path: Path) -> Path:
    from PIL import Image

    clean = Image.new("RGB", (40, 30), (10, 20, 30))
    degraded = Image.new("RGB", (40, 30), (12, 18, 33))
    clean_path = tmp_path / "fixture_clean.png"
    degraded_path = tmp_path / "fixture_degraded.jpg"
    clean.save(clean_path)
    degraded.save(degraded_path, quality=90)

    entries = [
        {
            "episode_file": "fixture.comics",
            "cluster_index": 0,
            "degraded_png": str(degraded_path),
            "clean_png": str(clean_path),
            "bbox": [0, 0, 40, 30],
            "layer_indexes": [3, 4],
            "kinds": ["background", "character"],
            "region_bboxes": [[0, 0, 40, 30], [5, 5, 20, 20]],
        }
    ]
    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return manifest_path


def test_missing_manifest_raises_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        ds.TrainingPairDataset(tmp_path / "does_not_exist.jsonl")


def test_dataset_loads_fixture_entry_with_correct_shapes(tmp_path):
    manifest = _write_fixture_manifest(tmp_path)
    dset = ds.TrainingPairDataset(manifest)
    assert len(dset) == 1

    image_tensor, target = dset[0]
    assert image_tensor.shape == (3, 30, 40)  # (C, H, W)
    assert image_tensor.dtype == torch.float32
    assert image_tensor.max() <= 1.0 and image_tensor.min() >= 0.0

    assert target["crop_size"] == (40, 30)
    assert torch.equal(target["layer_indexes"], torch.tensor([3, 4]))
    assert torch.equal(
        target["labels"],
        torch.tensor([ds.KIND_TO_LABEL["background"], ds.KIND_TO_LABEL["character"]]),
    )
    assert target["boxes"].shape == (2, 4)
    assert torch.equal(target["boxes"][0], torch.tensor([0.0, 0.0, 40.0, 30.0]))


def test_unknown_kind_falls_back_to_art_label(tmp_path):
    from PIL import Image

    clean = Image.new("RGB", (10, 10))
    clean_path = tmp_path / "c.png"
    clean.save(clean_path)
    degraded_path = tmp_path / "d.jpg"
    clean.save(degraded_path)

    manifest_path = tmp_path / "manifest.jsonl"
    entry = {
        "episode_file": "x.comics",
        "cluster_index": 0,
        "degraded_png": str(degraded_path),
        "clean_png": str(clean_path),
        "bbox": [0, 0, 10, 10],
        "layer_indexes": [0],
        "kinds": ["motion_fx"],  # not in KIND_TO_LABEL
        "region_bboxes": [[0, 0, 10, 10]],
    }
    manifest_path.write_text(json.dumps(entry) + "\n")

    dset = ds.TrainingPairDataset(manifest_path)
    _, target = dset[0]
    assert target["labels"].tolist() == [ds.KIND_TO_LABEL["art"]]


def test_collate_fn_returns_parallel_lists_not_stacked_tensor(tmp_path):
    manifest = _write_fixture_manifest(tmp_path)
    dset = ds.TrainingPairDataset(manifest)
    batch = [dset[0], dset[0]]
    images, targets = ds.collate_fn(batch)
    assert isinstance(images, list) and len(images) == 2
    assert isinstance(targets, list) and len(targets) == 2


def test_loads_without_crashing_against_real_manifest_if_present():
    if not ds.DEFAULT_MANIFEST.is_file():
        pytest.skip("work/train_pairs/manifest.jsonl not present -- run augment.py first")
    dset = ds.TrainingPairDataset()
    assert len(dset) > 0
    image_tensor, target = dset[0]
    assert image_tensor.ndim == 3
    assert image_tensor.shape[0] == 3
    assert len(target["labels"]) == len(target["layer_indexes"])
