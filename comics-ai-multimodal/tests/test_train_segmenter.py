import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_segmenter as ts  # noqa: E402
from dataset import TrainingPairDataset  # noqa: E402


def _write_fixture_manifest(tmp_path: Path, entries: list[dict]) -> Path:
    from PIL import Image

    for e in entries:
        Image.new("RGB", (e["_w"], e["_h"]), (10, 20, 30)).save(e["clean_png"])
        Image.new("RGB", (e["_w"], e["_h"]), (12, 18, 33)).save(e["degraded_png"])

    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for e in entries:
            e = {k: v for k, v in e.items() if not k.startswith("_")}
            f.write(json.dumps(e) + "\n")
    return manifest_path


def test_compute_class_weights_gives_higher_weight_to_rarer_class(tmp_path):
    # Entry 0: almost entirely "art" (label 0), tiny "balloon" (label 3) sliver.
    entries = [
        {
            "episode_file": "x.comics",
            "cluster_index": 0,
            "clean_png": str(tmp_path / "c0.png"),
            "degraded_png": str(tmp_path / "d0.jpg"),
            "bbox": [0, 0, 100, 100],
            "layer_indexes": [0],
            "kinds": ["balloon"],
            "region_bboxes": [[0, 0, 5, 5]],  # tiny sliver
            "_w": 100,
            "_h": 100,
        }
    ]
    manifest = _write_fixture_manifest(tmp_path, entries)
    dset = TrainingPairDataset(manifest)
    weights = ts.compute_class_weights(dset, [0], size=(100, 100))

    assert weights.shape == (4,)
    # "balloon" (rare, tiny sliver) must get a higher weight than "art" (dominant, most of the area)
    assert weights[ts.NUM_CLASSES - 1] > weights[0]  # balloon index 3 > art index 0


def test_unet_collate_produces_correctly_shaped_batch(tmp_path):
    entries = [
        {
            "episode_file": "x.comics",
            "cluster_index": 0,
            "clean_png": str(tmp_path / "c1.png"),
            "degraded_png": str(tmp_path / "d1.jpg"),
            "bbox": [0, 0, 50, 80],
            "layer_indexes": [0, 1],
            "kinds": ["background", "character"],
            "region_bboxes": [[0, 0, 50, 80], [10, 10, 30, 40]],
            "_w": 50,
            "_h": 80,
        },
        {
            "episode_file": "x.comics",
            "cluster_index": 1,
            "clean_png": str(tmp_path / "c2.png"),
            "degraded_png": str(tmp_path / "d2.jpg"),
            "bbox": [0, 0, 30, 30],
            "layer_indexes": [2],
            "kinds": ["balloon"],
            "region_bboxes": [[5, 5, 20, 20]],
            "_w": 30,
            "_h": 30,
        },
    ]
    manifest = _write_fixture_manifest(tmp_path, entries)
    dset = TrainingPairDataset(manifest)
    batch = [dset[0], dset[1]]
    images, label_maps = ts.unet_collate(batch)
    assert images.shape == (2, 3, *ts.TRAIN_SIZE)
    assert label_maps.shape == (2, *ts.TRAIN_SIZE)
