"""Cheap fixture-based smoke test for train_maskrcnn's wiring (collate, target conversion, loss
computation, checkpoint save) BEFORE committing to an expensive real-dataset run. Uses tiny
synthetic images and pretrained=False-equivalent speed isn't controllable via the CLI, so this is
marked slow-ish but still much cheaper than a real run (small images, 1 epoch, 2 samples).
"""

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import train_segmenter as ts  # noqa: E402


def _write_fixture_manifest(tmp_path: Path) -> Path:
    from PIL import Image

    entries = [
        {
            "episode_file": "fixture.comics",
            "cluster_index": 0,
            "bbox": [0, 0, 64, 96],
            "layer_indexes": [0, 1],
            "kinds": ["background", "character"],
            "region_bboxes": [[0, 0, 64, 96], [10, 10, 40, 60]],
        },
        {
            "episode_file": "fixture.comics",
            "cluster_index": 1,
            "bbox": [0, 0, 48, 48],
            "layer_indexes": [2],
            "kinds": ["balloon"],
            "region_bboxes": [[5, 5, 30, 30]],
        },
    ]
    for i, e in enumerate(entries):
        w, h = e["bbox"][2], e["bbox"][3]
        clean_path = tmp_path / f"c{i}.png"
        degraded_path = tmp_path / f"d{i}.jpg"
        Image.new("RGB", (w, h), (20, 40, 60)).save(clean_path)
        Image.new("RGB", (w, h), (22, 38, 58)).save(degraded_path)
        e["clean_png"] = str(clean_path)
        e["degraded_png"] = str(degraded_path)

    manifest_path = tmp_path / "manifest.jsonl"
    with manifest_path.open("w") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")
    return manifest_path


@pytest.mark.slow
def test_train_maskrcnn_runs_end_to_end_on_tiny_fixture(tmp_path):
    manifest = _write_fixture_manifest(tmp_path)
    checkpoint_out = tmp_path / "maskrcnn_smoke.pt"

    history = ts.train_maskrcnn(
        manifest_path=manifest,
        checkpoint_out=checkpoint_out,
        epochs=1,
        batch_size=1,
        val_fraction=0.5,
        device="cpu",
    )

    assert len(history["train_loss"]) == 1
    assert history["train_loss"][0] == history["train_loss"][0]  # not NaN
    assert "val_iou" in history
    assert checkpoint_out.is_file()
    assert checkpoint_out.with_suffix(".history.json").is_file()
