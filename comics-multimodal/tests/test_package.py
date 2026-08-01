import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baloons_bridge  # noqa: E402
import package as pkg  # noqa: E402


def test_package_photo_page_produces_valid_reopenable_comics(tmp_path, monkeypatch):
    fake_crop = np.full((30, 20, 3), (10, 20, 30), dtype=np.uint8)
    monkeypatch.setattr(pkg, "extract_crop_image", lambda *a, **k: fake_crop)

    regions = [
        {"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "character", "confidence": 0.9, "bbox": [5, 5, 25, 35]},
        {"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "balloon", "confidence": 0.8, "bbox": [50, 50, 70, 80]},
    ]
    result = pkg.package_photo_page("p1.jpg", 0, "ep1.comics", regions, tmp_path, tmp_path / "out")

    assert result.status == "packaged"
    assert result.layer_count == 2
    out_path = Path(result.output_path)
    assert out_path.is_file()

    # Round-trip: reopen with our own ComicsArchive/tiling (same pattern used throughout this
    # project to verify .comics output validity) and confirm the schema + stitched pixels are
    # correct.
    archive = baloons_bridge.ComicsArchive(out_path)
    data = archive.read_data_json()
    assert data["width"] == pkg.TRAIN_SIZE[1]
    assert data["height"] == pkg.TRAIN_SIZE[0]
    assert len(data["layers"]) == 2
    assert data["layers"][0]["kind"] == "character"
    assert data["layers"][1]["kind"] == "balloon"

    img_meta = data["layers"][0]["images"][0]
    stitched = baloons_bridge.stitch_image(archive, img_meta["file"], img_meta["width"], img_meta["height"])
    assert stitched.size == (20, 30)
    assert stitched.convert("RGB").getpixel((0, 0)) == (10, 20, 30)


def test_package_photo_page_skips_degenerate_boxes(tmp_path, monkeypatch):
    fake_crop = np.zeros((5, 5, 3), dtype=np.uint8)
    monkeypatch.setattr(pkg, "extract_crop_image", lambda *a, **k: fake_crop)
    regions = [
        {"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "art", "confidence": 0.5, "bbox": [0, 0, 1, 1]},  # degenerate (0-width after dim check)
    ]
    result = pkg.package_photo_page("p1.jpg", 0, "ep1.comics", regions, tmp_path, tmp_path / "out")
    # bbox [0,0,1,1] has w=h=1, below MIN_REGION_DIM=2 -- must be skipped, not crash
    assert result.status == "skipped_no_regions"
    assert result.output_path is None


def test_package_photo_page_with_no_regions_at_all_is_skipped(tmp_path):
    result = pkg.package_photo_page("p1.jpg", 0, "ep1.comics", [], tmp_path, tmp_path / "out")
    assert result.status == "skipped_no_regions"


def test_package_all_real_data_if_present(tmp_path):
    if not pkg.DEFAULT_ALIGNMENT.is_file() or not pkg.DEFAULT_REGIONS.is_file():
        pytest.skip("work/alignment.jsonl or work/regions.jsonl not present -- run the pipeline first")

    out_dir = tmp_path / "output"
    results = pkg.package_all(out_dir=out_dir)
    assert len(results) > 0
    packaged = [r for r in results if r.status == "packaged"]
    assert len(packaged) > 0

    # Verify at least one real packaged file is a structurally valid zip with parseable data.json.
    sample = packaged[0]
    with zipfile.ZipFile(sample.output_path) as zf:
        assert "data.json" in zf.namelist()
        data = json.loads(zf.read("data.json").decode("utf-8-sig"))
        assert "layers" in data and len(data["layers"]) == sample.layer_count
