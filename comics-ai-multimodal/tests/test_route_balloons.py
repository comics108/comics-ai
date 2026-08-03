import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import route_balloons as rb  # noqa: E402


def test_real_balloon_layers_for_cluster_filters_by_kind_and_membership(tmp_path):
    gt_data = {
        "episode_file": "ep1.comics",
        "width": 10,
        "height": 10,
        "composite_png": "x.png",
        "regions": [
            {"layer_index": 0, "kind": "background", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 1, "kind": "balloon", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 2, "kind": "balloon", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 3, "kind": "balloon", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},  # not in cluster
        ],
    }
    (tmp_path / "ep1.gt.json").write_text(json.dumps(gt_data))
    result = rb.real_balloon_layers_for_cluster("ep1.comics", [0, 1, 2], canvas_dir=tmp_path)
    assert result == [1, 2]


def test_real_balloon_layers_missing_gt_file_returns_empty(tmp_path):
    assert rb.real_balloon_layers_for_cluster("missing.comics", [0, 1], canvas_dir=tmp_path) == []


def test_route_all_integrates_alignment_regions_and_baloons_lookups(tmp_path, monkeypatch):
    canvas_dir = tmp_path / "canvas"
    canvas_dir.mkdir()
    gt_data = {
        "episode_file": "ep1.comics",
        "width": 10,
        "height": 10,
        "composite_png": "x.png",
        "regions": [
            {"layer_index": 5, "kind": "balloon", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 6, "kind": "balloon", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
        ],
    }
    (canvas_dir / "ep1.gt.json").write_text(json.dumps(gt_data))

    alignment_path = tmp_path / "alignment.jsonl"
    alignment_path.write_text(
        json.dumps(
            {
                "photo_file": "p1.jpg",
                "page_index": 0,
                "episode_file": "ep1.comics",
                "matched_layer_indexes": [5],
                "ground_truth_cluster": [5, 6],
                "confidence": 0.9,
                "status": "matched",
                "reason": "",
            }
        )
        + "\n"
    )

    regions_path = tmp_path / "regions.jsonl"
    regions_path.write_text(
        json.dumps({"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "balloon", "confidence": 0.5, "bbox": [0, 0, 1, 1]})
        + "\n"
        + json.dumps({"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "character", "confidence": 0.5, "bbox": [0, 0, 1, 1]})
        + "\n"
    )

    # Fake comics-ai-baloons work files: monkeypatch the path constants to point at real temp
    # fixture files, rather than mocking _load_jsonl itself -- exercises the real read path.
    fake_matches = tmp_path / "matches.jsonl"
    fake_matches.write_text(
        json.dumps({"source_file": "ep1.comics", "layer_index": 5, "status": "matched", "csv_row_id": "P1_1"}) + "\n"
    )
    fake_renders = tmp_path / "renders.jsonl"
    fake_renders.write_text(
        json.dumps({"source_file": "ep1.comics", "layer_index": 5, "lang_code": "ru", "rendered": True}) + "\n"
        + json.dumps({"source_file": "ep1.comics", "layer_index": 5, "lang_code": "uk", "rendered": False}) + "\n"
    )
    fake_balloons = tmp_path / "balloons.jsonl"
    fake_balloons.write_text("")

    monkeypatch.setattr(rb, "BALOONS_MATCHES_JSONL", fake_matches)
    monkeypatch.setattr(rb, "BALOONS_RENDERS_JSONL", fake_renders)
    monkeypatch.setattr(rb, "BALOONS_BALLOONS_JSONL", fake_balloons)
    monkeypatch.setattr(rb, "BALOONS_OUTPUT_DIR", tmp_path / "nonexistent_output")

    results = rb.route_all(alignment_path, regions_path, canvas_dir, tmp_path / "out.jsonl")
    assert len(results) == 1
    r = results[0]
    assert r.real_balloon_layer_indexes == [5, 6]
    assert r.predicted_balloon_region_count == 1
    assert r.translated_layer_indexes == [5]
    assert r.rendered_languages_by_layer == {5: ["ru"]}
    assert r.packaged_output_available is False


def test_route_all_real_data_if_present(tmp_path):
    import pytest

    if not rb.DEFAULT_ALIGNMENT.is_file() or not rb.DEFAULT_REGIONS.is_file():
        pytest.skip("work/alignment.jsonl or work/regions.jsonl not present -- run the pipeline first")
    if not rb.BALOONS_BALLOONS_JSONL.is_file():
        pytest.skip("comics-ai-baloons work/balloons.jsonl not present")

    out_path = tmp_path / "handoff.jsonl"
    results = rb.route_all(out_path=out_path)
    assert out_path.is_file()
    for r in results:
        assert isinstance(r.real_balloon_layer_indexes, list)
        assert isinstance(r.packaged_output_available, bool)
