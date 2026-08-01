import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import evaluate as ev  # noqa: E402


def test_count_agreement_exact_match_is_one():
    assert ev._count_agreement(3, 3) == 1.0


def test_count_agreement_both_zero_is_one_not_undefined():
    assert ev._count_agreement(0, 0) == 1.0


def test_count_agreement_decays_with_divergence():
    assert ev._count_agreement(1, 5) == pytest.approx(1 - 4 / 5)
    assert ev._count_agreement(0, 5) == pytest.approx(0.0)


def test_load_ground_truth_kind_counts_filters_by_layer_indexes(tmp_path):
    gt_data = {
        "episode_file": "ep1.comics",
        "width": 10,
        "height": 10,
        "composite_png": "x.png",
        "regions": [
            {"layer_index": 0, "kind": "background", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 1, "kind": "character", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 2, "kind": "balloon", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 3, "kind": "character", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},  # not in wanted set
        ],
    }
    (tmp_path / "ep1.gt.json").write_text(json.dumps(gt_data))
    counts = ev.load_ground_truth_kind_counts("ep1.comics", [0, 1, 2], canvas_dir=tmp_path)
    assert counts == {"background": 1, "character": 1, "balloon": 1}


def test_load_ground_truth_kind_counts_missing_file_returns_empty(tmp_path):
    assert ev.load_ground_truth_kind_counts("missing.comics", [0, 1], canvas_dir=tmp_path) == {}


def test_evaluate_page_computes_per_kind_agreement(tmp_path):
    gt_data = {
        "episode_file": "ep1.comics",
        "width": 10,
        "height": 10,
        "composite_png": "x.png",
        "regions": [
            {"layer_index": 0, "kind": "background", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 1, "kind": "character", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 2, "kind": "character", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
            {"layer_index": 3, "kind": "balloon", "kind_source": "explicit", "bbox": [0, 0, 1, 1]},
        ],
    }
    (tmp_path / "ep1.gt.json").write_text(json.dumps(gt_data))

    predicted = [
        {"predicted_kind": "background", "confidence": 0.9},
        {"predicted_kind": "character", "confidence": 0.8},
        {"predicted_kind": "balloon", "confidence": 0.7},
        {"predicted_kind": "balloon", "confidence": 0.6},  # over-predicted balloon (truth has 1)
    ]

    result = ev.evaluate_page("p.jpg", 0, "ep1.comics", [0, 1, 2, 3], predicted, canvas_dir=tmp_path)
    assert result.predicted_counts == {"art": 0, "background": 1, "character": 1, "balloon": 2}
    assert result.ground_truth_counts == {"art": 0, "background": 1, "character": 2, "balloon": 1}
    assert result.per_kind_agreement["background"] == 1.0  # 1 vs 1
    assert result.per_kind_agreement["art"] == 1.0  # 0 vs 0
    assert result.per_kind_agreement["character"] == pytest.approx(0.5)  # 1 vs 2
    assert result.per_kind_agreement["balloon"] == pytest.approx(0.5)  # 2 vs 1


def test_evaluate_all_real_data_if_present(tmp_path):
    if not ev.DEFAULT_ALIGNMENT.is_file() or not ev.DEFAULT_REGIONS.is_file():
        pytest.skip("work/alignment.jsonl or work/regions.jsonl not present -- run the pipeline first")
    out_path = tmp_path / "eval_report.jsonl"
    results = ev.evaluate_all(out_path=out_path)
    assert out_path.is_file()
    for r in results:
        assert 0.0 <= r.mean_agreement <= 1.0
