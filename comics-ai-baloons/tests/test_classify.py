import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import classify


def _feat(source_file, layer_index, lang_index, swcv, conf=0.9, wobble=0.2, fallback=False):
    return {
        "source_file": source_file,
        "layer_index": layer_index,
        "lang_index": lang_index,
        "stroke_width_cv": swcv,
        "baseline_wobble": wobble,
        "ocr_confidence": conf,
        "needed_crop_fallback": fallback,
    }


def test_uniform_font_classified_machine_set():
    rows = [_feat("f.comics", 0, 0, 0.3), _feat("f.comics", 0, 1, 0.28)]
    results = classify.classify_all(rows)
    assert len(results) == 1
    assert results[0].label == "machine_set"


def test_high_stroke_width_variance_classified_hand_lettered():
    rows = [_feat("f.comics", 0, 0, 1.2, conf=0.5), _feat("f.comics", 0, 1, 1.1, conf=0.6)]
    results = classify.classify_all(rows)
    assert results[0].label == "hand_lettered"


def test_takes_max_across_language_slots():
    # one slot uniform, the other spiky -- balloon should still be flagged (max, not average)
    rows = [_feat("f.comics", 0, 0, 0.2), _feat("f.comics", 0, 1, 1.5, conf=0.4)]
    results = classify.classify_all(rows)
    assert results[0].label == "hand_lettered"


def test_groups_multiple_balloons_independently():
    rows = [
        _feat("f.comics", 0, 0, 0.2),
        _feat("f.comics", 1, 0, 1.3, conf=0.4),
    ]
    results = classify.classify_all(rows)
    by_layer = {r.layer_index: r.label for r in results}
    assert by_layer[0] == "machine_set"
    assert by_layer[1] == "hand_lettered"


def test_known_real_dataset_findings(tmp_path):
    features_path = Path(__file__).resolve().parents[1] / "work" / "lettering_features.jsonl"
    if not features_path.exists():
        return  # only runs after the full pipeline has been executed at least once
    rows = classify.read_features_jsonl(features_path)
    results = classify.classify_all(rows)
    hand_lettered = {(r.source_file, r.layer_index) for r in results if r.label == "hand_lettered"}
    assert ("96d4fcd2f634404494c1ffdef201b503.comics", 181) in hand_lettered
    assert ("d00c610a6f4647dcbd8116014674d255.comics", 67) in hand_lettered
    assert len(hand_lettered) == 2
