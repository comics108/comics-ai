import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from positioner_features import build_features, feature_names, KNOWN_KINDS


def test_feature_vector_length_matches_names():
    vec = build_features("balloon", (10, 20, 30, 60), 2, 5, 0.8, text_context_length=42)
    assert len(vec) == len(feature_names())


def test_known_kind_one_hot_is_exclusive():
    vec = build_features("balloon", (0, 0, 10, 10), 0, 1, 1.0)
    names = feature_names()
    kind_values = {n: v for n, v in zip(names, vec) if n.startswith("kind_")}
    assert kind_values["kind_balloon"] == 1.0
    assert sum(kind_values.values()) == 1.0


def test_unknown_kind_is_all_zero_not_a_crash():
    vec = build_features("motion-fx", (0, 0, 10, 10), 0, 1, 1.0)
    names = feature_names()
    kind_values = [v for n, v in zip(names, vec) if n.startswith("kind_")]
    assert sum(kind_values) == 0.0


def test_text_context_length_feature():
    vec_long = build_features("art", (0, 0, 10, 10), 0, 1, 1.0, text_context_length=120)
    vec_none = build_features("art", (0, 0, 10, 10), 0, 1, 1.0, text_context_length=0)
    idx = feature_names().index("text_context_length")
    assert vec_long[idx] == 120.0
    assert vec_none[idx] == 0.0
