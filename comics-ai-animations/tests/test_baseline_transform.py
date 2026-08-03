import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_transform import propose_property, propose_reveal
from transform_stats import compute_stats

REAL_STATS = compute_stats()  # real, full-dataset stats -- these tests assert against real behavior


def test_balloon_predicts_alpha_and_scale_but_not_translate_or_rotate():
    reveal = propose_reveal("balloon", REAL_STATS)
    assert reveal["alpha"].occurs is True
    assert reveal["alpha"].from_value == {"alpha": 0.0}
    assert reveal["alpha"].to_value == {"alpha": 1.0}
    assert reveal["scale"].occurs is True
    assert reveal["scale"].from_value == {"scaleX": 0.6, "scaleY": 0.6}
    assert reveal["translate"].occurs is False
    assert reveal["rotate"].occurs is False


def test_background_predicts_no_animation_at_all():
    # Real fact: background's highest occurrence rate (translate, ~32.5%) is still under the
    # majority threshold -- the calibrated baseline correctly predicts "static" for backgrounds.
    reveal = propose_reveal("background", REAL_STATS)
    assert all(not r.occurs for r in reveal.values())


def test_character_and_art_predict_translate_only():
    for kind in ("character", "art"):
        reveal = propose_reveal(kind, REAL_STATS)
        assert reveal["translate"].occurs is True
        assert reveal["alpha"].occurs is False
        assert reveal["scale"].occurs is False


def test_translate_occurrence_has_no_fabricated_direction():
    reveal = propose_property("character", "translate", REAL_STATS)
    assert reveal.occurs is True
    assert reveal.from_value == {}
    assert reveal.to_value == {}
    assert reveal.end > 0  # duration is real and calibrated, even though direction isn't


def test_unknown_kind_defaults_to_no_occurrence_not_a_crash():
    reveal = propose_reveal("nonexistent_kind", REAL_STATS)
    assert all(not r.occurs for r in reveal.values())
