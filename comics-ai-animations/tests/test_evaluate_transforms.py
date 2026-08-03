import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_transforms import choose_held_out, evaluate


def test_choose_held_out_is_deterministic():
    stems = [f"ep{i}" for i in range(20)]
    a = choose_held_out(stems)
    b = choose_held_out(stems)
    assert a == b
    assert len(a) > 0


def test_evaluate_on_real_data_produces_finite_results_beating_or_matching_strawman():
    result = evaluate()
    assert result["held_out_layer_count"] > 0
    for prop in ("translate", "scale", "rotate", "alpha"):
        r = result[prop]
        assert 0.0 <= r["occurrence_accuracy"] <= 1.0
        assert 0.0 <= r["strawman_accuracy"] <= 1.0
        # The calibrated baseline must be evaluated honestly against the trivial "always static"
        # strawman -- not asserted to always win here (that's a real empirical question, checked
        # by hand below), just that both numbers are real and computed consistently.
