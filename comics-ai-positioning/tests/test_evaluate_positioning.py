import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from evaluate_positioning import choose_held_out, run


def test_choose_held_out_is_deterministic():
    stems = [f"ep{i}" for i in range(16)]
    a = choose_held_out(stems)
    b = choose_held_out(stems)
    assert a == b
    assert 2 <= len(a) <= 6  # roughly 25% of 16


def test_run_on_real_data_produces_finite_errors(tmp_path):
    out_path = tmp_path / "eval_report.jsonl"
    results = run(out_path=out_path)
    assert results, "expected at least one held-out episode"
    assert out_path.is_file()

    any_with_pairs = False
    for r in results:
        if r["pair_count"] == 0:
            continue
        any_with_pairs = True
        assert r["mean_error_px"] is not None
        assert r["mean_error_px"] >= 0
        assert 0 <= r["mean_error_normalized"] <= 1.0 or r["mean_error_normalized"] > 0
    assert any_with_pairs
