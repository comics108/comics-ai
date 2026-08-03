import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_positioner import train
from evaluate_positioning import run


def test_run_with_model_adds_comparable_learned_model_errors(tmp_path):
    model_path = tmp_path / "model.joblib"
    train(model_path=model_path)

    out_path = tmp_path / "eval_report.jsonl"
    results = run(out_path=out_path, model_path=model_path)

    assert results
    any_compared = False
    for r in results:
        if r["pair_count"] == 0:
            continue
        any_compared = True
        assert "learned_model_mean_error_px" in r
        assert r["learned_model_mean_error_px"] is not None
        assert r["learned_model_mean_error_px"] >= 0
    assert any_compared
