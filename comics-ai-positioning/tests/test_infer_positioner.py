import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_positioner import train
from infer_positioner import load_model, position_page_with_model
from positioning_models import RegionFeatures
from spacing_stats import compute_stats


def test_position_page_with_model_matches_baseline_when_residual_zero(tmp_path):
    class ZeroModel:
        def predict(self, X):
            return [0.0] * len(X)

    stats = compute_stats()
    regions = [
        RegionFeatures("balloon", "predicted", (0, 0, 50, 50), 0, reading_order_index=1),
        RegionFeatures("background", "predicted", (0, 0, 256, 256), 0, reading_order_index=0),
    ]
    from baseline_position import position_page

    baseline = {p.region_id: p for p in position_page(regions, stats, region_ids=["b", "a"])}
    modeled = position_page_with_model(
        regions, stats, ZeroModel(), region_ids=["b", "a"]
    )
    for p in modeled:
        assert p.proposed_y == baseline[p.region_id].proposed_y
        assert p.source == "learned_model"


def test_real_trained_model_loads_and_runs(tmp_path):
    model_path = tmp_path / "model.joblib"
    train(model_path=model_path)
    model = load_model(model_path)

    stats = compute_stats()
    regions = [
        RegionFeatures("balloon", "predicted", (10, 10, 60, 60), 0, reading_order_index=0),
        RegionFeatures("character", "predicted", (0, 0, 200, 220), 0, reading_order_index=1),
    ]
    proposals = position_page_with_model(regions, stats, model, region_ids=["r0", "r1"])
    assert len(proposals) == 2
    for p in proposals:
        assert isinstance(p.proposed_y, int)
