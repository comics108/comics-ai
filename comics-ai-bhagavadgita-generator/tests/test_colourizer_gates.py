import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from colourize_assets import decide_colourizer


def test_colourizer_gate_accepts_ink_preserving_low_drift_candidate():
    result = decide_colourizer([
        {"ink_edge_f1": .98, "mean_delta_e76": 20., "luminance_mae": 1., "valid_fraction": 1.},
        {"ink_edge_f1": .97, "mean_delta_e76": 22., "luminance_mae": 2., "valid_fraction": .95},
    ], learned=True)
    assert result["decision"] == "accepted"


def test_colourizer_gate_rejects_geometry_or_learned_palette_drift():
    result = decide_colourizer([
        {"ink_edge_f1": .8, "mean_delta_e76": 40., "luminance_mae": 5., "valid_fraction": 1.},
    ], learned=True)
    assert result["decision"] == "rejected"
    assert set(result["failures"]) == {
        "ink_edge_preservation_below_0.95", "luminance_geometry_drift",
        "held_out_palette_error_above_30",
    }
