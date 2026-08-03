import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from spacing_stats import compute_stats, CANVAS_WIDTH


def test_compute_stats_on_real_data_is_sane():
    stats = compute_stats()
    assert stats["canvas_width"] == CANVAS_WIDTH == 1080

    assert stats["gap"]["count"] > 0
    # real comic layout: consecutive regions shouldn't have a wildly negative or huge median gap
    assert -5000 < stats["gap"]["median"] < 5000

    assert "background" in stats["height_by_kind"]
    assert "balloon" in stats["height_by_kind"]
    for kind, summary in stats["height_by_kind"].items():
        assert summary["count"] > 0
        assert summary["median"] > 0, f"{kind} median height should be positive"


def test_compute_stats_exclude_reduces_counts():
    full = compute_stats()
    partial = compute_stats(exclude_episode_stems={"096e28e97ad843e9bae94902eb85755d"})
    full_total = sum(v["count"] for v in full["height_by_kind"].values())
    partial_total = sum(v["count"] for v in partial["height_by_kind"].values())
    assert partial_total < full_total
    assert partial["gap"]["count"] < full["gap"]["count"]
