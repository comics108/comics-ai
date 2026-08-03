import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from transform_stats import compute_stats


def test_compute_stats_on_real_data_is_sane():
    stats = compute_stats()
    assert stats["counts_by_kind"]["balloon"] == 825
    # Real, checked fact: balloons animate alpha/scale most of the time (fade+grow-in reveal).
    assert stats["occurrence_rate"]["balloon"]["alpha"] > 0.7
    assert stats["occurrence_rate"]["balloon"]["scale"] > 0.7
    # Real, checked fact: backgrounds almost never animate scale/alpha.
    assert stats["occurrence_rate"]["background"]["alpha"] < 0.05
    # Real, checked fact: alpha reveals are a fade-in (0 -> 1), scale reveals a grow-in (0.6 -> 1).
    assert stats["alpha_from_to"] == [0.0, 1.0]
    assert stats["scale_from_to"] == [0.6, 1.0]
    assert stats["median_duration"]["alpha"] > 0


def test_compute_stats_exclude_reduces_counts():
    full = compute_stats()
    reduced = compute_stats(exclude_episode_stems={"8a89f7d689fb441ea280cd782276bd7a"})
    full_total = sum(full["counts_by_kind"].values())
    reduced_total = sum(reduced["counts_by_kind"].values())
    assert reduced_total < full_total
    assert full_total - reduced_total == 200  # that file's real known layer count
