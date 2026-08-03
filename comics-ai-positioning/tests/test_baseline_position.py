import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from baseline_position import position_page, load_stats
from positioning_models import RegionFeatures

FIXTURE_STATS = {
    "canvas_width": 1080,
    "height_by_kind": {
        "background": {"count": 10, "median": 2000},
        "balloon": {"count": 10, "median": 200},
    },
    "gap": {"count": 10, "median": 50},
}


def test_position_page_is_deterministic_and_order_preserving():
    regions = [
        RegionFeatures("balloon", "predicted", (0, 0, 100, 50), 0, reading_order_index=1),
        RegionFeatures("background", "predicted", (0, 0, 256, 256), 0, reading_order_index=0),
    ]
    proposals_1 = position_page(regions, FIXTURE_STATS, region_ids=["b", "a"])
    proposals_2 = position_page(regions, FIXTURE_STATS, region_ids=["b", "a"])
    assert [p.to_jsonable() for p in proposals_1] == [p.to_jsonable() for p in proposals_2]

    # reading_order_index=0 (background) must come first regardless of input list order
    by_id = {p.region_id: p for p in proposals_1}
    assert by_id["a"].proposed_y < by_id["b"].proposed_y
    assert by_id["a"].proposed_y == 0
    assert by_id["b"].proposed_y == 2000 + 50  # background height + gap


def test_position_page_unseen_kind_falls_back_to_overall_median_not_crash():
    regions = [RegionFeatures("motion-fx", "predicted", (0, 0, 50, 50), 0, reading_order_index=0)]
    proposals = position_page(regions, FIXTURE_STATS, region_ids=["x"])
    assert proposals[0].proposed_y == 0  # first region always starts at 0


def test_position_page_on_real_training_pairs():
    pairs_dir = Path(__file__).resolve().parents[1] / "work" / "train_pairs"
    files = sorted(pairs_dir.glob("*.jsonl"))
    assert files, "expected build_pairs.py to have already run (work/train_pairs/*.jsonl)"

    stats = load_stats()
    rows = [json.loads(line) for line in files[0].read_text().splitlines() if line.strip()]
    regions = [
        RegionFeatures(
            kind=r["region"]["kind"],
            kind_source=r["region"]["kind_source"],
            local_bbox=tuple(r["region"]["local_bbox"]),
            page_index=r["region"]["page_index"],
            reading_order_index=r["region"]["reading_order_index"],
        )
        for r in rows
    ]
    proposals = position_page(regions, stats)
    assert len(proposals) == len(regions)
    for p in proposals:
        assert isinstance(p.proposed_x, int)
        assert isinstance(p.proposed_y, int)
