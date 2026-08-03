import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import positioning_bridge as pb
from build_pairs import (
    build_all,
    build_pairs_for_row,
    _load_regions_by_page,
    _sort_reading_order,
    _sort_top_to_bottom,
)


def test_sort_reading_order_handles_3x3_grid_like_naive_sort_cannot():
    """The motivating real-world case: a page with a 3x3 grid of panels (confirmed real layout
    shape -- sdd-comics-ai-multimodal's Checkpoint A found the printed source is "a conventionally
    paginated comic, fixed rectangular panel grids"). Same-row panels have close-but-not-identical
    Y (hand-drawn), which the naive (y, x) sort misorders since x only breaks *exact* Y ties."""
    # 3 rows x 3 cols, each panel ~100x100, with a few px of hand-drawn Y jitter within each row
    grid = {
        "1": [0, 0, 100, 100], "2": [110, 3, 210, 103], "3": [220, -2, 320, 98],
        "4": [0, 210, 100, 310], "5": [110, 205, 210, 305], "6": [220, 208, 320, 308],
        "7": [0, 415, 100, 515], "8": [110, 412, 210, 512], "9": [220, 418, 320, 518],
    }
    regions = [{"bbox": bbox, "id": name} for name, bbox in grid.items()]

    naive = [r["id"] for r in _sort_top_to_bottom(regions)]
    fixed = [r["id"] for r in _sort_reading_order(regions)]

    assert fixed == ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
    # the naive sort is NOT raster order here -- this is exactly the bug, not a strawman
    assert naive != ["1", "2", "3", "4", "5", "6", "7", "8", "9"]


def test_sort_reading_order_matches_naive_sort_for_single_column_page():
    """For an already vertically-stacked page (no side-by-side panels), the fix must not change
    behavior -- strict improvement for grids, no regression for the common single-column case."""
    regions = [
        {"bbox": [10, 0, 200, 100], "id": "a"},
        {"bbox": [10, 120, 200, 220], "id": "b"},
        {"bbox": [10, 240, 200, 340], "id": "c"},
    ]
    naive = [r["id"] for r in _sort_top_to_bottom(regions)]
    fixed = [r["id"] for r in _sort_reading_order(regions)]
    assert fixed == naive == ["a", "b", "c"]


def test_build_pairs_for_known_real_row():
    row = {
        "photo_file": "20260731_153228.jpg",
        "page_index": 0,
        "episode_file": "d00c610a6f4647dcbd8116014674d255.comics",
        "matched_layer_indexes": [87, 95, 109],
        "ground_truth_cluster": [2, 6, 14, 20, 25, 28, 29, 30, 31, 32, 67, 68, 87, 95, 97, 108, 109],
        "confidence": 0.8,
        "status": "matched",
        "reason": "",
    }
    regions_by_page = _load_regions_by_page()
    canvas_cache: dict = {}
    pairs = build_pairs_for_row(row, regions_by_page, canvas_cache)

    assert pairs, "expected at least one paired region for this known real row"
    for pair in pairs:
        assert pair.episode_file == row["episode_file"]
        assert pair.photo_file == row["photo_file"]
        # target_bbox must be a real GroundTruthRegion bbox for this episode's canvas
        canvas = canvas_cache[row["episode_file"]]
        matching = [
            r for r in canvas["regions"]
            if r["layer_index"] == pair.target_layer_index and tuple(r["bbox"]) == pair.target_bbox
        ]
        assert matching, f"target_bbox for layer {pair.target_layer_index} doesn't match real ground truth"
        assert pair.target_transform["x"] == pair.target_bbox[0]
        assert pair.target_transform["y"] == pair.target_bbox[1]
        # kind must actually have existed as both a predicted and ground-truth kind for this page
        assert pair.region.kind in {"balloon", "art", "background", "character"}


def test_build_pairs_never_exceeds_min_count_per_kind():
    row = {
        "photo_file": "20260731_153228.jpg",
        "page_index": 0,
        "episode_file": "d00c610a6f4647dcbd8116014674d255.comics",
        "ground_truth_cluster": [2, 6, 14, 20, 25, 28, 29, 30, 31, 32, 67, 68, 87, 95, 97, 108, 109],
        "confidence": 0.8,
    }
    regions_by_page = _load_regions_by_page()
    canvas_cache: dict = {}
    pairs = build_pairs_for_row(row, regions_by_page, canvas_cache)
    # real counts established by manual inspection: predicted balloon=5 art=4 background=4 character=3
    # ground truth (within cluster) balloon=7 character=5 art=4 background=1
    # -> paired counts must be the min per kind: balloon=5 art=4 background=1 character=3
    from collections import Counter
    kind_counts = Counter(p.region.kind for p in pairs)
    assert kind_counts == {"balloon": 5, "art": 4, "background": 1, "character": 3}


def test_build_all_on_real_data_produces_nonempty_output(tmp_path):
    counts = build_all(out_dir=tmp_path)
    assert sum(counts.values()) > 0
    assert len(counts) > 0
    for episode_file, n in counts.items():
        path = tmp_path / f"{Path(episode_file).stem}.jsonl"
        assert path.is_file()
        lines = path.read_text().strip().splitlines()
        assert len(lines) == n
