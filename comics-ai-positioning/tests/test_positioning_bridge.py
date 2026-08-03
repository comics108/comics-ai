import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import positioning_bridge as pb


def test_canvas_dir_has_all_27_ground_truth_files():
    pb.require_canvas_ground_truth()
    files = sorted(pb.CANVAS_DIR.glob("*.gt.json"))
    assert len(files) == 27, f"expected 27 ground-truth files, found {len(files)}"


def test_load_canvas_reference_shape_matches_specifications():
    ref = pb.load_canvas_reference("096e28e97ad843e9bae94902eb85755d.comics")
    assert set(ref.keys()) == {"episode_file", "width", "height", "composite_png", "regions"}
    assert ref["regions"], "expected at least one region"
    region = ref["regions"][0]
    assert set(region.keys()) == {"layer_index", "kind", "kind_source", "bbox"}
    assert len(region["bbox"]) == 4


def test_load_canvas_reference_missing_episode_raises():
    import pytest

    with pytest.raises(FileNotFoundError):
        pb.load_canvas_reference("does-not-exist.comics")


def test_load_all_canvas_references_returns_27():
    refs = pb.load_all_canvas_references()
    assert len(refs) == 27


def test_iter_matched_alignment_rows_real_data():
    rows = list(pb.iter_matched_alignment_rows())
    assert len(rows) > 0, "expected at least some real matched photo/page rows"
    for row in rows:
        assert row["status"] == "matched"
        assert row["ground_truth_cluster"]
        assert row["episode_file"]


def test_import_multimodal_module_resting_position():
    mod = pb.import_multimodal_module("resting_position")
    assert hasattr(mod, "resolve_resting_transform")
