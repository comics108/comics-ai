import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import augment as aug  # noqa: E402
from render_canvas import GroundTruthRegion  # noqa: E402


def _region(layer_index, y0, y1, kind="art", x0=0, x1=100):
    return GroundTruthRegion(layer_index=layer_index, kind=kind, kind_source="explicit", bbox=(x0, y0, x1, y1))


def test_cluster_layers_by_y_groups_nearby_and_splits_far():
    regions = [
        _region(0, 0, 100, kind="background"),
        _region(1, 50, 150, kind="character"),
        _region(2, 20000, 20100, kind="background"),  # far away -> new cluster
        _region(3, 20050, 20150, kind="character"),
    ]
    clusters = aug.cluster_layers_by_y(regions)
    assert len(clusters) == 2
    assert {r.layer_index for r in clusters[0]} == {0, 1}
    assert {r.layer_index for r in clusters[1]} == {2, 3}


def test_cluster_layers_transitive_chain_joins_even_if_ends_are_far_apart():
    # 0 and 2 are individually within y_window of their neighbor, but 0<->2 directly would not be,
    # if compared naively -- the chain must still merge them into one cluster.
    y_window = aug.Y_WINDOW
    regions = [
        _region(0, 0, 10),
        _region(1, y_window - 100, y_window - 90),
        _region(2, 2 * y_window - 150, 2 * y_window - 140),
    ]
    clusters = aug.cluster_layers_by_y(regions, y_window=y_window)
    assert len(clusters) == 1
    assert {r.layer_index for r in clusters[0]} == {0, 1, 2}


def test_cluster_layers_by_scene_handles_region_whose_y_center_precedes_its_own_background():
    # Regression test (found via Phase 5's test_align_photo.py): a balloon positioned near the top
    # of its panel can have a smaller y-center than its own scene's background layer -- a
    # sequential "flush on background encounter" scan (the original implementation) would sort the
    # balloon before the background and spuriously isolate it into its own single-item cluster.
    # Nearest-background assignment must still group it correctly.
    regions = [
        _region(0, 0, 50, kind="background"),  # y-center 25
        _region(1, 10, 40, kind="character"),  # y-center 25
        _region(2, 15, 25, kind="balloon"),  # y-center 20 -- smaller than its own background's 25
    ]
    clusters = aug.cluster_layers_by_scene(regions)
    assert len(clusters) == 1
    assert {r.layer_index for r in clusters[0]} == {0, 1, 2}


def test_build_page_groups_merges_consecutive_scene_clusters():
    scene_clusters = [
        [_region(0, 0, 100, kind="background")],
        [_region(1, 200, 300, kind="background")],
        [_region(2, 400, 500, kind="background")],
        [_region(3, 600, 700, kind="background")],
        [_region(4, 800, 900, kind="background")],
        [_region(5, 1000, 1100, kind="background")],
    ]
    groups = aug.build_page_groups(scene_clusters, group_size=4)
    assert len(groups) == 2
    assert {r.layer_index for r in groups[0]} == {0, 1, 2, 3}
    assert {r.layer_index for r in groups[1]} == {4, 5}


def test_build_page_groups_handles_empty_input():
    assert aug.build_page_groups([]) == []


def test_cluster_bbox_is_union_of_members():
    regions = [_region(0, 10, 50, x0=5, x1=60), _region(1, 30, 90, x0=20, x1=40)]
    assert aug.cluster_bbox(regions) == (5, 10, 60, 90)


def test_cluster_layers_by_scene_anchors_on_background_layers():
    # Two scenes, each starting with a background, packed densely enough that the old y-window
    # chaining rule would have merged them into one giant cluster -- scene-anchoring must not.
    regions = [
        _region(0, 0, 50, kind="background"),
        _region(1, 100, 400, kind="character"),
        _region(2, 200, 250, kind="balloon"),
        _region(3, 900, 950, kind="background"),  # new scene starts here
        _region(4, 1000, 1300, kind="character"),
    ]
    clusters = aug.cluster_layers_by_scene(regions)
    assert len(clusters) == 2
    assert {r.layer_index for r in clusters[0]} == {0, 1, 2}
    assert {r.layer_index for r in clusters[1]} == {3, 4}


def test_cluster_layers_by_scene_splits_oversized_sparse_background_span():
    # Only one background anchor but content spans far more than max_cluster_height -- must be
    # defensively re-split rather than producing one giant crop.
    regions = [_region(0, 0, 50, kind="background")]
    y = 0
    for i in range(1, 8):
        y += 1000
        regions.append(_region(i, y, y + 100, kind="character"))
    clusters = aug.cluster_layers_by_scene(regions, max_cluster_height=2500)
    assert len(clusters) > 1
    for cluster in clusters:
        bbox = aug.cluster_bbox(cluster)
        assert bbox[3] - bbox[1] <= 2500


def test_cluster_layers_by_scene_bounds_true_bbox_not_just_center_distance():
    # Regression test: a real-dataset bug where individual regions' own height compounded the
    # true union bbox well past max_cluster_height even though center-to-center distances were
    # all individually small. Each region here is 900px tall with centers only 200px apart, so
    # naive center-distance chunking would keep merging indefinitely -- the true bbox must still
    # be bounded.
    regions = [_region(0, 0, 50, kind="background")]
    y = 0
    for i in range(1, 20):
        y += 200
        regions.append(_region(i, y, y + 900, kind="character"))
    clusters = aug.cluster_layers_by_scene(regions, max_cluster_height=2500)
    assert len(clusters) > 1
    for cluster in clusters:
        bbox = aug.cluster_bbox(cluster)
        assert bbox[3] - bbox[1] <= 2500


def test_degrade_preserves_size_and_produces_valid_image():
    im = Image.new("RGB", (300, 200), (128, 64, 200))
    rng = random.Random(42)
    out = aug.degrade(im, rng)
    assert out.size == (300, 200)
    assert out.mode == "RGB"


def test_degrade_is_deterministic_given_a_seeded_rng():
    im = Image.new("RGB", (150, 150), (10, 20, 30))
    out1 = aug.degrade(im, random.Random(7))
    out2 = aug.degrade(im, random.Random(7))
    # Same seed sequence -> same random draws -> pixel-identical output (numpy noise uses the
    # global np.random state though, so this only holds if that's also reset -- documented below).
    import numpy as np

    np.random.seed(7)
    out1 = aug.degrade(im, random.Random(7))
    np.random.seed(7)
    out2 = aug.degrade(im, random.Random(7))
    assert list(out1.getdata()) == list(out2.getdata())


def test_degrade_with_boxes_keeps_boxes_within_crop_bounds():
    im = Image.new("RGB", (200, 150), (100, 100, 100))
    boxes = [(10, 10, 50, 50), (0, 0, 200, 150)]
    _, transformed = aug.degrade_with_boxes(im, boxes, random.Random(3))
    assert len(transformed) == 2
    for (x0, y0, x1, y1) in transformed:
        assert 0 <= x0 < x1 <= 200
        assert 0 <= y0 < y1 <= 150


def test_degrade_with_boxes_is_deterministic_given_seed():
    im = Image.new("RGB", (120, 90), (5, 5, 5))
    boxes = [(20, 20, 60, 60)]
    np = __import__("numpy")
    np.random.seed(11)
    _, t1 = aug.degrade_with_boxes(im, boxes, random.Random(11))
    np.random.seed(11)
    _, t2 = aug.degrade_with_boxes(im, boxes, random.Random(11))
    assert t1 == t2


def test_degrade_with_boxes_full_image_box_stays_large_under_mild_distortion():
    # Rotation is capped at 8deg and perspective jitter at 4% of each dimension -- a box spanning
    # the whole image should still cover most of it after transform, not collapse to near-nothing.
    im = Image.new("RGB", (300, 200), (50, 50, 50))
    _, transformed = aug.degrade_with_boxes(im, [(0, 0, 300, 200)], random.Random(5))
    x0, y0, x1, y1 = transformed[0]
    area_ratio = ((x1 - x0) * (y1 - y0)) / (300 * 200)
    assert area_ratio > 0.6


def test_build_training_pairs_end_to_end_on_real_canvas_output(tmp_path):
    # Reuses Task 2.4's real output (apps/comics-ai/comics-multimodal/work/canvas/) if present --
    # skips gracefully if that stage hasn't been run yet in this environment.
    if not any(aug.DEFAULT_CANVAS_DIR.glob("*.gt.json")):
        import pytest

        pytest.skip("work/canvas/ not populated -- run render_canvas.py first")

    out_dir = tmp_path / "train_pairs"
    pairs = aug.build_training_pairs(aug.DEFAULT_CANVAS_DIR, out_dir, seed=1)
    assert len(pairs) > 0
    manifest = out_dir / "manifest.jsonl"
    assert manifest.is_file()

    sample = pairs[0]
    assert Path(sample.clean_png).is_file()
    assert Path(sample.degraded_png).is_file()
    clean_im = Image.open(sample.clean_png)
    x0, y0, x1, y1 = sample.bbox
    assert clean_im.size == (x1 - x0, y1 - y0)

    assert len(sample.region_bboxes) == len(sample.layer_indexes) == len(sample.kinds)
    crop_w, crop_h = x1 - x0, y1 - y0
    for (rx0, ry0, rx1, ry1) in sample.region_bboxes:
        assert 0 <= rx0 <= rx1 <= crop_w
        assert 0 <= ry0 <= ry1 <= crop_h
