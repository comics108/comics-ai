import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from import_lottie import _sample_layer, derive_z_depth, import_lottie_file


REAL_LOTTIE = (
    Path(__file__).resolve().parents[4]
    / "dataset/bhagavadgita/vaishnav/bhagavadgita/lottie_unzip/Mediation of the Bhagavat Gita"
    / "Mediation of the Bhagavat Gita_content/Mediation of the Bhagavat Gita.json"
)


def _static(value):
    return {"a": 0, "k": value}


def test_root_sweep_is_cancelled_once_not_added_twice_to_absolute_y():
    root = {
        "st": 0, "ip": 0, "sr": 1,
        "ks": {
            "a": _static([360, 100, 0]),
            "p": {"a": 1, "k": [
                {"t": 0, "s": [360, 100, 0]},
                {"t": 10, "s": [360, 0, 0]},
            ]},
            "s": _static([100, 100, 100]), "r": _static(0),
        },
    }
    layer = {
        "ind": 1,
        "ks": {
            "a": _static([10, 20, 0]), "p": _static([50, 200, 0]),
            "s": _static([100, 100, 100]), "r": _static(0),
        },
    }
    start = _sample_layer(layer, {1: layer}, root, 0, 0)
    end = _sample_layer(layer, {1: layer}, root, 10, 0)
    assert (start.x, start.y) == (40, 180)
    assert (end.x, end.y) == (40, 180)
    assert end.position == 100


def test_z_depth_formula_has_neutral_far_and_near_cases():
    from import_lottie import PointSample

    def sample(position, x, scale=1):
        return PointSample(0, position, x, 0, scale, scale, 0)

    camera = [sample(0, 0), sample(100, 100)]
    assert derive_z_depth([sample(0, 0), sample(100, 100)], camera, True) == 0
    assert derive_z_depth([sample(0, 0), sample(100, 50)], camera, False) == 1
    assert derive_z_depth([sample(0, 0), sample(100, 200)], camera, False) == -0.5


def test_real_lottie_extracts_all_three_scenes_and_canonical_camera_depth_data():
    imported = import_lottie_file(REAL_LOTTIE)
    assert imported.scene_count == 3
    assert imported.image_layer_count == 519
    assert imported.reference_layers == (
        "0_1/Layer 432 (ind=101)",
        "0_2/6 (ind=163)",
        "0_3/177 (ind=43)",
    )
    positions = [point["position"] for point in imported.camera_path]
    assert len(positions) == 19
    assert positions == sorted(set(positions))
    nonzero_depths = {asset.z_depth for asset in imported.assets if asset.z_depth != 0}
    assert len(nonzero_depths) > 2
    assert any(depth < 0 for depth in nonzero_depths)
    assert any(depth > 0 for depth in nonzero_depths)

    for asset in imported.assets:
        translations = [a for a in asset.animations if "TranslateAnim" in a["$type"]]
        assert translations
        assert translations[0]["start"] == translations[0]["end"]
        assert all(a["start"] <= a["end"] for a in translations)
