import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baloons_bridge  # noqa: E402
import resting_position as rp  # noqa: E402

# Real fixtures, verbatim from the dataset (see flows/sdd-comics-ai-multimodal/
# 04-implementation-log.md for provenance) -- not synthesized, to catch real-world surprises.

# 8a89f7d689fb441ea280cd782276bd7a.comics (episode 21, ambas_plea): a 3-way crossfade layer with
# multi-keyframe Translate + Alpha.
CROSSFADE_ANIMS = [
    {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "y": 498},
    {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "x": 134, "y": -165, "end": 2361},
    {"$type": "Comics.Editor.Models.AlphaAnim, Comics.Editor", "type": 3, "alpha": 0.2},
    {"$type": "Comics.Editor.Models.AlphaAnim, Comics.Editor", "type": 3, "alpha": 1.0, "end": 446},
]

# 096e28e97ad843e9bae94902eb85755d.comics, layer 18: Translate + fade-in Alpha + Rotate.
ROTATE_ANIMS = [
    {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "y": 4109},
    {"$type": "Comics.Editor.Models.AlphaAnim, Comics.Editor", "type": 3},
    {
        "$type": "Comics.Editor.Models.AlphaAnim, Comics.Editor",
        "type": 3,
        "alpha": 1.0,
        "start": 2766,
        "end": 3876,
    },
    {
        "$type": "Comics.Editor.Models.RotateAnim, Comics.Editor",
        "type": 1,
        "pivotX": 0.5,
        "pivotY": 0.5,
    },
    {
        "$type": "Comics.Editor.Models.RotateAnim, Comics.Editor",
        "type": 1,
        "angle": 3.0,
        "pivotX": 0.5,
        "pivotY": 0.5,
        "start": 3209,
        "end": 3601,
    },
]

# Same file, layer 33: Translate + Scale, deliberately no AlphaAnim at all.
SCALE_ANIMS = [
    {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "x": 250, "y": 8575},
    {
        "$type": "Comics.Editor.Models.ScaleAnim, Comics.Editor",
        "type": 2,
        "scaleX": 1.0,
        "scaleY": 1.0,
        "pivotX": 0.5,
        "pivotY": 0.5,
    },
    {
        "$type": "Comics.Editor.Models.ScaleAnim, Comics.Editor",
        "type": 2,
        "scaleX": 1.08,
        "scaleY": 1.08,
        "pivotX": 0.5,
        "pivotY": 0.5,
        "start": 7103,
        "end": 8311,
    },
]


def test_crossfade_resolves_to_last_translate_and_alpha_keyframe():
    t = rp.resolve_resting_transform(CROSSFADE_ANIMS)
    assert (t.x, t.y) == (134, -165)
    assert t.alpha == 1.0
    # No ScaleAnim/RotateAnim present at all -> Init() defaults
    assert (t.scale_x, t.scale_y) == (1.0, 1.0)
    assert (t.scale_pivot_x, t.scale_pivot_y) == (0.5, 0.5)
    assert t.angle == 0.0
    assert (t.rotate_pivot_x, t.rotate_pivot_y) == (0.5, 0.5)


def test_rotate_layer_resolves_angle_and_pivot():
    t = rp.resolve_resting_transform(ROTATE_ANIMS)
    assert (t.x, t.y) == (0, 4109)  # x never set in any TranslateAnim keyframe -> 0
    assert t.alpha == 1.0
    assert t.angle == 3.0
    assert (t.rotate_pivot_x, t.rotate_pivot_y) == (0.5, 0.5)


def test_scale_layer_with_no_alpha_anim_defaults_to_fully_visible():
    t = rp.resolve_resting_transform(SCALE_ANIMS)
    assert (t.x, t.y) == (250, 8575)
    assert (t.scale_x, t.scale_y) == (1.08, 1.08)
    assert (t.scale_pivot_x, t.scale_pivot_y) == (0.5, 0.5)
    # No AlphaAnim entries at all for this layer -> Init() default alpha = 1 (fully visible),
    # NOT 0 -- this is the exact distinction the module's docstring calls out.
    assert t.alpha == 1.0


def test_empty_animations_list_uses_all_init_defaults():
    t = rp.resolve_resting_transform([])
    assert (t.x, t.y) == (0, 0)
    assert (t.scale_x, t.scale_y) == (1.0, 1.0)
    assert (t.scale_pivot_x, t.scale_pivot_y) == (0.5, 0.5)
    assert t.angle == 0.0
    assert (t.rotate_pivot_x, t.rotate_pivot_y) == (0.5, 0.5)
    assert t.alpha == 1.0


def test_out_of_order_start_values_are_sorted_not_trusted_as_array_order():
    # Deliberately reversed array order vs. Start -- resolver must sort by start, mirroring
    # Anim.FindNearest's own `.OrderBy(x => x.Start)`.
    anims = [
        {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "x": 999, "start": 500},
        {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "x": 1, "start": 0},
    ]
    t = rp.resolve_resting_transform(anims)
    assert t.x == 999  # the one with the later start wins, regardless of array position


def test_resolves_without_crashing_across_all_real_dataset_layers():
    # Integration smoke test: every real layer's animations[] parses without error and produces
    # a RestingTransform (values not asserted here -- just no crashes/exceptions across all 4594
    # layers in the real dataset).
    count = 0
    for f in baloons_bridge.find_comics_files():
        archive = baloons_bridge.ComicsArchive(f)
        data = archive.read_data_json()
        for layer in data["layers"]:
            rp.resolve_resting_transform(layer.get("animations", []))
            count += 1
    assert count == 4594
