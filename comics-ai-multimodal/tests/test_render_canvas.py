import json
import sys
import zipfile
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baloons_bridge  # noqa: E402
import render_canvas as rc  # noqa: E402


def _make_fixture_comics(path: Path) -> None:
    """A tiny, fully-controlled .comics fixture (no dataset dependency):
    - layer 0: 100x100 solid red background, no animations (rests at 0,0)
    - layer 1: 20x20 solid blue "character", TranslateAnim to (10, 10), no start (base position)
    - layer 2: 20x20 solid green, AlphaAnim with alpha omitted -> resting alpha 0 (invisible,
      must be excluded from both the composite and the ground-truth regions)
    """
    data = {
        "width": 100,
        "height": 100,
        "layers": [
            {
                "images": [{"file": "bg_{0}_{1}_{2}.png", "width": 100, "height": 100}, {}, {}],
                "animations": [],
            },
            {
                "images": [{"file": "chr_{0}_{1}_{2}.png", "width": 20, "height": 20}, {}, {}],
                "animations": [
                    {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "x": 10, "y": 10}
                ],
            },
            {
                "images": [{"file": "ghost_{0}_{1}_{2}.png", "width": 20, "height": 20}, {}, {}],
                "animations": [
                    {"$type": "Comics.Editor.Models.AlphaAnim, Comics.Editor", "type": 3}
                ],
            },
        ],
        "sounds": [],
    }

    bg = Image.new("RGBA", (100, 100), (255, 0, 0, 255))
    chr_img = Image.new("RGBA", (20, 20), (0, 0, 255, 255))
    ghost_img = Image.new("RGBA", (20, 20), (0, 255, 0, 255))

    def _png_bytes(im: Image.Image) -> bytes:
        import io

        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    with zipfile.ZipFile(path, mode="w") as zf:
        zf.writestr("data.json", json.dumps(data))
        zf.writestr("layers/bg_1000_0_0.png", _png_bytes(bg))
        zf.writestr("layers/chr_1000_0_0.png", _png_bytes(chr_img))
        zf.writestr("layers/ghost_1000_0_0.png", _png_bytes(ghost_img))


def test_composite_positions_and_excludes_invisible_layer(tmp_path):
    fixture = tmp_path / "fixture.comics"
    _make_fixture_comics(fixture)
    archive = baloons_bridge.ComicsArchive(fixture)

    out_dir = tmp_path / "canvas_out"
    ref = rc.render_canvas_reference(archive, out_dir)

    assert ref.width == 100
    assert ref.height == 100
    # Only the background + character layers are visible at rest; the alpha=0 "ghost" layer must
    # be excluded entirely.
    assert len(ref.regions) == 2
    assert {r.layer_index for r in ref.regions} == {0, 1}

    composite = Image.open(ref.composite_png).convert("RGBA")
    assert composite.size == (100, 100)
    # Background red, away from the character
    assert composite.getpixel((5, 5))[:3] == (255, 0, 0)
    # Character blue square translated to (10,10)-(30,30)
    assert composite.getpixel((15, 15))[:3] == (0, 0, 255)
    assert composite.getpixel((35, 35))[:3] == (255, 0, 0)  # back to background outside the square

    char_region = next(r for r in ref.regions if r.layer_index == 1)
    assert char_region.bbox == (10, 10, 30, 30)
    assert char_region.kind_source == "inferred_heuristic"


def test_ground_truth_is_json_serializable(tmp_path):
    fixture = tmp_path / "fixture.comics"
    _make_fixture_comics(fixture)
    archive = baloons_bridge.ComicsArchive(fixture)
    ref = rc.render_canvas_reference(archive, tmp_path / "canvas_out2")
    # Must round-trip through json without error (dataclasses -> dict -> json)
    json.dumps(ref.to_jsonable())


def test_runs_on_real_dataset_file_and_matches_declared_dimensions():
    files = baloons_bridge.find_comics_files()
    target = next(f for f in files if f.name == "8a89f7d689fb441ea280cd782276bd7a.comics")
    archive = baloons_bridge.ComicsArchive(target)
    data = archive.read_data_json()

    ref = rc.render_canvas_reference(archive, Path(baloons_bridge.__file__).parent.parent / "work" / "canvas_test")
    assert ref.width == data["width"]
    assert ref.height == data["height"]
    assert 0 < len(ref.regions) <= len(data["layers"])
    assert Path(ref.composite_png).is_file()
