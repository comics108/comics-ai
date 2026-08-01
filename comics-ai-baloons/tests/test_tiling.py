import io
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import tiling

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"

# Known-good sample established during Requirements investigation: b1_eng in this file is a
# 648x152, 2-tile-wide balloon layer.
SAMPLE_FILE = DATASET_DIR / "8a89f7d689fb441ea280cd782276bd7a.comics"


def _find_b1_eng_slot():
    archive = comics_io.ComicsArchive(SAMPLE_FILE)
    data = archive.read_data_json()
    for layer in data["layers"]:
        for img in layer.get("images", []):
            if img.get("file", "").startswith("b1_eng_"):
                return archive, img
    raise AssertionError("expected b1_eng_* layer in sample file")


def test_tile_grid_matches_known_layout():
    # 648 wide -> ceil(648/512)=2 cols; 152 tall -> ceil(152/512)=1 row
    grid = tiling.tile_grid(648, 152)
    assert {(c, r) for c, r, _ in grid} == {(0, 0), (1, 0)}


def test_stitch_produces_declared_size():
    archive, img_meta = _find_b1_eng_slot()
    stitched = tiling.stitch_image(archive, img_meta["file"], img_meta["width"], img_meta["height"])
    assert stitched.size == (img_meta["width"], img_meta["height"])


def test_stitch_retile_round_trip_pixel_identical():
    archive, img_meta = _find_b1_eng_slot()
    stitched = tiling.stitch_image(archive, img_meta["file"], img_meta["width"], img_meta["height"])

    retiled = tiling.retile_image(stitched, img_meta["file"])
    expected_names = {
        tiling.tile_filename(img_meta["file"], c, r)
        for c, r, _ in tiling.tile_grid(img_meta["width"], img_meta["height"])
    }
    assert set(retiled.keys()) == expected_names

    # Re-stitch from the newly-produced tiles (in-memory) and confirm pixel-identical to the
    # original stitch -- this is the actual round-trip invariant that matters.
    canvas = Image.new("RGBA", (img_meta["width"], img_meta["height"]))
    for name, png_bytes in retiled.items():
        # filenames are "<basename>_1000_<col>_<row>.png" -- parse col/row back out
        stem = name.rsplit(".", 1)[0]
        parts = stem.split("_")
        col, row = int(parts[-2]), int(parts[-1])
        tile_img = Image.open(io.BytesIO(png_bytes)).convert("RGBA")
        canvas.paste(tile_img, (col * tiling.TILE_SIZE, row * tiling.TILE_SIZE))

    assert canvas.tobytes() == stitched.tobytes()


def test_edge_tile_is_clipped_not_padded():
    # 648 wide, tile 2 (col=1) should be 648-512=136px wide, not 512.
    grid = tiling.tile_grid(648, 152)
    edge = [box for c, r, box in grid if c == 1][0]
    assert edge[2] - edge[0] == 648 - 512
