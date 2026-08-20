import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from PIL import Image

from tile_assets import retile_image, stitch_tiles, tile_filename, tile_grid


def test_tile_filename_matches_the_real_dataset_convention():
    # Verified this session against a real archive's actual tile name:
    # work/comics-ai-multimodal/output/20260731_154003_p0.comics -> "layers/r0_1000_0_0.png"
    assert tile_filename("r0", 0, 0) == "r0_1000_0_0.png"


def test_tile_grid_covers_an_exact_multiple_of_the_tile_size():
    grid = tile_grid(1024, 512)
    assert len(grid) == 2
    assert [(col, row) for col, row, _ in grid] == [(0, 0), (1, 0)]
    assert [box for _, _, box in grid] == [(0, 0, 512, 512), (512, 0, 1024, 512)]


def test_tile_grid_clips_edge_tiles_instead_of_padding():
    grid = tile_grid(600, 600)
    assert len(grid) == 4  # ceil(600/512) == 2 per axis
    boxes = {(col, row): box for col, row, box in grid}
    assert boxes[(1, 1)] == (512, 512, 600, 600)  # clipped, not padded to 1024x1024


def test_retile_then_stitch_reconstructs_a_byte_identical_image():
    original = Image.new("RGBA", (700, 550))
    for y in range(original.height):
        for x in range(original.width):
            original.putpixel((x, y), ((x * 3) % 256, (y * 5) % 256, (x + y) % 256, 255))

    tiles = retile_image(original, "verse")
    expected_names = {tile_filename("verse", col, row) for col, row, _ in tile_grid(700, 550)}
    assert set(tiles.keys()) == expected_names

    stitched = stitch_tiles(tiles, "verse", 700, 550)
    assert stitched.tobytes() == original.convert("RGBA").tobytes()


def test_stitch_raises_on_a_missing_tile():
    import pytest

    tiles = retile_image(Image.new("RGBA", (600, 600)), "art")
    del tiles["art_1000_1_1.png"]
    with pytest.raises(FileNotFoundError):
        stitch_tiles(tiles, "art", 600, 600)
