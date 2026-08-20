import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest
from PIL import Image

from package_comics import (
    PackagingAsset,
    build_archive_bytes,
    build_data_json,
    build_tiles,
    write_comics_archive,
)


def _asset(kind, stem, x=0, y=0, size=(100, 60), russian=False) -> PackagingAsset:
    return PackagingAsset(
        kind=kind, image=Image.new("RGBA", size, "#123456"), x=x, y=y, stem=stem,
        contains_russian_text=russian,
    )


def test_root_json_has_the_real_required_keys_and_empty_sounds():
    data = build_data_json(1080, 500, [_asset("background", "bg")])
    assert set(data.keys()) == {"width", "height", "layers", "sounds"}
    assert data["width"] == 1080
    assert data["height"] == 500
    assert data["sounds"] == []


def test_language_neutral_asset_lands_in_slot_0():
    data = build_data_json(1080, 500, [_asset("art", "psd_panel", russian=False)])
    images = data["layers"][0]["images"]
    assert images[0].get("file") == "psd_panel_{0}_{1}_{2}.png"
    assert images[1] == {}
    assert images[2] == {}


def test_russian_text_asset_lands_in_slot_1_not_slot_0():
    """Dedicated regression test for the v0.2 correction (Specifications: Cultures.cs's real
    {En=0, Ru=1, Hi=2} enum) -- must never silently regress back to the original backwards slot-0
    bug."""
    data = build_data_json(1080, 500, [_asset("balloon", "verse_001", russian=True)])
    images = data["layers"][0]["images"]
    assert images[0] == {}, "slot 0 (En) must stay empty for Russian-only content"
    assert images[1].get("file") == "verse_001_{0}_{1}_{2}.png"
    assert images[2] == {}, "slot 2 (Hi) must stay empty"


def test_layer_has_an_explicit_static_translate_anim_with_the_asset_position():
    data = build_data_json(1080, 500, [_asset("art", "title", x=72, y=1234)])
    anims = data["layers"][0]["animations"]
    assert anims == [
        {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "x": 72, "y": 1234}
    ]
    assert data["layers"][0]["kind"] == "art"


def test_animated_depth_layer_and_camera_path_use_the_canonical_additive_schema():
    anims = (
        {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "start": 10, "end": 10,
         "x": 5, "y": 8},
        {"$type": "Comics.Editor.Models.TranslateAnim, Comics.Editor", "start": 10, "end": 20,
         "x": 7, "y": 12},
    )
    asset = PackagingAsset(
        kind="art", image=Image.new("RGBA", (10, 10)), x=5, y=8, stem="animated",
        contains_russian_text=False, animations=anims, z_depth=0.5,
    )
    path = [{"position": 10, "x": 1.25, "y": 2.5}, {"position": 20, "x": 3, "y": 5}]
    data = build_data_json(
        720, 1600, [asset], camera_path=path,
        preferred_viewport_width=720, preferred_viewport_height=1600,
    )
    assert data["layers"][0]["animations"] == list(anims)
    assert data["layers"][0]["zDepth"] == 0.5
    assert data["cameraPath"] == path
    assert data["preferredViewportWidth"] == 720
    assert data["preferredViewportHeight"] == 1600


def test_camera_path_rejects_duplicate_or_decreasing_positions():
    with pytest.raises(ValueError, match="strictly increasing"):
        build_data_json(720, 1600, [], camera_path=[
            {"position": 10, "x": 0, "y": 0},
            {"position": 10, "x": 1, "y": 1},
        ])


def test_duplicate_stems_are_rejected():
    with pytest.raises(ValueError, match="duplicate asset stems"):
        build_data_json(1080, 500, [_asset("art", "dup"), _asset("balloon", "dup")])


def test_path_traversal_stem_is_rejected():
    with pytest.raises(ValueError, match="unsafe asset stem"):
        build_data_json(1080, 500, [_asset("art", "../evil")])


def test_build_tiles_rejects_colliding_tile_names_across_assets():
    small = _asset("art", "same", size=(10, 10))
    with pytest.raises(ValueError, match="tile name collision"):
        build_tiles([small, small])


def test_archive_bytes_are_deterministic_across_repeated_builds():
    assets = [_asset("background", "bg", size=(600, 600)), _asset("balloon", "verse", russian=True)]
    first = build_archive_bytes(1080, 700, assets)
    second = build_archive_bytes(1080, 700, assets)
    assert first == second


def test_zip_entry_order_is_data_json_then_lexically_sorted_tiles():
    assets = [_asset("balloon", "zzz_last", size=(10, 10)), _asset("art", "aaa_first", size=(10, 10))]
    archive_bytes = build_archive_bytes(1080, 700, assets)
    import io

    names = zipfile.ZipFile(io.BytesIO(archive_bytes)).namelist()
    assert names[0] == "data.json"
    assert names[1:] == sorted(names[1:])


def test_write_comics_archive_leaves_no_staging_file_and_writes_a_valid_zip(tmp_path):
    out_path = tmp_path / "chapter.comics"
    write_comics_archive(out_path, 1080, 500, [_asset("background", "bg", size=(1080, 500))])
    assert out_path.exists()
    assert not (tmp_path / "chapter.comics.staging").exists()
    with zipfile.ZipFile(out_path) as zf:
        assert zf.testzip() is None  # valid, uncorrupted zip


def test_real_end_to_end_package_and_reopen_chapter_twelve():
    """Real integration test: renders chapter 12's actual title card and all 16 real verse cards
    (the smallest real chapter, to keep this bounded), lays them out, packages a real archive, and
    reopens it -- verifying the exact Russian-slot placement against real content, not a fixture."""
    import render_cards
    from layout_chapter import CANVAS_WIDTH, layout_chapter, layout_chapter_content
    from load_dataset import DATASET_DIR, load_book_one

    chapters = load_book_one(dataset_dir=DATASET_DIR, book_id=1)
    chapter_twelve = next(c for c in chapters if c.order == 12)
    assert len(chapter_twelve.slokas) == 16  # real, previously-verified fact

    try:
        title_image = render_cards.render_title_card(chapter_twelve.order, chapter_twelve.title, book_id=1)
        content = [("art", title_image)]
        for sloka in chapter_twelve.slokas:
            content.append(("balloon", render_cards.render_verse_card(sloka, chapter_twelve.order, book_id=1)))

        _, total_height = layout_chapter_content(content)
        background = render_cards.render_chapter_background(
            render_cards.theme_for_chapter(chapter_twelve.order), CANVAS_WIDTH, total_height
        )
        layout = layout_chapter(chapter_twelve, content, background)
    finally:
        render_cards.shutdown_browser()

    packaging_assets = []
    for index, layout_asset in enumerate(layout.assets):
        packaging_assets.append(
            PackagingAsset(
                kind=layout_asset.kind,
                image=layout_asset.image,
                x=layout_asset.x,
                y=layout_asset.y,
                stem=f"{layout_asset.kind}_{index:03d}",
                contains_russian_text=layout_asset.kind in ("art", "balloon"),
            )
        )

    out_path = Path(__file__).resolve().parent.parent / "_tmp_test_output" / "chapter_12.comics"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        write_comics_archive(out_path, layout.width, layout.height, packaging_assets)

        with zipfile.ZipFile(out_path) as zf:
            data = json.loads(zf.read("data.json"))
            assert data["width"] == layout.width
            assert data["height"] == layout.height
            assert len(data["layers"]) == 1 + 1 + 16  # background + title + 16 real verse cards

            background_layer = data["layers"][0]
            assert background_layer["images"][0].get("file")  # language-neutral -> slot 0
            assert background_layer["images"][1] == {}

            first_verse_layer = data["layers"][2]
            assert first_verse_layer["kind"] == "balloon"
            assert first_verse_layer["images"][0] == {}
            assert first_verse_layer["images"][1].get("file")  # Russian -> slot 1, real regression check

            # every referenced tile file actually exists in the archive
            names = set(zf.namelist())
            for layer in data["layers"]:
                for image in layer["images"]:
                    if image:
                        template = image["file"]
                        assert f"layers/{template.format(1000, 0, 0)}" in names
    finally:
        out_path.unlink(missing_ok=True)
