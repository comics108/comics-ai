import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from adapters.psd import recover_psd_layer, recover_psd_structure
from adapters.pdf import recover_pdf_structure
from adapters.lottie import recover_lottie_package
from adapters.comics import recover_comics_layer, recover_comics_structure


REAL_DRAWING = Path(__file__).resolve().parents[4] / "dataset/bhagavadgita/vaishnav/drawing"
REAL_LOTTIE_PACKAGE = (
    Path(__file__).resolve().parents[4]
    / "dataset/bhagavadgita/vaishnav/bhagavadgita/lottie_unzip/Mediation of the Bhagavat Gita"
)
REAL_COMICS = Path(__file__).resolve().parents[4] / "work/bhagavadgita/chapter_05.comics"


def test_real_psd_native_hierarchy_matches_reviewed_checkpoints():
    expected = {
        "5_1.psd": (9449, 7087, 5, 1),
        "5_2.psd": (9977, 8101, 32, 6),
        "app_BG._chiba5.psd": (4127, 26421, 419, 92),
    }

    recovered = {
        name: recover_psd_structure(REAL_DRAWING / name)
        for name in expected
    }

    for name, (width, height, descendants, groups) in expected.items():
        document = recovered[name]
        assert (document.width, document.height) == (width, height)
        assert len(document.nodes) == descendants
        assert sum(node.kind == "group" for node in document.nodes) == groups
        assert all(node.native_path and node.name for node in document.nodes)

    chapter = recovered["app_BG._chiba5.psd"]
    assert chapter.text_group_names == tuple(f"text{i}" for i in range(1, 16))
    assert {"5_1", "5_2_a", "5_2_b", "5_2_c"}.issubset(
        {node.name for node in chapter.nodes if node.kind == "group"}
    )


def test_real_psd_pixel_layer_recovers_rgba_and_bitmap_alpha_mask():
    recovered = recover_psd_layer(REAL_DRAWING / "5_1.psd", "0/1")

    assert recovered.name == "Layer 2"
    assert recovered.bbox == (1635, 2799, 1944, 3027)
    assert recovered.rgba.mode == "RGBA"
    assert recovered.rgba.size == (309, 228)
    assert recovered.bitmap_mask.mode == "L"
    assert recovered.bitmap_mask.size == recovered.rgba.size
    assert recovered.bitmap_mask.tobytes() == recovered.rgba.getchannel("A").tobytes()


def test_real_pdf_adapter_recovers_embedded_images_without_page_rendering():
    black_and_white = recover_pdf_structure(REAL_DRAWING / "All_Black-n-White.pdf")
    coloured = recover_pdf_structure(REAL_DRAWING / "All_Coloured.pdf")

    assert black_and_white.page_count == 12
    assert coloured.page_count == 6
    assert len(black_and_white.embedded_images) == 12
    assert len(coloured.embedded_images) == 6
    assert (black_and_white.embedded_images[0].width, black_and_white.embedded_images[0].height) == (21649, 2913)
    assert black_and_white.embedded_images[-1].x_ppi == 600
    assert coloured.embedded_images[0].encoding == "jpeg"


def test_real_lottie_adapter_recovers_native_and_language_audio_provenance():
    recovered = recover_lottie_package(REAL_LOTTIE_PACKAGE)

    assert (recovered.width, recovered.height, recovered.frame_rate) == (720, 1600, 60.0)
    assert recovered.root_layer_count == 3
    assert recovered.precomposition_count == 3
    assert recovered.referenced_image_count == 514
    assert recovered.translation_image_counts == (("en", 9), ("ru", 9))
    assert {path.name for path in recovered.audio_files} == {
        "BG_MusicLoop.aac", "OutroHit_Oneshot.aac",
    }
    assert recovered.semantic_scope_id == "scope-gita-dhyanam-nine-stanzas"
    assert recovered.camera_depth_authority == "derived_evidence_not_gold"
    assert all(layer.transform for layer in recovered.layers)


def test_real_comics_adapter_recovers_slots_transforms_and_selected_tiles():
    document = recover_comics_structure(REAL_COMICS)

    assert (document.width, document.height) == (1080, 20811)
    assert len(document.layers) == 32
    assert document.evidence_class == "runtime_reference_unapproved"
    title = document.layers[1]
    assert title.kind == "art"
    assert title.populated_slots == (1,)
    assert title.animations[0]["x"] == 72
    assert title.animations[0]["y"] == 72

    pixels = recover_comics_layer(REAL_COMICS, layer_index=1, slot=1)
    assert pixels.mode == "RGBA"
    assert pixels.size == (936, 200)
