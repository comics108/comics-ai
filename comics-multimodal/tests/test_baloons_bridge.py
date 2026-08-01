import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import baloons_bridge  # noqa: E402

# Same known-good sample used by comics-ai-baloons' own test_tiling.py. The real dataset is nested
# (dataset/boranko/mahabharata/book1/comics_interactive/*.comics), not flat -- located via rglob
# rather than a hardcoded path, since a second book/chapter could nest differently.
SAMPLE_FILE = next(
    p
    for p in baloons_bridge.find_comics_files()
    if p.name == "8a89f7d689fb441ea280cd782276bd7a.comics"
)


def _find_b1_eng_slot():
    archive = baloons_bridge.ComicsArchive(SAMPLE_FILE)
    data = archive.read_data_json()
    for layer in data["layers"]:
        for img in layer.get("images", []):
            if img.get("file", "").startswith("b1_eng_"):
                return archive, img
    raise AssertionError("expected b1_eng_* layer in sample file")


def test_repo_root_and_dataset_dir_resolve_correctly():
    assert (baloons_bridge.REPO_ROOT / "dataset").is_dir()
    assert baloons_bridge.DATASET_DIR.is_dir()
    assert SAMPLE_FILE.is_file()


def test_find_comics_files_locates_all_27_dataset_files():
    found = baloons_bridge.find_comics_files()
    assert len(found) == 27


def test_bridge_stitch_matches_direct_comics_ai_baloons_call():
    # Direct call into comics-ai-baloons' own tiling module (already on sys.path via the bridge).
    import tiling as direct_tiling  # noqa: E402

    archive, img_meta = _find_b1_eng_slot()
    via_bridge = baloons_bridge.stitch_image(
        archive, img_meta["file"], img_meta["width"], img_meta["height"]
    )
    direct = direct_tiling.stitch_image(
        archive, img_meta["file"], img_meta["width"], img_meta["height"]
    )
    assert via_bridge.size == direct.size == (img_meta["width"], img_meta["height"])
    assert via_bridge.tobytes() == direct.tobytes()


def test_bridge_write_comics_is_the_same_object_as_direct_import():
    import comics_io as direct_comics_io  # noqa: E402

    assert baloons_bridge.write_comics is direct_comics_io.write_comics
    assert baloons_bridge.ComicsArchive is direct_comics_io.ComicsArchive


def test_baloons_models_py_and_our_own_segmenter_models_package_dont_collide():
    # comics-ai-baloons/scripts/models.py (single file, shared dataclasses: MatchResult, OcrResult,
    # ...) originally collided by name with our own scripts/models/ package -- fixed by renaming
    # ours to scripts/segmenter_models/ (Phase 5, discovered when align_photo.py needed to import
    # comics-ai-baloons' match.py, which itself imports `from models import MatchResult,
    # OcrResult`). Both must now be independently importable in the same process.
    import models  # noqa: E402 -- comics-ai-baloons' single-file models.py, via the bridge's sys.path

    assert not hasattr(models, "__path__"), "'models' should resolve to comics-ai-baloons' single-file module, not a package"
    assert hasattr(models, "MatchResult")

    from segmenter_models.unet_baseline import UNetBaseline  # noqa: E402, F401 -- must not raise
