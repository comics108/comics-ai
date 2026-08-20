import io
import json
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from PIL import Image

from build_storyboard import ChapterStoryboard, StoryScene
from models import CanonicalChapter, SlokaSource
from package_comics import PackagingAsset, write_comics_archive
from validate_output import validate_archive_structure, validate_storyboard_citations


def _build_baseline_archive(tmp_path) -> Path:
    """One background + one real Russian balloon layer -- a minimal, structurally valid archive
    built through the real packager (not hand-crafted), used as the passing-fixture baseline and
    as the starting point for the mutated failing fixtures below."""
    assets = [
        PackagingAsset(
            kind="background", image=Image.new("RGBA", (1080, 400), "#eeeeee"),
            x=0, y=0, stem="background_000", contains_russian_text=False,
        ),
        PackagingAsset(
            kind="art", image=Image.new("RGBA", (936, 100), (10, 10, 10, 255)),
            x=72, y=72, stem="art_001", contains_russian_text=True,
        ),
        PackagingAsset(
            kind="balloon", image=Image.new("RGBA", (936, 150), (20, 20, 20, 255)),
            x=72, y=204, stem="balloon_002", contains_russian_text=True,
        ),
    ]
    path = tmp_path / "chapter.comics"
    write_comics_archive(path, 1080, 400, assets)
    return path


def _mutate_data_json(path: Path, mutate) -> Path:
    """Rewrites `path`'s data.json in place via `mutate(data: dict) -> None`, keeping every other
    ZIP member byte-identical -- lets each failing fixture change exactly one thing."""
    original = path.read_bytes()
    zin = zipfile.ZipFile(io.BytesIO(original))
    data = json.loads(zin.read("data.json").decode("utf-8"))
    mutate(data)

    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w") as zout:
        for name in zin.namelist():
            content = json.dumps(data).encode("utf-8") if name == "data.json" else zin.read(name)
            zout.writestr(name, content)
    path.write_bytes(out_buf.getvalue())
    return path


def _remove_member(path: Path, member_name: str) -> Path:
    original = path.read_bytes()
    zin = zipfile.ZipFile(io.BytesIO(original))
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w") as zout:
        for name in zin.namelist():
            if name != member_name:
                zout.writestr(name, zin.read(name))
    path.write_bytes(out_buf.getvalue())
    return path


def test_baseline_archive_passes_with_no_issues(tmp_path):
    path = _build_baseline_archive(tmp_path)
    result = validate_archive_structure(path, expected_verse_count=1)
    assert result.ok, result.issues


def test_corrupt_zip_is_rejected(tmp_path):
    path = tmp_path / "broken.comics"
    path.write_bytes(b"not a zip file at all")
    result = validate_archive_structure(path, expected_verse_count=1)
    assert not result.ok
    assert result.issues[0].check == "zip_open"


def test_wrong_verse_count_is_flagged(tmp_path):
    path = _build_baseline_archive(tmp_path)
    result = validate_archive_structure(path, expected_verse_count=2)
    assert any(i.check == "verse_count" for i in result.issues)


def test_missing_background_layer_is_flagged(tmp_path):
    path = _build_baseline_archive(tmp_path)

    def drop_background(data):
        data["layers"] = [l for l in data["layers"] if l["kind"] != "background"]

    _mutate_data_json(path, drop_background)
    result = validate_archive_structure(path, expected_verse_count=1)
    assert any(i.check == "layer_counts" for i in result.issues)


def test_russian_slot_regression_is_detected_on_real_generated_output(tmp_path):
    """The Plan's own explicitly-required regression test: asserts the check fires on a real
    archive whose balloon layer has been mutated back to the original buggy slot-0 placement,
    not just on the packager's in-memory data model (already covered in test_package_comics.py)."""
    path = _build_baseline_archive(tmp_path)

    def break_slot(data):
        for layer in data["layers"]:
            if layer["kind"] == "balloon":
                layer["images"] = [layer["images"][1], {}, {}]  # move Ru content back to slot 0

    _mutate_data_json(path, break_slot)
    result = validate_archive_structure(path, expected_verse_count=1)
    assert any(i.check == "russian_slot_regression" for i in result.issues)


def test_sounds_not_empty_is_flagged(tmp_path):
    path = _build_baseline_archive(tmp_path)
    _mutate_data_json(path, lambda data: data.update(sounds=["layers/oops.mp3"]))
    result = validate_archive_structure(path, expected_verse_count=1)
    assert any(i.check == "sounds_not_empty" for i in result.issues)


def test_out_of_bounds_layer_is_flagged(tmp_path):
    path = _build_baseline_archive(tmp_path)

    def push_offscreen(data):
        for layer in data["layers"]:
            if layer["kind"] == "balloon":
                layer["animations"][0]["x"] = 5000  # far past the 1080px canvas width

    _mutate_data_json(path, push_offscreen)
    result = validate_archive_structure(path, expected_verse_count=1)
    assert any(i.check == "out_of_bounds" for i in result.issues)


def test_missing_tile_is_flagged(tmp_path):
    path = _build_baseline_archive(tmp_path)
    _remove_member(path, "layers/balloon_002_1000_0_0.png")
    result = validate_archive_structure(path, expected_verse_count=1)
    assert any(i.check == "missing_tile" for i in result.issues)


def test_unsafe_path_member_is_flagged(tmp_path):
    path = _build_baseline_archive(tmp_path)
    original = path.read_bytes()
    zin = zipfile.ZipFile(io.BytesIO(original))
    out_buf = io.BytesIO()
    with zipfile.ZipFile(out_buf, "w") as zout:
        for name in zin.namelist():
            zout.writestr(name, zin.read(name))
        zout.writestr("../evil.png", b"x")
    path.write_bytes(out_buf.getvalue())
    result = validate_archive_structure(path, expected_verse_count=1)
    assert any(i.check == "unsafe_path" for i in result.issues)


def _sloka(order: int) -> SlokaSource:
    return SlokaSource(
        id=order, chapter_id=1, order=order, name=f"1.{order}", sanskrit="s", transcription="t",
        translation_ru="tr", comment_ru="", audio_ref="", sanskrit_audio_ref="",
    )


def test_storyboard_citations_within_the_chapter_pass():
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=(_sloka(1), _sloka(2)))
    storyboard = ChapterStoryboard(
        schema_version=1, mode="deterministic", model=None, prompt_version="n/a",
        chapter_summary_ru=None,
        scenes=(StoryScene("ch01-scene01", "x", None, (1, 2), (), None, None),),
        warnings=(), raw_model_output=None,
    )
    assert validate_storyboard_citations(chapter, storyboard).ok


def test_storyboard_citation_outside_the_chapter_is_flagged():
    """The concrete failure mode a hallucinating Task 2.2 (Ollama) storyboard could introduce."""
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=(_sloka(1),))
    storyboard = ChapterStoryboard(
        schema_version=1, mode="ollama", model="qwen2.5-coder:32b", prompt_version="v1",
        chapter_summary_ru=None,
        scenes=(StoryScene("ch01-scene01", "x", None, (1, 99), (), None, None),),  # 99 doesn't exist
        warnings=(), raw_model_output=None,
    )
    result = validate_storyboard_citations(chapter, storyboard)
    assert not result.ok
    assert result.issues[0].check == "citation_out_of_chapter"


def test_real_chapter_twelve_output_passes_structural_validation():
    """Real integration test against Task 6.1's own real chapter-12 output (re-generated here
    rather than depending on test execution order/leftover files from test_package_comics.py)."""
    import render_cards
    from layout_chapter import CANVAS_WIDTH, layout_chapter, layout_chapter_content
    from load_dataset import DATASET_DIR, load_book_one

    chapters = load_book_one(dataset_dir=DATASET_DIR, book_id=1)
    chapter_twelve = next(c for c in chapters if c.order == 12)

    try:
        content = [("art", render_cards.render_title_card(chapter_twelve.order, chapter_twelve.title, book_id=1))]
        for sloka in chapter_twelve.slokas:
            content.append(("balloon", render_cards.render_verse_card(sloka, chapter_twelve.order, book_id=1)))
        _, total_height = layout_chapter_content(content)
        background = render_cards.render_chapter_background(
            render_cards.theme_for_chapter(chapter_twelve.order), CANVAS_WIDTH, total_height
        )
        layout = layout_chapter(chapter_twelve, content, background)
    finally:
        render_cards.shutdown_browser()

    assets = [
        PackagingAsset(
            kind=a.kind, image=a.image, x=a.x, y=a.y, stem=f"{a.kind}_{i:03d}",
            contains_russian_text=a.kind in ("art", "balloon"),
        )
        for i, a in enumerate(layout.assets)
    ]

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        path = Path(tmp_dir) / "chapter_12.comics"
        write_comics_archive(path, layout.width, layout.height, assets)
        result = validate_archive_structure(path, expected_verse_count=len(chapter_twelve.slokas))

    assert result.ok, result.issues
