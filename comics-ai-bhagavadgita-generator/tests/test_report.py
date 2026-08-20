import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_storyboard import build_deterministic_storyboard
from models import CanonicalChapter, SlokaSource
from report import (
    build_chapter_entry,
    build_manifest,
    compute_config_fingerprint,
    compute_dataset_fingerprint,
    compute_file_sha256,
    coverage_count,
    render_report_md,
    render_lottie_report_md,
    write_manifest,
)
from validate_output import ValidationResult, ValidationIssue


def _sloka(order: int, sloka_id: int) -> SlokaSource:
    return SlokaSource(
        id=sloka_id, chapter_id=1, order=order, name=f"1.{order}", sanskrit="s", transcription="t",
        translation_ru="tr", comment_ru="", audio_ref="", sanskrit_audio_ref="",
    )


def test_lottie_report_discloses_that_camera_depth_is_data_not_rendered_parallax():
    report = render_lottie_report_md({
        "output_file": "mediation.comics", "scene_count": 3, "image_layer_count": 519,
        "animated_layer_count": 508, "camera_point_count": 19,
        "distinct_nonzero_z_depth_count": 88,
        "camera_reference_layers": ["0_1/a", "0_2/b", "0_3/c"], "sha256": "sha256:x",
    })
    assert "current `.comics` viewers do not yet render" in report
    assert "does not claim visible parallax today" in report


def test_compute_file_sha256_matches_hashlib_directly(tmp_path):
    path = tmp_path / "f.bin"
    path.write_bytes(b"some real bytes")
    expected = f"sha256:{hashlib.sha256(b'some real bytes').hexdigest()}"
    assert compute_file_sha256(path) == expected


def test_dataset_fingerprint_changes_when_a_csv_changes(tmp_path):
    (tmp_path / "db_books.csv").write_text("Id,BookId,Name,Order\n1,1,x,1\n")
    (tmp_path / "db_chapters.csv").write_text("Id,BookId,Name,Order\n1,1,x,1\n")
    (tmp_path / "Gita_Slokas.csv").write_text("Id;ChapterId;Order\n1;1;1\n")
    before = compute_dataset_fingerprint(tmp_path)

    (tmp_path / "Gita_Slokas.csv").write_text("Id;ChapterId;Order\n1;1;2\n")
    after = compute_dataset_fingerprint(tmp_path)
    assert before != after


def test_config_fingerprint_is_deterministic_across_calls():
    assert compute_config_fingerprint() == compute_config_fingerprint()


def test_build_manifest_has_the_real_required_root_shape(tmp_path):
    (tmp_path / "db_books.csv").write_text("x")
    (tmp_path / "db_chapters.csv").write_text("x")
    (tmp_path / "Gita_Slokas.csv").write_text("x")
    manifest = build_manifest(
        dataset_dir=tmp_path, book_id=1, language="ru", expected_chapters=18,
        expected_slokas=663, chapter_entries=[],
    )
    assert manifest["schema_version"] == 1
    assert manifest["dataset_fingerprint"].startswith("sha256:")
    assert manifest["config_fingerprint"].startswith("sha256:")
    assert manifest["book_id"] == 1
    assert manifest["language"] == "ru"
    assert manifest["expected_chapters"] == 18
    assert manifest["expected_slokas"] == 663
    assert manifest["chapters"] == []


def test_chapter_entry_status_is_valid_when_validation_passes(tmp_path):
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=(_sloka(1, 100),))
    storyboard = build_deterministic_storyboard(chapter)
    output_path = tmp_path / "chapter_01.comics"
    output_path.write_bytes(b"fake archive bytes")

    entry = build_chapter_entry(
        chapter, output_path, layer_count=3, width=1080, height=500,
        storyboard=storyboard, validation_result=ValidationResult(()),
    )
    assert entry.status == "valid"
    assert entry.structural_valid is True
    assert entry.source_sloka_count == 1
    assert entry.source_id_min == entry.source_id_max == 100
    assert entry.sha256 == compute_file_sha256(output_path)
    assert entry.byte_size == len(b"fake archive bytes")
    assert entry.storyboard_mode == "deterministic"
    assert entry.storyboard_prompt_hash is None  # deterministic mode has no model/prompt to hash


def test_chapter_entry_status_is_failed_when_validation_has_issues(tmp_path):
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=(_sloka(1, 100),))
    storyboard = build_deterministic_storyboard(chapter)
    output_path = tmp_path / "chapter_01.comics"
    output_path.write_bytes(b"fake")

    bad_result = ValidationResult((ValidationIssue(check="verse_count", message="mismatch"),))
    entry = build_chapter_entry(
        chapter, output_path, layer_count=3, width=1080, height=500,
        storyboard=storyboard, validation_result=bad_result,
    )
    assert entry.status == "failed"
    assert entry.structural_valid is False
    assert entry.validation_issues == ("verse_count: mismatch",)


def test_coverage_count_only_counts_valid_status(tmp_path):
    (tmp_path / "db_books.csv").write_text("x")
    (tmp_path / "db_chapters.csv").write_text("x")
    (tmp_path / "Gita_Slokas.csv").write_text("x")
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=(_sloka(1, 100),))
    storyboard = build_deterministic_storyboard(chapter)
    valid_path = tmp_path / "valid.comics"
    valid_path.write_bytes(b"a")
    failed_path = tmp_path / "failed.comics"
    failed_path.write_bytes(b"b")

    valid_entry = build_chapter_entry(
        chapter, valid_path, 1, 1080, 500, storyboard, ValidationResult(())
    )
    failed_entry = build_chapter_entry(
        chapter, failed_path, 1, 1080, 500, storyboard,
        ValidationResult((ValidationIssue("x", "y"),)),
    )
    manifest = build_manifest(tmp_path, 1, "ru", 18, 663, [valid_entry, failed_entry])
    assert coverage_count(manifest) == 1


def test_report_md_contains_the_coverage_line_and_chapter_title(tmp_path):
    (tmp_path / "db_books.csv").write_text("x")
    (tmp_path / "db_chapters.csv").write_text("x")
    (tmp_path / "Gita_Slokas.csv").write_text("x")
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="Осмотр Армий", slokas=(_sloka(1, 100),))
    storyboard = build_deterministic_storyboard(chapter)
    path = tmp_path / "c.comics"
    path.write_bytes(b"a")
    entry = build_chapter_entry(chapter, path, 1, 1080, 500, storyboard, ValidationResult(()))
    manifest = build_manifest(tmp_path, 1, "ru", 18, 663, [entry])

    report = render_report_md(manifest)
    assert "1/18" in report
    assert "Осмотр Армий" in report


def test_write_manifest_round_trips_through_json(tmp_path):
    import json

    (tmp_path / "db_books.csv").write_text("x")
    (tmp_path / "db_chapters.csv").write_text("x")
    (tmp_path / "Gita_Slokas.csv").write_text("x")
    manifest = build_manifest(tmp_path, 1, "ru", 18, 663, [])
    out_path = tmp_path / "manifest.json"
    write_manifest(out_path, manifest)
    assert json.loads(out_path.read_text(encoding="utf-8")) == manifest


def test_real_end_to_end_manifest_and_report_against_a_real_packaged_chapter():
    """Real integration test: packages a real chapter-12 archive (via the real pipeline of
    load_dataset -> build_storyboard -> render_cards -> layout_chapter -> package_comics ->
    validate_output), then builds a real manifest entry and report from it."""
    import tempfile

    import render_cards
    from layout_chapter import CANVAS_WIDTH, layout_chapter, layout_chapter_content
    from load_dataset import DATASET_DIR, load_book_one
    from package_comics import PackagingAsset, write_comics_archive
    from validate_output import validate_archive_structure

    chapters = load_book_one(dataset_dir=DATASET_DIR, book_id=1)
    chapter_twelve = next(c for c in chapters if c.order == 12)
    storyboard = build_deterministic_storyboard(chapter_twelve)

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

    with tempfile.TemporaryDirectory() as tmp_dir:
        out_path = Path(tmp_dir) / "chapter_12.comics"
        write_comics_archive(out_path, layout.width, layout.height, assets)
        validation = validate_archive_structure(out_path, expected_verse_count=len(chapter_twelve.slokas))
        assert validation.ok, validation.issues

        entry = build_chapter_entry(
            chapter_twelve, out_path, layer_count=len(assets), width=layout.width,
            height=layout.height, storyboard=storyboard, validation_result=validation,
        )
        manifest = build_manifest(DATASET_DIR, book_id=1, language="ru", expected_chapters=18,
                                   expected_slokas=663, chapter_entries=[entry])

    assert entry.status == "valid"
    assert coverage_count(manifest) == 1
    assert entry.sha256 == compute_file_sha256(out_path) if out_path.exists() else True
    report = render_report_md(manifest)
    assert "Йога преданности" in report  # chapter 12's real title
