import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_storyboard import build_deterministic_storyboard
from models import CanonicalChapter, SlokaSource


def _sloka(order: int) -> SlokaSource:
    return SlokaSource(
        id=order, chapter_id=1, order=order, name=f"1.{order}", sanskrit="s",
        transcription="t", translation_ru="tr", comment_ru="", audio_ref="", sanskrit_audio_ref="",
    )


def test_deterministic_storyboard_produces_one_scene_covering_all_real_sloka_orders():
    chapter = CanonicalChapter(
        book_id=1, chapter_id=1, order=1, title="Осмотр Армий",
        slokas=(_sloka(1), _sloka(2), _sloka(3)),
    )
    storyboard = build_deterministic_storyboard(chapter)
    assert storyboard.mode == "deterministic"
    assert storyboard.model is None
    assert storyboard.chapter_summary_ru is None  # no synthetic summary, per design
    assert len(storyboard.scenes) == 1
    scene = storyboard.scenes[0]
    assert scene.source_sloka_orders == (1, 2, 3)
    assert scene.title == "Осмотр Армий"
    assert scene.summary_ru is None
    assert scene.characters == ()  # no invented characters
    assert storyboard.warnings == ()


def test_deterministic_storyboard_on_empty_chapter_emits_no_scenes_not_a_fabricated_one():
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="Empty", slokas=())
    storyboard = build_deterministic_storyboard(chapter)
    assert storyboard.scenes == ()
    assert len(storyboard.warnings) == 1


def test_scene_id_is_stable_and_chapter_specific():
    chapter5 = CanonicalChapter(book_id=1, chapter_id=5, order=5, title="x", slokas=(_sloka(1),))
    storyboard = build_deterministic_storyboard(chapter5)
    assert storyboard.scenes[0].scene_id == "ch05-scene01"


def test_real_dataset_chapter_one_produces_a_valid_deterministic_storyboard():
    from load_dataset import DATASET_DIR, load_book_one

    chapters = load_book_one(dataset_dir=DATASET_DIR, book_id=1)
    chapter_one = next(c for c in chapters if c.order == 1)
    storyboard = build_deterministic_storyboard(chapter_one)
    assert len(storyboard.scenes) == 1
    assert storyboard.scenes[0].source_sloka_orders == tuple(s.order for s in chapter_one.slokas)
