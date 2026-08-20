import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from load_dataset import (
    DATASET_DIR,
    DatasetIntegrityError,
    load_book_one,
    verify_dataset_integrity,
)


def _write_fixture_dataset(
    tmp_path: Path,
    chapters_rows: list[str],
    slokas_rows: list[str],
) -> Path:
    (tmp_path / "db_chapters.csv").write_text(
        "Id,BookId,Name,Order\n" + "\n".join(chapters_rows) + "\n", encoding="utf-8"
    )
    (tmp_path / "Gita_Slokas.csv").write_text(
        "Id;ChapterId;Name;Text;Transcription;Translation;Comment;Order;Audio;AudioSanskrit\n"
        + "\n".join(slokas_rows)
        + "\n",
        encoding="utf-8-sig",
    )
    return tmp_path


def test_load_book_one_on_a_clean_fixture_joins_and_sorts_correctly(tmp_path):
    dataset_dir = _write_fixture_dataset(
        tmp_path,
        chapters_rows=["1,1,Chapter One,1", "2,1,Chapter Two,2"],
        slokas_rows=[
            "2;1;1.2;Sanskrit B;Transcript B;Translation B;;2;;",
            "1;1;1.1;Sanskrit A;Transcript A;Translation A;;1;;",
            "3;2;2.1;Sanskrit C;Transcript C;Translation C;;1;;",
        ],
    )
    chapters = load_book_one(dataset_dir=dataset_dir, book_id=1)
    assert [c.order for c in chapters] == [1, 2]
    assert [s.order for s in chapters[0].slokas] == [1, 2]  # sorted despite file order 2,1
    assert chapters[0].slokas[0].translation_ru == "Translation A"
    assert chapters[0].title == "Chapter One"


def test_load_book_one_filters_to_the_requested_book_id(tmp_path):
    dataset_dir = _write_fixture_dataset(
        tmp_path,
        chapters_rows=["1,1,Book One Chapter,1", "2,2,Book Two Chapter,1"],
        slokas_rows=["1;1;1.1;S;T;Tr;;1;;", "2;2;S;T;Tr;;1;;"],
    )
    chapters = load_book_one(dataset_dir=dataset_dir, book_id=1)
    assert len(chapters) == 1
    assert chapters[0].title == "Book One Chapter"


def test_duplicate_chapter_order_is_rejected(tmp_path):
    dataset_dir = _write_fixture_dataset(
        tmp_path,
        chapters_rows=["1,1,Chapter A,1", "2,1,Chapter B,1"],  # both Order=1
        slokas_rows=["1;1;1.1;S;T;Tr;;1;;"],
    )
    with pytest.raises(DatasetIntegrityError, match="Duplicate chapter Order"):
        load_book_one(dataset_dir=dataset_dir, book_id=1)


def test_duplicate_sloka_order_within_chapter_is_rejected(tmp_path):
    dataset_dir = _write_fixture_dataset(
        tmp_path,
        chapters_rows=["1,1,Chapter A,1"],
        slokas_rows=["1;1;1.1;S;T;Tr;;1;;", "2;1;1.1;S2;T2;Tr2;;1;;"],  # both Order=1
    )
    with pytest.raises(DatasetIntegrityError, match="Duplicate sloka Order"):
        load_book_one(dataset_dir=dataset_dir, book_id=1)


def test_empty_required_field_is_rejected(tmp_path):
    dataset_dir = _write_fixture_dataset(
        tmp_path,
        chapters_rows=["1,1,Chapter A,1"],
        slokas_rows=["1;1;1.1;;T;Tr;;1;;"],  # empty Text (sanskrit)
    )
    with pytest.raises(DatasetIntegrityError, match="Empty required field"):
        load_book_one(dataset_dir=dataset_dir, book_id=1)


def test_non_integer_order_is_rejected(tmp_path):
    dataset_dir = _write_fixture_dataset(
        tmp_path,
        chapters_rows=["1,1,Chapter A,not-a-number"],
        slokas_rows=["1;1;1.1;S;T;Tr;;1;;"],
    )
    with pytest.raises(DatasetIntegrityError, match="not an integer"):
        load_book_one(dataset_dir=dataset_dir, book_id=1)


def test_verify_dataset_integrity_rejects_wrong_chapter_count():
    from models import CanonicalChapter

    chapters = tuple(
        CanonicalChapter(book_id=1, chapter_id=i, order=i, title="x", slokas=()) for i in range(1, 5)
    )
    with pytest.raises(DatasetIntegrityError, match="Expected chapter orders"):
        verify_dataset_integrity(chapters)


def test_verify_dataset_integrity_rejects_wrong_sloka_count():
    from models import CanonicalChapter, SlokaSource

    sloka = SlokaSource(
        id=1, chapter_id=1, order=1, name="1.1", sanskrit="s", transcription="t",
        translation_ru="tr", comment_ru="", audio_ref="", sanskrit_audio_ref="",
    )
    chapters = tuple(
        CanonicalChapter(book_id=1, chapter_id=i, order=i, title="x", slokas=(sloka,))
        for i in range(1, 19)
    )
    with pytest.raises(DatasetIntegrityError, match="Expected 663 total slokas"):
        verify_dataset_integrity(chapters)


def test_real_dataset_loads_exactly_18_chapters_and_663_slokas():
    chapters = load_book_one(dataset_dir=DATASET_DIR, book_id=1)
    verify_dataset_integrity(chapters)  # must not raise
    assert len(chapters) == 18
    assert sum(len(c.slokas) for c in chapters) == 663
    assert [c.order for c in chapters] == list(range(1, 19))
    assert chapters[0].title == "Осмотр Армий"
    assert chapters[17].title == "Йога освобождения"
