import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

import pytest

from models import CanonicalChapter, SlokaSource


def _real_sloka(order: int = 1) -> SlokaSource:
    return SlokaSource(
        id=1,
        chapter_id=1,
        order=order,
        name="1.1",
        sanskrit="धृतराष्ट्र उवाच",
        transcription="дхр̣тара̄ш̣т̣рах̣ ува̄ча",
        translation_ru="Дхритараштра сказал",
        comment_ru="",
        audio_ref="",
        sanskrit_audio_ref="",
    )


def test_sloka_source_holds_real_field_values():
    s = _real_sloka()
    assert s.chapter_id == 1
    assert s.order == 1
    assert "Дхритараштра" in s.translation_ru


def test_sloka_source_is_frozen_immutable():
    s = _real_sloka()
    with pytest.raises(dataclasses.FrozenInstanceError):
        s.order = 2


def test_canonical_chapter_holds_a_tuple_of_slokas_in_order():
    slokas = (_real_sloka(order=1), _real_sloka(order=2))
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="Осмотр Армий", slokas=slokas)
    assert chapter.title == "Осмотр Армий"
    assert len(chapter.slokas) == 2
    assert [s.order for s in chapter.slokas] == [1, 2]


def test_canonical_chapter_is_frozen_immutable():
    chapter = CanonicalChapter(book_id=1, chapter_id=1, order=1, title="x", slokas=())
    with pytest.raises(dataclasses.FrozenInstanceError):
        chapter.title = "y"
