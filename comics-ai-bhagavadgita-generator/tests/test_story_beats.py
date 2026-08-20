import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_story_beats import apply_independent_review, validate_candidate
from models import CanonicalChapter, SlokaSource


def _chapter():
    slokas = tuple(SlokaSource(i, 1, i, "", "s", "t", f"translation {i}", "", "", "")
                   for i in range(1, 13))
    return CanonicalChapter(1, 1, 1, "Chapter", slokas)


def _candidate():
    return {"beats": [{
        "order": index, "title": f"Beat {index}", "first_sloka_order": index * 2 - 1,
        "last_sloka_order": index * 2, "source_quote_orders": [index * 2 - 1],
        "synopsis": f"translation {index * 2 - 1}", "required_entities": [],
        "required_actions": [], "required_location": None, "required_shots": [],
    } for index in range(1, 7)]}


def test_story_beats_cover_every_source_once_and_map_real_ids():
    beats = validate_candidate(_chapter(), _candidate())
    assert len(beats) == 6
    assert tuple(value for beat in beats for value in beat.source_sloka_ids) == tuple(range(1, 13))


def test_story_beat_validator_rejects_gap_or_out_of_range_quote():
    candidate = _candidate()
    candidate["beats"][1]["first_sloka_order"] = 4
    with pytest.raises(ValueError, match="contiguous"):
        validate_candidate(_chapter(), candidate)
    candidate = _candidate()
    candidate["beats"][0]["source_quote_orders"] = [3]
    with pytest.raises(ValueError, match="inside"):
        validate_candidate(_chapter(), candidate)


def test_independent_reviewer_must_cover_every_beat_but_is_advisory_on_proven_fields():
    beats = validate_candidate(_chapter(), _candidate())
    reviews = {"reviews": [{"order": index, "grounded": index != 3,
                            "citations_support_synopsis": True,
                            "requirements_not_invented": True, "reason": "checked"}
                           for index in range(1, 7)]}
    accepted = apply_independent_review(beats, reviews, "reviewer")
    assert all(beat.review_state == "machine_verified_source_grounding" for beat in accepted)
    reviews["reviews"].pop()
    with pytest.raises(ValueError, match="cover every beat"):
        apply_independent_review(beats, reviews, "reviewer")
