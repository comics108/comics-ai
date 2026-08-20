import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from resolve_coverage import CoverageCandidate, resolve_beat


BEAT = {"id": "ch01-beat01", "chapter_order": 1}


def test_confirmed_accepted_source_suppresses_generation_and_paid_action():
    candidate = CoverageCandidate(
        "asset:v1", "bhagavad_gita", (1,), "confirmed", "accepted", "direct", ("scope",)
    )
    result = resolve_beat(BEAT, (candidate,))
    assert result["state"] == "accepted_source"
    assert result["proposed_action_ids"] == []
    assert result["paid_generation_suppressed"] is True


def test_inferred_or_gita_dhyanam_asset_cannot_satisfy_canonical_coverage():
    candidates = (
        CoverageCandidate("inferred:v1", "bhagavad_gita", (1,), "inferred", "proposed", "direct", ()),
        CoverageCandidate("dhyanam:v1", "gita_dhyanam", (), "not_applicable", "accepted", "reuse", ()),
    )
    result = resolve_beat(BEAT, candidates)
    assert result["state"] == "generation_required"
    assert result["asset_version_ids"] == []
    assert result["proposed_action_ids"] == ["local:grounded-visual:ch01-beat01:v1"]
    assert {item["excluded"] for item in result["considered_candidates"]} == {
        "mapping_or_review_not_accepted", "noncanonical_gita_dhyanam",
    }
