import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import positioning_bridge as pb
from data_checkpoint import summarize


def test_summarize_on_fixture(tmp_path, monkeypatch):
    fixture = tmp_path / "alignment.jsonl"
    rows = [
        {"episode_file": "a.comics", "status": "matched", "ground_truth_cluster": [1, 2, 3]},
        {"episode_file": "a.comics", "status": "matched", "ground_truth_cluster": [4, 5]},
        {"episode_file": "b.comics", "status": "matched", "ground_truth_cluster": [6]},
        {"episode_file": None, "status": "skipped_no_match", "ground_truth_cluster": []},
    ]
    fixture.write_text("\n".join(json.dumps(r) for r in rows))
    monkeypatch.setattr(pb, "ALIGNMENT_JSONL", fixture)

    summary = summarize()
    assert summary["matched_pairs"] == 3
    assert summary["distinct_episodes"] == 2
    assert summary["episodes"] == {"a.comics": 2, "b.comics": 1}
    assert summary["min_cluster_size"] == 1
    assert summary["max_cluster_size"] == 3


def test_summarize_on_real_data_matches_manual_count():
    # Real counts as of 2026-08-02: `sdd-comics-ai-transformations`'s re-matching refinement
    # (align_photo.py's MARGIN_FOR_SINGLE_HIT rule, flows/sdd-comics-ai-transformations/
    # 02-specifications.md) recovered 22 of 99 previously-unmatched page-rows, raising this from
    # 37/16 to 59 matched rows across 19 distinct episodes.
    summary = summarize()
    assert summary["matched_pairs"] == 59
    assert summary["distinct_episodes"] == 19
