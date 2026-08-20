import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from summarize_independent_reviewer_remediation import build


def test_summary_forbids_fragment_background_and_threshold_shortcuts(tmp_path):
    artifact = tmp_path / "report.json"
    artifact.write_text(json.dumps({"state": "blocked", "accepted_pair_count": 0,
                                    "rejected_fragment_count": 9}))
    summary = build([artifact])
    assert summary["state"] == "production_cutting_supervision_blocked"
    assert summary["next_autonomous_input_contract"]["complete_non_border_foreground_instances"] == 30
    assert "lower_iou_or_completeness_gates_to_reach_quota" in summary["forbidden_shortcuts"]
    assert "count_high_iou_fragments_as_complete_instances" in summary["forbidden_shortcuts"]
