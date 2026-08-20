import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from summarize_segmenter_competition import summarize


def _report(path: Path, name: str, iou: float, promotion: str = "rejected") -> Path:
    path.write_text(json.dumps({
        "dataset_version": "gold-v1", "dataset_manifest_sha256": "a" * 64,
        "test_instances": 40, "predictor": name, "family": "test",
        "promotion": promotion, "promotion_failures": [] if promotion == "accepted" else ["gate"],
        "metrics": {"mean_mask_iou": iou, "mean_boundary_f1": .7,
                    "instance_recall_at_iou_0_5": .9, "automated_artifact_failure_count": 0,
                    "duplicate_instance_rate": 0, "semantic_kind_macro_f1": .8},
    }), encoding="utf-8")
    return path


def test_summary_ranks_evidence_but_does_not_promote_rejected_candidate(tmp_path):
    summary = summarize([
        _report(tmp_path / "a.json", "weaker", .2),
        _report(tmp_path / "b.json", "stronger", .8),
    ])
    assert [item["predictor"] for item in summary["ranking"]] == ["stronger", "weaker"]
    assert summary["decision"] == "no_candidate_promoted"
    assert summary["selected_predictor"] is None


def test_summary_rejects_mixed_gold_sets(tmp_path):
    left = _report(tmp_path / "a.json", "left", .2)
    right = _report(tmp_path / "b.json", "right", .3)
    payload = json.loads(right.read_text())
    payload["dataset_manifest_sha256"] = "b" * 64
    right.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="one identical Gold"):
        summarize([left, right])
