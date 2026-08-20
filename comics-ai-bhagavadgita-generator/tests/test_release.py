import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from release import (DIMENSIONS, GateDecision, evaluate_release,
                     invalidate_changed_dependencies, publish_immutable, real_golden_gates)


SHA = "a" * 64


def approved_gates():
    return [GateDecision(name, "approved", "auto:test", "passed", ("proof",), (SHA,)) for name in DIMENSIONS]


def test_all_six_dimensions_and_artifact_are_required(tmp_path):
    artifact = tmp_path / "chapter.comics"
    artifact.write_bytes(b"archive")
    assert evaluate_release(approved_gates(), [artifact])["release_state"] == "accepted"
    assert evaluate_release(approved_gates()[:-1], [artifact])["release_state"] == "blocked"
    rejected = approved_gates()
    rejected[2] = GateDecision("art_direction", "abstained", "auto:test", "unknown", ("proof",), (SHA,))
    report = evaluate_release(rejected, [artifact])
    assert report["release_state"] == "blocked"
    assert "art_direction:abstained" in report["blockers"]


def test_dependency_change_marks_approval_stale_and_blocks(tmp_path):
    stale = invalidate_changed_dependencies(approved_gates(), {"b" * 64})
    assert all(item.state == "stale" for item in stale)
    artifact = tmp_path / "chapter.comics"
    artifact.write_bytes(b"archive")
    assert evaluate_release(stale, [artifact])["release_state"] == "blocked"


def test_publish_is_immutable_and_blocked_report_never_creates_release(tmp_path):
    blocked = evaluate_release(approved_gates(), [])
    report_path, release_root = tmp_path / "report.json", tmp_path / "release-v1"
    publish_immutable(blocked, report_path, release_root)
    assert report_path.is_file() and not release_root.exists()
    with pytest.raises(FileExistsError):
        publish_immutable(blocked, report_path, release_root)


def test_real_golden_state_has_six_independent_fail_closed_dimensions():
    root = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    gates = real_golden_gates(root)
    assert [gate.dimension for gate in gates] == list(DIMENSIONS)
    assert {gate.state for gate in gates} <= {"rejected", "abstained"}
    report = evaluate_release(gates, [])
    assert report["release_state"] == "blocked"
    assert len(report["blockers"]) == 7  # six dimensions plus the absent release archive


def test_real_v2_lettering_gate_uses_promoted_aggregate():
    root = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    gates = real_golden_gates(root, lettering_relative="lettering/fixtures-v2.json")
    lettering = next(gate for gate in gates if gate.dimension == "lettering")
    assert lettering.state == "rejected"
    assert lettering.evidence == ("4/6",)


def test_real_v3_identity_is_partial_and_still_fails_closed():
    root = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    gates = real_golden_gates(root, identity_relative="identity-style/retrieval-v3-explicit.json")
    identity = next(gate for gate in gates if gate.dimension == "identity_style")
    assert identity.state == "abstained"
    assert "identity_abstained" in identity.evidence
