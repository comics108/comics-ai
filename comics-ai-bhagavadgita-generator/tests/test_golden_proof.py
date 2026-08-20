import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from golden_proof import build_proof, file_sha256, write_immutable


def test_real_proof_is_complete_reproducible_and_blocks_scale_out(tmp_path):
    production = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    proof = build_proof(production)
    assert proof["proof_scope"]["golden_chapters"] == [1, 11]
    assert proof["proof_scope"]["target_scroll_type"] == "vertical"
    assert len(proof["artifacts"]) == 13
    for record in proof["artifacts"]:
        assert file_sha256(production / record["path"]) == record["sha256"]
    assert [item["generation_required_count"] for item in proof["chapters"]] == [6, 6]
    assert proof["lettering"] == {"accepted": 3, "total": 6, "state": "blocked"}
    assert proof["golden_release_state"] == "blocked"
    assert proof["scale_out_to_all_18"] == "blocked"
    assert proof["human_participation_required"] is False
    assert [item["order"] for item in proof["next_actions"]] == [1, 2, 3, 4, 5]

    target = tmp_path / "proof.json"
    write_immutable(proof, target)
    assert json.loads(target.read_text()) == proof
    with pytest.raises(FileExistsError):
        write_immutable(proof, target)


def test_v2_proof_tracks_promoted_lettering_without_rewriting_v1():
    production = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    proof = build_proof(
        production, lettering_relative="lettering/fixtures-v2.json",
        validation_relative="releases/golden-validation-v2.json",
    )
    assert proof["lettering"] == {"accepted": 4, "total": 6, "state": "blocked"}
    assert proof["golden_release_state"] == "blocked"


def test_v3_proof_accepts_new_identity_lineage_as_evidence_but_not_release():
    production = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    proof = build_proof(
        production, lettering_relative="lettering/fixtures-v2.json",
        identity_relative="identity-style/retrieval-v3-explicit.json",
        validation_relative="releases/golden-validation-v3.json",
    )
    assert any(item["path"] == "identity-style/retrieval-v3-explicit.json"
               for item in proof["artifacts"])
    assert proof["scale_out_to_all_18"] == "blocked"
