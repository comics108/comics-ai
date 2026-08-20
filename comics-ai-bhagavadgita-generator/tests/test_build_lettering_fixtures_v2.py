import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_lettering_fixtures_v2 import build


def test_verified_promotion_replaces_one_fixture_and_preserves_order(tmp_path):
    artifact = tmp_path / "candidate.png"
    artifact.write_bytes(b"png")
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"manifest_sha256": "a" * 64, "results": [
        {"id": "one", "decision": "rejected", "exact_readback": False},
        {"id": "two", "decision": "accepted", "exact_readback": True},
    ]}))
    promotion = tmp_path / "promotion.json"
    promotion.write_text(json.dumps({"candidate": {
        "id": "one", "decision": "accepted", "exact_readback": True,
        "files": {"rgba": str(artifact)},
    }}))
    report = build(base, [promotion])
    assert [item["id"] for item in report["results"]] == ["one", "two"]
    assert report["accepted_count"] == 2
    assert report["release_state"] == "accepted"


def test_rejected_promotion_fails_closed(tmp_path):
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"manifest_sha256": "a" * 64, "results": [
        {"id": "one", "decision": "rejected"},
    ]}))
    promotion = tmp_path / "promotion.json"
    promotion.write_text(json.dumps({"candidate": {
        "id": "one", "decision": "rejected", "exact_readback": False, "files": {},
    }}))
    with pytest.raises(ValueError, match="accepted exact-readback"):
        build(base, [promotion])
