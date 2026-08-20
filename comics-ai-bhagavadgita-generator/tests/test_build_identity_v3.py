import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_identity_v3 import build


def _write(path, payload):
    path.write_text(json.dumps(payload))
    return path


def test_only_explicit_source_identity_resolves(tmp_path):
    retrieval = _write(tmp_path / "retrieval.json", {"results": [
        {"query_asset_version_id": "a:v1", "identity_action": "abstained"},
        {"query_asset_version_id": "b:v1", "identity_action": "abstained"},
    ]})
    gold = _write(tmp_path / "gold.json", {"annotations": [{
        "id": "a", "asset_version_id": "a:v2", "canonical_entity_id": "krishna",
        "mask_sha256": "a" * 64,
        "review_evidence": ["canonical_identity:explicit_psd_parent_group:1:krishna"],
    }]})
    result = build(retrieval, gold)
    assert result["source_explicit_count"] == 1
    assert result["abstained_count"] == 1
    assert result["identity_merges_from_similarity"] == 0
    assert result["results"][0]["canonical_entity_id"] == "krishna"
    assert result["results"][0]["resolved_asset_version_id"] == "a:v2"


def test_identity_without_explicit_hierarchy_fails_closed(tmp_path):
    retrieval = _write(tmp_path / "retrieval.json", {"results": []})
    gold = _write(tmp_path / "gold.json", {"annotations": [{
        "id": "a", "asset_version_id": "a:v2", "canonical_entity_id": "krishna",
        "mask_sha256": "a" * 64, "review_evidence": ["similarity:0.999"],
    }]})
    with pytest.raises(ValueError, match="explicit source-hierarchy"):
        build(retrieval, gold)


def test_newer_explicit_asset_missing_from_old_index_is_appended(tmp_path):
    retrieval = _write(tmp_path / "retrieval.json", {"results": []})
    gold = _write(tmp_path / "gold.json", {"annotations": [{
        "id": "a", "asset_version_id": "a:v2", "canonical_entity_id": "krishna",
        "mask_sha256": "a" * 64,
        "review_evidence": ["canonical_identity:explicit_psd_parent_group:1:krishna"],
    }]})
    result = build(retrieval, gold)
    assert result["query_count"] == result["source_explicit_count"] == 1
    assert result["results"][0]["query_asset_version_id"] == "a:v2"
