import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_gold_v2 import audit


def test_real_gold_v1_cannot_non_circularly_promote_production_segmenter():
    root = Path(__file__).resolve().parents[4]
    result = audit(root / "work/bhagavadgita/production/gold-v1/manifest.json")
    assert result["accepted_count"] == 130
    assert result["held_out_count"] == 40
    assert result["independent_held_out_count"] == 0
    assert result["principal_identity_count"] == 0
    assert result["gold_v2_readiness"] == "blocked"
    assert "independent_held_out_instances_at_least_30" in result["missing"]
    assert "principal_identity_labels_present" in result["missing"]
    assert "tiled_duplicate_metric_fixture_present" in result["missing"]


def test_tiled_report_must_match_exact_gold_manifest(tmp_path):
    root = Path(__file__).resolve().parents[4]
    manifest = root / "work/bhagavadgita/production/gold-v1/manifest.json"
    report = tmp_path / "tiled.json"
    report.write_text('{"dataset_manifest_sha256":"' + "b" * 64 + '"}')
    import pytest
    with pytest.raises(ValueError, match="different Gold"):
        audit(manifest, report)
