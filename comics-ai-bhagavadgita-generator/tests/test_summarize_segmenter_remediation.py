import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from summarize_segmenter_remediation import summarize


def test_real_summary_blocks_circular_shortcut_and_needs_no_human():
    root = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    report = summarize(
        root / "gold-v2.2/manifest.json",
        root / "segmenter-competition-v2/promotion-border-matting-v2.json",
        root / "segmenter-competition-v2/promotion-maskrcnn-v2.json",
    )
    assert report["state"] == "production_cutting_blocked"
    assert all(item["promotion"] == "rejected" for item in report["candidates"])
    assert report["forbidden_shortcut"]["reason"] == "evaluation_family_participated_in_label_consensus"
    assert report["human_participation_required"] is False
