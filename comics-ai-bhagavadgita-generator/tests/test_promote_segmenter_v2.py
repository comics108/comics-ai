import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from promote_segmenter_v2 import decide


def test_real_candidate_passes_crop_masks_but_fails_tiled_and_semantic_gates():
    root = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    report = decide(
        root / "segmenter-competition-v2/border-matting-gold-v2.1.json",
        root / "segmenter-competition-v2/tiled-merge-gold-v2.1.json",
        root / "gold-v2.1/readiness-v2.json",
    )
    assert report["gates"]["gold_readiness"]
    assert report["gates"]["mask_iou"] and report["gates"]["boundary_f1"]
    assert not report["gates"]["tiled_duplicate_rate"]
    assert not report["gates"]["tiled_instance_recall"]
    assert not report["gates"]["tiled_instance_precision"]
    assert not report["gates"]["tiled_no_collapsed_instances"]
    assert not report["gates"]["semantic_macro_f1"]
    assert report["promotion"] == "rejected"


def test_mixed_gold_hashes_are_rejected(tmp_path):
    root = Path(__file__).resolve().parents[4] / "work/bhagavadgita/production"
    tiled = json.loads((root / "segmenter-competition-v2/tiled-merge-gold-v2.1.json").read_text())
    tiled["dataset_manifest_sha256"] = "f" * 64
    changed = tmp_path / "tiled.json"
    changed.write_text(json.dumps(tiled))
    with pytest.raises(ValueError, match="one Gold"):
        decide(root / "segmenter-competition-v2/border-matting-gold-v2.1.json", changed, root / "gold-v2.1/readiness-v2.json")
