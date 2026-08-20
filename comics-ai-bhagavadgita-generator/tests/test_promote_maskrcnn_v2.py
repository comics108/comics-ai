import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from promote_maskrcnn_v2 import decide


def test_real_maskrcnn_passes_crop_iou_recall_but_not_release_gates():
    root=Path(__file__).resolve().parents[4]/"work/bhagavadgita/production/segmenter-competition-v2"
    report=decide(root/"gold-v2.2-maskrcnn-balanced-epoch6.json",root/"gold-v2.2-maskrcnn-balanced-epoch6-tiled.json")
    assert report["gates"]["mask_iou"] and report["gates"]["crop_instance_recall"]
    assert not report["gates"]["boundary_f1"]
    assert not report["gates"]["semantic_above_majority"]
    assert not report["gates"]["tiled_recall"]
    assert report["promotion"] == "rejected"
