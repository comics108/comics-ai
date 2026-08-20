import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_true_mask_target_uses_bitmap_not_bbox_fill():
    torch = pytest.importorskip("torch")
    from build_gold_dataset import GoldAnnotation
    from train_gold_maskrcnn import GoldInstanceDataset
    from PIL import Image

    class Fake(GoldInstanceDataset):
        pass

    root = Path(__file__).resolve().parents[4]
    import json
    from gold_segmenter_data import load_gold_dataset
    gold = load_gold_dataset(root / "work/bhagavadgita/production/gold-v2.2/manifest.json", root)
    item = GoldInstanceDataset([next(x for x in gold.annotations if x.split == "test")], root)[0]
    mask = item[1]["masks"][0].numpy().astype(bool)
    x0, y0, x1, y1 = item[1]["boxes"][0].int().tolist()
    assert mask.sum() < (x1 - x0) * (y1 - y0)
    assert item[1]["labels"].item() in {1, 2, 3, 4}
