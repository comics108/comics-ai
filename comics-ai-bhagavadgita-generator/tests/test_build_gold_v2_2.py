import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_gold_v2_2 import derive


def test_semantic_split_is_source_disjoint_and_test_is_native_alpha():
    root = Path(__file__).resolve().parents[4]
    dataset = derive(root / "work/bhagavadgita/production/gold-v2.1/manifest.json")
    assert {split: sum(item.split == split for item in dataset.annotations)
            for split in ("train", "validation", "test")} == {
                "train": 81, "validation": 20, "test": 30,
            }
    test = [item for item in dataset.annotations if item.split == "test"]
    assert all(item.label_origin == "psd_alpha_reviewed" for item in test)
    assert {item.semantic_kind for item in test} == {"art", "animal"}
    sources = {split: {item.source_composition_id for item in dataset.annotations if item.split == split}
               for split in ("train", "validation", "test")}
    assert not (sources["train"] & sources["test"] or sources["validation"] & sources["test"])
