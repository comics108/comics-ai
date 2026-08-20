import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from audit_gold_v2 import audit
from build_gold_v2 import derive


def test_real_gold_v2_split_has_independent_native_alpha_heldout():
    root = Path(__file__).resolve().parents[4]
    dataset = derive(root / "work/bhagavadgita/production/gold-v1/manifest.json")
    counts = {split: sum(item.split == split for item in dataset.annotations) for split in ("train", "validation", "test")}
    assert counts == {"train": 66, "validation": 4, "test": 60}
    assert {item.source_composition_id for item in dataset.annotations if item.split == "test"} == {"psd-app-bg-chiba5"}
    assert all(item.label_origin == "psd_alpha_reviewed" for item in dataset.annotations if item.split == "test")
    kinds = {item.semantic_kind for item in dataset.annotations if item.split == "test"}
    assert {"art", "animal", "character", "fx"} <= kinds
    relabelled = [item for item in dataset.annotations if item.semantic_kind != "art" and item.split == "test"]
    assert all(any(value.startswith("semantic_label:explicit_psd_group:") for value in item.review_evidence) for item in relabelled)
