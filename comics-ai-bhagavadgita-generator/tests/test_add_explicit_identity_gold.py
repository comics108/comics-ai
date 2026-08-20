import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from add_explicit_identity_gold import augment


def test_identity_is_scoped_to_explicit_krishna_native_layer(tmp_path):
    root = Path(__file__).resolve().parents[4]
    dataset = augment(
        root / "work/bhagavadgita/production/gold-v2/manifest.json",
        root / "dataset/bhagavadgita/vaishnav/drawing/app_BG._chiba5.psd", tmp_path,
    )
    principals = [item for item in dataset.annotations if item.principal_character]
    assert len(principals) == 1
    assert principals[0].canonical_entity_id == "krishna"
    assert principals[0].semantic_kind == "character"
    assert "identity_scope:this_native_layer_only" in principals[0].review_evidence
