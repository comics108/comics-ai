import dataclasses
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from production_models import Asset, AssetVersion, Lineage, SourceRecord, SourceSemanticScope


def test_source_scope_and_asset_versions_are_immutable_provenance_records():
    scope = SourceSemanticScope(
        id="scope-ch05-14-29",
        work="bhagavad_gita",
        scope="canonical_verse_range",
        chapter_orders=(5,),
        verse_ranges=((5, 14, 29),),
        mapping_state="confirmed",
        evidence=("reviewed PSD balloon sequence",),
        reviewer="anton",
    )
    source = SourceRecord(
        id="source-ch05-psd",
        kind="psd",
        relative_path="vaishnav/drawing/app_BG._chiba5.psd",
        sha256="a" * 64,
        byte_size=123,
        media_type="image/vnd.adobe.photoshop",
        metadata={"layer_count": 15},
        semantic_scope_id=scope.id,
    )
    lineage = Lineage(
        input_checksums=(source.sha256,),
        action_id="recover-psd-ch05",
        code_revision="71d3b30",
        model_checkpoint=None,
        configuration_hash="b" * 64,
        environment={"adapter": "psd-tools"},
        timestamp="2026-08-10T00:00:00Z",
        cost_usage={},
        reviewer_decision_ids=(),
    )
    version = AssetVersion(
        version=1,
        source_id=source.id,
        source_region=None,
        rgba_file="rgba.png",
        bitmap_mask_file="mask.png",
        contour_file=None,
        width=640,
        height=480,
        art_stage="final",
        style_tags=("vaishnav",),
        palette=("#112233",),
        pose=None,
        expression=None,
        costume=None,
        view=None,
        allowed_transformations=("translate", "scale"),
        lineage=lineage,
        metrics={"alpha_coverage": 0.52},
        review_state="proposed",
    )
    asset = Asset(
        id="asset-ch05-panel",
        canonical_entity_ids=[],
        semantic_kind="art",
        versions=[version],
    )

    assert asset.versions == [version]
    with pytest.raises(dataclasses.FrozenInstanceError):
        scope.work = "gita_dhyanam"
    with pytest.raises(dataclasses.FrozenInstanceError):
        version.review_state = "accepted"
