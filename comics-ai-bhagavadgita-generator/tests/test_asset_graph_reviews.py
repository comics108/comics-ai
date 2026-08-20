import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from asset_graph import AssetEntityLink, AssetGraph, EntityRevision
from production_models import Asset, AssetVersion, Lineage, ReviewDecision
from reviews import ReviewLedger


def _version(*, mask: str | None = "mask.png") -> AssetVersion:
    return AssetVersion(
        version=1,
        source_id="source-1",
        source_region=(0, 0, 20, 30),
        rgba_file="rgba.png",
        bitmap_mask_file=mask,
        contour_file=None,
        width=20,
        height=30,
        art_stage="ink",
        style_tags=(),
        palette=(),
        pose=None,
        expression=None,
        costume=None,
        view=None,
        allowed_transformations=("translate",),
        lineage=Lineage(
            input_checksums=("a" * 64,),
            action_id="extract-1",
            code_revision="rev",
            model_checkpoint=None,
            configuration_hash="b" * 64,
            environment={},
            timestamp="2026-08-11T00:00:00Z",
            cost_usage={},
            reviewer_decision_ids=(),
        ),
        metrics={},
        review_state="proposed",
    )


def test_uncertain_identity_link_is_a_proposal_and_does_not_mutate_asset(tmp_path):
    asset = Asset("asset-1", [], "character", [_version()])
    graph = AssetGraph()
    graph.add_asset(asset)
    graph.add_identity_proposal(AssetEntityLink(
        id="link-1", asset_id="asset-1", asset_version=1,
        entity_id="entity-krishna", role="depicted_character", confidence=.62,
        method="clip:model-v1", review_state="proposed", supersedes_id=None,
    ))

    assert asset.canonical_entity_ids == []
    assert graph.identity_links[-1].review_state == "proposed"
    snapshot = tmp_path / "graph.json"
    graph.write_snapshot(snapshot)
    restored = AssetGraph.read_snapshot(snapshot)
    assert restored.identity_links == graph.identity_links


def test_merge_and_split_revisions_remain_in_append_only_history():
    graph = AssetGraph()
    graph.record_entity_revision(EntityRevision(
        id="rev-merge", entity_id="entity-krishna", revision=2,
        operation="merge", source_entity_ids=("candidate-a", "candidate-b"),
        reviewer="anton", timestamp="2026-08-11T01:00:00Z", rationale="same iconography",
    ))
    graph.record_entity_revision(EntityRevision(
        id="rev-split", entity_id="entity-krishna", revision=3,
        operation="split", source_entity_ids=("entity-krishna",),
        reviewer="anton", timestamp="2026-08-11T02:00:00Z", rationale="distinct costume",
    ))
    assert [item.operation for item in graph.entity_revisions] == ["merge", "split"]


def test_upstream_change_invalidates_all_dependent_approvals_without_deleting_history(tmp_path):
    graph = AssetGraph()
    graph.add_dependency("source-1", "asset-1:v1")
    graph.add_dependency("asset-1:v1", "composition-1")
    ledger = ReviewLedger()
    for subject in ("asset-1:v1", "composition-1"):
        ledger.append(ReviewDecision(
            id=f"approve-{subject}", subject_id=subject, subject_version="1",
            dimension="art_direction", state="approved", reviewer="anton",
            timestamp="2026-08-11T00:00:00Z", rationale="approved",
        ))

    invalidated = ledger.invalidate_from(
        "source-1", graph, invalidated_by="source-1:v2",
        timestamp="2026-08-11T03:00:00Z",
    )
    assert {item.subject_id for item in invalidated} == {"asset-1:v1", "composition-1"}
    assert len(ledger.decisions) == 4
    assert all(item.invalidated_by == "source-1:v2" for item in invalidated)
    snapshot = tmp_path / "reviews.json"
    ledger.write_snapshot(snapshot)
    assert ReviewLedger.read_snapshot(snapshot).decisions == ledger.decisions


def test_bbox_only_foreground_cannot_receive_an_approval():
    asset = Asset("asset-1", [], "character", [_version(mask=None)])
    ledger = ReviewLedger()
    with pytest.raises(ValueError, match="bitmap mask"):
        ledger.approve_asset_version(
            asset, 1, decision_id="approve-1", dimension="technical",
            reviewer="anton", timestamp="2026-08-11T00:00:00Z", rationale="looks fine",
        )
