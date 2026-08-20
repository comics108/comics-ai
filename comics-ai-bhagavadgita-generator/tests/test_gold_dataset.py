import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_gold_dataset import GoldAnnotation, GoldDataset, validate_gold_dataset, verify_gold_artifacts
from automated_review import AutomatedMaskReview, validate_automated_mask_review


def _dataset(*, origin=None, reviewer="auto:automated-mask-review-v1") -> GoldDataset:
    annotations = []
    for index in range(120):
        composition = f"psd-{index % 2}" if index < 60 else f"panorama-{index % 2}"
        source_kind = "psd" if index < 60 else "panorama"
        split = "test" if index >= 90 else "train"
        # Held-out annotations use panorama-0/1 only; train panorama identities would leak, so
        # keep all panorama examples held out and fill training with the two PSD compositions.
        if source_kind == "panorama":
            split = "test"
        review = validate_automated_mask_review(AutomatedMaskReview(
            source_kind=source_kind,
            method_families=("native_alpha",) if source_kind == "psd" else ("instance_model", "edge_matting"),
            agreement_iou=.9, boundary_f1=.82, foreground_coverage=.35,
            rectangularity=.7, source_sha256="c" * 64, mask_sha256="a" * 64,
        ))
        annotations.append(GoldAnnotation(
            id=f"gold-{index:03}", asset_version_id=f"asset-{index}:v1",
            source_composition_id=composition, source_kind=source_kind, split=split,
            semantic_kind="character", canonical_entity_id="entity-krishna",
            principal_character=True, bitmap_mask_file=f"masks/{index}.png",
            mask_sha256="a" * 64, source_region=(0, 0, 20, 30),
            review_resolution=(20, 30), source_to_review_scale=(1.0, 1.0),
            label_origin=origin or ("psd_alpha_reviewed" if source_kind == "psd" else "automated_consensus"),
            reviewer=reviewer,
            accepted_at="2026-08-11T00:00:00Z", accepted=True,
            review_mode="automated", review_evidence=review.evidence,
        ))
    return GoldDataset(version="gold-v1", annotations=tuple(annotations))


def test_gold_v1_contract_accepts_reviewed_source_disjoint_true_masks():
    result = validate_gold_dataset(_dataset())
    assert result.accepted_count == 120
    assert result.held_out_count == 60
    assert result.source_composition_count == 4


def test_bbox_bootstrap_or_missing_reviewer_cannot_enter_gold():
    with pytest.raises(ValueError, match="bbox bootstrap"):
        validate_gold_dataset(_dataset(origin="bbox_bootstrap"))
    with pytest.raises(ValueError, match="reviewer"):
        validate_gold_dataset(_dataset(reviewer=""))


def test_source_composition_cannot_cross_train_and_held_out_splits():
    dataset = _dataset()
    first = dataset.annotations[0]
    leaked = GoldAnnotation(**{**first.__dict__, "id": "leak", "split": "test"})
    with pytest.raises(ValueError, match="split leakage"):
        validate_gold_dataset(GoldDataset(dataset.version, (*dataset.annotations, leaked)))


def test_native_psd_geometry_may_have_signed_origin():
    dataset = _dataset()
    first = dataset.annotations[0]
    signed = GoldAnnotation(**{**first.__dict__, "source_region": (-12, -4, 20, 30)})
    validate_gold_dataset(GoldDataset(dataset.version, (signed, *dataset.annotations[1:])))


def test_gold_artifact_verification_fails_on_tampering(tmp_path):
    dataset = _dataset()
    mask = tmp_path / "mask.png"
    mask.write_bytes(b"not-the-expected-mask")
    annotations = tuple(
        GoldAnnotation(**{**annotation.__dict__, "bitmap_mask_file": "mask.png"})
        for annotation in dataset.annotations
    )
    with pytest.raises(ValueError, match="checksum mismatch"):
        verify_gold_artifacts(GoldDataset(dataset.version, annotations), tmp_path)
