import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from automated_review import AutomatedMaskReview, validate_automated_mask_review


def _review(source_kind="panorama", families=("instance_model", "edge_matting"), **changes):
    values = dict(
        source_kind=source_kind, method_families=families, agreement_iou=.9,
        boundary_f1=.82, foreground_coverage=.35, rectangularity=.7,
        source_sha256="a" * 64, mask_sha256="b" * 64,
    )
    values.update(changes)
    return AutomatedMaskReview(**values)


def test_panorama_consensus_produces_versioned_machine_reviewer_evidence():
    result = validate_automated_mask_review(_review())
    assert result.reviewer_id == "auto:automated-mask-review-v1"
    assert "automated-mask-review-v1" in result.evidence


def test_box_supervised_single_family_or_rectangular_mask_fails_closed():
    with pytest.raises(ValueError, match="two independent"):
        validate_automated_mask_review(_review(families=("box_supervised_models",)))
    with pytest.raises(ValueError, match="rectangular"):
        validate_automated_mask_review(_review(rectangularity=.99))


def test_native_psd_alpha_needs_no_human_or_second_model_family():
    result = validate_automated_mask_review(_review(
        source_kind="psd", families=("native_alpha",), agreement_iou=1,
    ))
    assert result.reviewer_id.startswith("auto:")
