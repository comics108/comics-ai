import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_gold_dataset import GoldAnnotation
from classify_assets import classify_annotation, describe_visual, evaluation_coverage
from retrieve_assets import build_retrieval_index, cosine_similarity


def test_visual_descriptor_is_deterministic_and_mask_bounded():
    image = Image.new("RGB", (20, 20), "white")
    ImageDraw.Draw(image).ellipse((3, 2, 16, 18), fill="black")
    mask = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(mask).ellipse((3, 2, 16, 18), fill=255)
    first = describe_visual(image, mask)
    second = describe_visual(image, mask)
    assert first == second
    assert len(first.vector) == 75
    assert "monochrome" in first.style_tags
    assert 0 < first.foreground_coverage < 1


def test_retrieval_similarity_never_becomes_identity_merge():
    catalog = {"dataset_version": "gold-v1", "proposals": [
        {"asset_version_id": "a:v1", "semantic_kind_proposal": "art", "descriptor": [1., 0.]},
        {"asset_version_id": "b:v1", "semantic_kind_proposal": "art", "descriptor": [1., 0.]},
    ]}
    index = build_retrieval_index(catalog, limit=1)
    assert cosine_similarity([1., 0.], [1., 0.]) == 1.
    assert index["identity_merges"] == 0
    assert index["results"][0]["neighbors"][0]["canonical_identity_match"] is None
    assert index["results"][0]["identity_action"] == "abstained"


def test_identity_evaluation_abstains_without_canonical_principal_coverage():
    annotation = GoldAnnotation(
        id="gold", asset_version_id="asset:v1", source_composition_id="psd-a",
        source_kind="psd", split="train", semantic_kind="art", canonical_entity_id=None,
        principal_character=False, bitmap_mask_file="mask.png", mask_sha256="a" * 64,
        source_region=(0, 0, 10, 10), review_resolution=(10, 10),
        source_to_review_scale=(1., 1.), label_origin="psd_alpha_reviewed",
        reviewer="auto:test", accepted_at="2026-08-11T00:00:00Z", accepted=True,
        review_mode="automated", review_evidence=("automated-mask-review-v1",),
    )
    evaluation = evaluation_coverage((annotation,))
    assert evaluation["decision"] == "abstained"
    assert evaluation["identity_top1"] is None
    assert "principal_canonical_identity_coverage_is_insufficient" in evaluation["reasons"]

    image = Image.new("RGB", (10, 10), "black")
    mask = Image.new("L", (10, 10), 255)
    proposal = classify_annotation(annotation, image, mask)
    assert proposal.semantic_kind_confidence is None
    assert proposal.semantic_kind_state == "seed_label_unscored"
    assert proposal.identity_state == "abstained"
