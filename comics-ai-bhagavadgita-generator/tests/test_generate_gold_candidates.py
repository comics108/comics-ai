import sys
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from generate_gold_candidates import generate_panorama_gold_candidates, mask_metrics


def test_mask_metrics_distinguish_true_shape_from_bbox_fill():
    rectangle = Image.new("L", (20, 20), 255)
    shaped = Image.new("L", (20, 20), 0)
    ImageDraw.Draw(shaped).ellipse((2, 2, 17, 17), fill=255)

    assert mask_metrics(rectangle) == (1.0, 1.0)
    coverage, rectangularity = mask_metrics(shaped)
    assert .4 < coverage < .8
    assert rectangularity < .9


def test_panorama_candidates_are_checksummed_and_mapped_back_to_source(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source-document")
    render = tmp_path / "page.jpg"
    Image.new("RGB", (100, 50), "white").save(render)
    mask = tmp_path / "mask.png"
    shaped = Image.new("L", (20, 10), 0)
    ImageDraw.Draw(shaped).ellipse((1, 1, 18, 8), fill=255)
    shaped.save(mask)
    mask_sha = hashlib.sha256(mask.read_bytes()).hexdigest()
    candidate_manifest = tmp_path / "candidates.json"
    candidate_manifest.write_text(json.dumps([{
        "id": "mask-000",
        "bbox": [10, 5, 30, 15],
        "mask_file": str(mask),
        "mask_sha256": mask_sha,
        "agreement_iou": .91,
        "boundary_f1": .84,
        "coverage": .5,
        "rectangularity": .7,
        "method_families": ["instance_model", "edge_matting"],
        "review_resolution": [20, 10],
        "page_resolution": [100, 50],
        "coco_label": 1,
        "coco_score": .8,
    }]), encoding="utf-8")

    annotations = generate_panorama_gold_candidates(
        candidate_manifest,
        source_document=source,
        rendered_page=render,
        source_composition_id="panorama-02",
        source_resolution=(200, 100),
        repository_root=tmp_path,
    )

    assert len(annotations) == 1
    annotation = annotations[0]
    assert annotation.source_region == (20, 10, 40, 20)
    assert annotation.source_to_review_scale == (.5, .5)
    assert annotation.semantic_kind == "character"
    assert annotation.canonical_entity_id is None
    assert annotation.principal_character is False
    assert annotation.bitmap_mask_file == "mask.png"
    assert "identity_status:unresolved_nonprincipal" in annotation.review_evidence


def test_panorama_candidate_checksum_mismatch_fails_closed(tmp_path):
    source = tmp_path / "source.pdf"
    source.write_bytes(b"source-document")
    render = tmp_path / "page.jpg"
    Image.new("RGB", (100, 50), "white").save(render)
    mask = tmp_path / "mask.png"
    Image.new("L", (20, 10), 0).save(mask)
    candidate_manifest = tmp_path / "candidates.json"
    candidate_manifest.write_text(json.dumps([{
        "id": "mask-000", "bbox": [10, 5, 30, 15], "mask_file": str(mask),
        "mask_sha256": "0" * 64, "agreement_iou": .91, "boundary_f1": .84,
        "coverage": .5, "rectangularity": .7,
        "method_families": ["instance_model", "edge_matting"],
        "review_resolution": [20, 10], "page_resolution": [100, 50],
        "coco_label": 1, "coco_score": .8,
    }]), encoding="utf-8")

    try:
        generate_panorama_gold_candidates(
            candidate_manifest,
            source_document=source,
            rendered_page=render,
            source_composition_id="panorama-02",
            source_resolution=(200, 100),
            repository_root=tmp_path,
        )
    except ValueError as error:
        assert "checksum mismatch" in str(error)
    else:
        raise AssertionError("checksum mismatch must fail closed")
