import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import discover
import extract
import lettering_features as lf

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"


def _extract(source_file, layer_index, tmp_path):
    balloons = discover.discover_all(DATASET_DIR)
    balloon = next(
        b for b in balloons if b.source_file == source_file and b.layer_index == layer_index
    )
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)
    return tmp_path / paths[0]


def test_uniform_font_balloon_has_low_stroke_width_cv(tmp_path):
    # Known machine-set balloon (visually confirmed during Checkpoint B manual review).
    img_path = _extract("8a89f7d689fb441ea280cd782276bd7a.comics", 174, tmp_path)
    swcv, wobble = lf.compute_features_for_image(img_path)
    assert swcv < 0.5
    assert wobble < 1.0


def test_sfx_panel_has_high_stroke_width_cv(tmp_path):
    # The one genuine hand-drawn/dynamic-SFX panel found during the Checkpoint B outlier sweep
    # ("AHAHAHAHAHA" -- gradient-colored, slanted, jagged burst outline).
    img_path = _extract("96d4fcd2f634404494c1ffdef201b503.comics", 181, tmp_path)
    swcv, wobble = lf.compute_features_for_image(img_path)
    assert swcv > 1.0


def test_run_over_small_ocr_subset(tmp_path):
    # Build a tiny fake ocr.jsonl covering one known balloon and run the full stage.
    import json

    img_path = _extract("8a89f7d689fb441ea280cd782276bd7a.comics", 174, tmp_path)
    extracted_root = img_path.parent.parent
    ocr_jsonl = tmp_path / "ocr.jsonl"
    ocr_jsonl.write_text(
        json.dumps(
            {
                "source_file": "8a89f7d689fb441ea280cd782276bd7a.comics",
                "layer_index": 174,
                "lang_index": 0,
                "text": "AND AMBA TOLD PARASHURAMA",
                "confidence": 0.95,
                "needed_crop_fallback": False,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    features = lf.run(ocr_jsonl, extracted_root)
    assert len(features) == 1
    assert features[0].layer_index == 174
