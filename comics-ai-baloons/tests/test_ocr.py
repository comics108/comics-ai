import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import discover
import extract
import ocr

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
SAMPLE_FILE = "8a89f7d689fb441ea280cd782276bd7a.comics"


def _known_b1_balloon():
    balloons = discover.discover_all(DATASET_DIR)
    for b in balloons:
        if b.source_file == SAMPLE_FILE and 0 in b.slots and b.slots[0].file_template.startswith("b1_eng_"):
            return b
    raise AssertionError("expected balloon not found")


def test_ocr_recovers_known_text(tmp_path):
    balloon = _known_b1_balloon()
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)

    text, conf, needed_fallback = ocr.ocr_image(tmp_path / paths[0], "eng")
    normalized = text.upper()
    # Known ground truth from manual inspection during Requirements: "AND AMBA TOLD PARASHURAMA
    # ABOUT ALL THE HARDSHIPS SHE HAD FACED WITH BHISHMA." -- allow OCR noise, just check the
    # distinctive proper nouns come through.
    assert "AMBA" in normalized
    assert "PARASHURAMA" in normalized
    assert "BHISHMA" in normalized
    assert conf > 0.5
    assert needed_fallback is False  # this balloon OCRs fine directly, no fallback needed


def test_outline_strip_fallback_recovers_short_balloon(tmp_path):
    # Known-hard case found during Checkpoint A: a short balloon ("HERE HE IS.") whose outline
    # confuses Tesseract's layout analysis enough that direct OCR returns empty text even under
    # --psm 6, but the outline-stripped crop recovers it correctly.
    balloons = discover.discover_all(DATASET_DIR)
    balloon = next(
        b
        for b in balloons
        if b.source_file == "6c690c679511407cb558a0dc347fdebf.comics"
        and b.layer_index == 102
    )
    archive = comics_io.ComicsArchive(DATASET_DIR / balloon.source_file)
    paths = extract.extract_balloon(balloon, archive, tmp_path)

    text, conf, needed_fallback = ocr.ocr_image(tmp_path / paths[0], "eng")
    assert "HERE HE IS" in text.upper()
    assert needed_fallback is True


def test_run_ocr_over_manifest(tmp_path):
    balloons = discover.discover_all(DATASET_DIR)[:3]
    manifest = extract.extract_all(balloons, DATASET_DIR, tmp_path)
    results = ocr.run_ocr(manifest, tmp_path)

    # each balloon has en (idx 0) + ru (idx 1) slots -> 2 OcrResults each
    assert len(results) == 2 * len(balloons)
    assert {r.lang_index for r in results} == {0, 1}


def test_write_jsonl(tmp_path):
    balloons = discover.discover_all(DATASET_DIR)[:2]
    manifest = extract.extract_all(balloons, DATASET_DIR, tmp_path)
    results = ocr.run_ocr(manifest, tmp_path)
    out_path = tmp_path / "ocr.jsonl"
    ocr.write_jsonl(results, out_path)
    lines = out_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == len(results)
