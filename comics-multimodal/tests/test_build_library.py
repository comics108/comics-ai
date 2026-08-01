import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import build_library as bl  # noqa: E402


def test_seed_name_from_episode_token_strips_leading_order_number_and_possessive_s():
    assert bl.seed_name_from_episode_token("21_ambas_plea") == "amba"


def test_seed_name_from_episode_token_handles_no_digit_prefix():
    assert bl.seed_name_from_episode_token("hastinapur") == "hastinapur"


def test_seed_name_from_episode_token_short_word_not_stripped():
    # "us" is too short (<=3 chars after context) to safely strip a trailing 's' as possessive
    assert bl.seed_name_from_episode_token("1_us") == "us"


def test_load_episode_seed_names_parses_real_csv_if_present():
    if not bl.DEFAULT_EPISODES_CSV.is_file():
        pytest.skip("Comics_Episodes.csv not present")
    seeds = bl.load_episode_seed_names()
    assert seeds.get("8a89f7d689fb441ea280cd782276bd7a.comics") == "amba"


def test_load_episode_seed_names_from_fixture(tmp_path):
    csv_path = tmp_path / "episodes.csv"
    csv_path.write_text(
        "Id,SeasonId,NameTokenId,Image,File,Version,Product,Date,Order\n"
        "1,1,1,/Images/x*.jpg,/Files/abc.comics,1,1_foo_bar,2020-01-01,1\n"
        "2,1,2,/Images/y*.jpg,/Files/def.comics,1,NULL,2020-01-02,2\n"
    )
    seeds = bl.load_episode_seed_names(csv_path)
    assert seeds == {"abc.comics": "foo"}


def test_compute_embedding_is_unit_normalized_and_stable_shape():
    crop = np.random.randint(0, 255, size=(80, 60, 3), dtype=np.uint8)
    emb = bl.compute_embedding(crop)
    assert emb.shape == (512,)
    assert abs(np.linalg.norm(emb) - 1.0) < 1e-4


def test_compute_embedding_similar_images_are_closer_than_dissimilar():
    # Pure random noise has no natural-image structure for an ImageNet-pretrained ResNet to
    # discriminate (its features cluster tightly for noise regardless of exact pixels) -- use
    # structured images (solid-ish colors with simple shapes) instead, a fairer test of real
    # discriminative ability.
    def _solid_with_shape(color: tuple[int, int, int]) -> np.ndarray:
        img = np.full((120, 120, 3), color, dtype=np.uint8)
        img[40:80, 40:80] = (255, 255, 255)  # a white square landmark
        return img

    base = _solid_with_shape((200, 30, 30))  # red-ish
    similar = _solid_with_shape((210, 35, 25))  # slightly different red-ish
    different = _solid_with_shape((20, 30, 200))  # blue-ish

    e_base = bl.compute_embedding(base)
    e_similar = bl.compute_embedding(similar)
    e_different = bl.compute_embedding(different)

    dist_similar = 1 - float(np.dot(e_base, e_similar))
    dist_different = 1 - float(np.dot(e_base, e_different))
    assert dist_similar < dist_different


def test_build_library_routes_low_confidence_regions_to_unclustered_not_silently_dropped(
    tmp_path, monkeypatch
):
    # Regression test: an earlier version defined an `unclustered_dir` but never actually wrote to
    # it -- low-confidence regions were silently skipped instead of landing in a reviewable bucket.
    fake_crop = np.full((20, 20, 3), 100, dtype=np.uint8)
    monkeypatch.setattr(bl, "extract_crop_image", lambda *a, **k: fake_crop)
    monkeypatch.setattr(bl, "compute_embedding", lambda crop: np.ones(512) / np.sqrt(512))

    alignment_path = tmp_path / "alignment.jsonl"
    alignment_path.write_text(
        json.dumps(
            {
                "photo_file": "p1.jpg",
                "page_index": 0,
                "episode_file": "ep1.comics",
                "status": "matched",
                "ground_truth_cluster": [],
                "matched_layer_indexes": [],
                "confidence": 0.9,
                "reason": "",
            }
        )
        + "\n"
    )
    regions_path = tmp_path / "regions.jsonl"
    regions_path.write_text(
        json.dumps(
            {"photo_file": "p1.jpg", "page_index": 0, "predicted_kind": "character", "confidence": 0.1, "bbox": [0, 0, 10, 10]}
        )
        + "\n"
    )
    episodes_csv = tmp_path / "episodes.csv"
    episodes_csv.write_text("Id,SeasonId,NameTokenId,Image,File,Version,Product,Date,Order\n")

    out_dir = tmp_path / "library"
    bl.build_library(
        regions_path=regions_path,
        alignment_path=alignment_path,
        lowcamera_dir=tmp_path,
        episodes_csv=episodes_csv,
        out_dir=out_dir,
    )

    unclustered = list((out_dir / "characters" / "unclustered").glob("*.png"))
    assert len(unclustered) == 1
    # And no spurious named identity folder was created for this low-confidence-only episode.
    named_dirs = [
        d for d in (out_dir / "characters").iterdir() if d.is_dir() and d.name != "unclustered"
    ]
    assert named_dirs == []


def test_build_library_real_data_amba_folder_is_pure_and_reasonable_if_present():
    if not bl.DEFAULT_REGIONS.is_file() or not bl.DEFAULT_ALIGNMENT.is_file():
        pytest.skip("work/regions.jsonl or work/alignment.jsonl not present -- run the pipeline first")

    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        out_dir = Path(tmp) / "library"
        result = bl.build_library(out_dir=out_dir)
        assert "character" in result

        amba_dir = out_dir / "characters" / "amba"
        if amba_dir.is_dir():
            crops = list(amba_dir.glob("*.png"))
            assert len(crops) >= 1
            # Every crop filename should trace back to a photo_file whose alignment.jsonl entry
            # matched episode 21 (ambas_plea) -- confirms no cross-contamination from other
            # episodes into the "amba" identity folder.
            alignment = {}
            with bl.DEFAULT_ALIGNMENT.open() as f:
                for line in f:
                    d = json.loads(line)
                    if d["status"] == "matched":
                        alignment[(d["photo_file"], d["page_index"])] = d["episode_file"]
            for crop_path in crops:
                stem = crop_path.stem  # "<photo_stem>_p<page_index>_<i>"
                photo_stem, page_part, _ = stem.rsplit("_", 2)
                page_index = int(page_part[1:])
                episode = alignment.get((photo_stem + ".jpg", page_index))
                assert episode == "8a89f7d689fb441ea280cd782276bd7a.comics", (
                    f"{crop_path.name} traces to episode {episode}, not Amba's episode 21"
                )
