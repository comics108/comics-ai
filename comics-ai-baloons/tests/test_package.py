import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import comics_io
import package
import tiling
from csv_loader import CsvRow


def test_package_all_writes_output_and_patches_data_json(tmp_path):
    matches = [
        {
            "source_file": "8a89f7d689fb441ea280cd782276bd7a.comics",
            "layer_index": 174,
            "csv_row_id": "TEST_ROW",
            "status": "matched",
        }
    ]
    lettering = [
        {
            "source_file": "8a89f7d689fb441ea280cd782276bd7a.comics",
            "layer_index": 174,
            "label": "machine_set",
        }
    ]
    csv_rows = [
        CsvRow(
            row_id="TEST_ROW",
            bubble_type="speech",
            chapter_label="",
            texts={
                "en": "irrelevant",
                "ru": "irrelevant",
                "uk": "Тестовий переклад для перевірки пакування.",
                "zh": "用于测试打包的测试翻译。",
            },
        )
    ]

    out_dir = tmp_path / "output"
    results = package.package_all(matches, lettering, csv_rows, out_dir)

    rendered = {r.lang_code: r for r in results if r.rendered}
    assert set(rendered) == {"uk", "zh"}  # en/ru already populated, hi/th/etc not in this row

    out_file = out_dir / "8a89f7d689fb441ea280cd782276bd7a.comics"
    assert out_file.exists()

    out_archive = comics_io.ComicsArchive(out_file)
    data = out_archive.read_data_json()
    images = data["layers"][174]["images"]
    assert images[0]["file"].startswith("b1_eng_")  # untouched
    assert images[1]["file"].startswith("b1_ru_")  # untouched
    uk_index = 3  # per languages.py table
    zh_index = 5
    assert images[uk_index]["file"].startswith("layer174_uk_")
    assert images[zh_index]["file"].startswith("layer174_zh_")

    # the new tiles must actually stitch back to a real image
    img = tiling.stitch_image(
        out_archive, images[uk_index]["file"], images[uk_index]["width"], images[uk_index]["height"]
    )
    assert img.size == (images[uk_index]["width"], images[uk_index]["height"])


def test_hand_lettered_balloon_is_not_rendered(tmp_path):
    matches = [
        {
            "source_file": "96d4fcd2f634404494c1ffdef201b503.comics",
            "layer_index": 181,
            "csv_row_id": "TEST_ROW",
            "status": "matched",
        }
    ]
    lettering = [
        {
            "source_file": "96d4fcd2f634404494c1ffdef201b503.comics",
            "layer_index": 181,
            "label": "hand_lettered",
        }
    ]
    csv_rows = [
        CsvRow(
            row_id="TEST_ROW",
            bubble_type="speech",
            chapter_label="",
            texts={"en": "irrelevant", "ru": "irrelevant", "uk": "Тест"},
        )
    ]

    out_dir = tmp_path / "output"
    results = package.package_all(matches, lettering, csv_rows, out_dir)

    assert len(results) == 1
    assert results[0].rendered is False
    assert "manual" in results[0].reason.lower()
    # no renders were produced -> no output file needed for this balloon
    assert not (out_dir / "96d4fcd2f634404494c1ffdef201b503.comics").exists()


def test_unmatched_status_is_ignored():
    matches = [
        {
            "source_file": "8a89f7d689fb441ea280cd782276bd7a.comics",
            "layer_index": 174,
            "csv_row_id": None,
            "status": "skipped_no_match",
        }
    ]
    results = package.package_all(matches, [], [], Path("/tmp/should_not_be_created_xyz"))
    assert results == []
