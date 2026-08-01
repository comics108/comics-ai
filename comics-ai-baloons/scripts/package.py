#!/usr/bin/env python3
"""Stage 7.1: re-tile new-language renders and package a complete output .comics per source file.

For every balloon that matched a CSV row (stage 4) and isn't hand-lettered (stage 5), renders
every CSV-provided language that isn't already populated (en/ru always are; hi's slot exists but
is empty in every balloon in the dataset -- see Specifications), re-tiles each render, and patches
a copy of the source file's data.json with new Image entries at the language table's indices
(languages.py). Writes one full .comics file per source to work/output/ -- every original entry
copied byte-for-byte, plus the new tiles, plus the patched data.json. dataset/ is opened read-only
throughout.

Hand-lettered balloons are *not* rendered here (Track 6b is flag-only per Checkpoint B) but do get
RenderResult entries via render_handlettered.py so stage 7.2's report accounts for them.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from PIL import Image

import comics_io
import erase
import languages
import render_handlettered
import render_latin
import render_shaped
import tiling
from csv_loader import CsvRow, load_csv
from models import RenderResult

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_MATCHES = WORK_DIR / "matches.jsonl"
DEFAULT_LETTERING = WORK_DIR / "lettering.jsonl"
DEFAULT_CSV = None  # resolved via csv_loader.DEFAULT_CSV_PATH
DEFAULT_OUT_DIR = WORK_DIR / "output"
DEFAULT_RENDERS_OUT = WORK_DIR / "renders.jsonl"

ALREADY_POPULATED = {"en", "ru"}  # always present on a matched balloon; never re-rendered


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _pick_source_slot(layer: dict) -> tuple[int, dict] | None:
    """Pick the populated image slot to erase from -- prefer en (0), fall back to ru (1)."""
    images = layer.get("images", [])
    for idx in (0, 1):
        if idx < len(images) and images[idx].get("file"):
            return idx, images[idx]
    return None


def render_balloon(
    archive: comics_io.ComicsArchive,
    layer: dict,
    csv_row: CsvRow,
    source_file: str,
    layer_index: int,
) -> tuple[dict[int, tuple[Image.Image, int, int]], list[RenderResult]]:
    """Returns ({lang_index: (image, width, height)}, [RenderResult...]) for every target language
    not already populated.
    """
    picked = _pick_source_slot(layer)
    if picked is None:
        return {}, []
    src_idx, src_image_meta = picked

    src_bytes_name_prefix = "layers/"
    source_img = tiling.stitch_image(
        archive, src_image_meta["file"], src_image_meta["width"], src_image_meta["height"],
        layer_dir=src_bytes_name_prefix,
    )
    empty = erase.erase_text_single_image(source_img)

    new_images: dict[int, tuple[Image.Image, int, int]] = {}
    results: list[RenderResult] = []

    for lang_code, text in csv_row.texts.items():
        if lang_code in ALREADY_POPULATED:
            continue
        if not languages.is_known_language(lang_code):
            continue
        lang_index = languages.lang_to_index(lang_code)

        try:
            if lang_code in languages.COMPLEX_SCRIPT_LANGUAGES:
                rendered, fit_ok = render_shaped.render_text_on_balloon(empty, text, lang_code)
            else:
                rendered, fit_ok = render_latin.render_text_on_balloon(empty, text)
        except Exception as e:  # noqa: BLE001 -- render failures must not kill the whole file
            results.append(
                RenderResult(
                    source_file=source_file,
                    layer_index=layer_index,
                    lang_code=lang_code,
                    rendered=False,
                    reason=f"render error: {e}",
                )
            )
            continue

        if not fit_ok:
            results.append(
                RenderResult(
                    source_file=source_file,
                    layer_index=layer_index,
                    lang_code=lang_code,
                    rendered=False,
                    reason="text_overflow: did not fit interior rect even at minimum font size",
                )
            )
            continue

        new_images[lang_index] = (rendered, rendered.width, rendered.height)
        results.append(
            RenderResult(
                source_file=source_file,
                layer_index=layer_index,
                lang_code=lang_code,
                rendered=True,
                image_path=None,
                reason="",
            )
        )

    return new_images, results


def process_file(
    source_file: str,
    file_matches: list[dict],
    lettering_by_key: dict[tuple[str, int], str],
    csv_rows_by_id: dict[str, CsvRow],
    out_dir: Path,
) -> list[RenderResult]:
    archive = comics_io.ComicsArchive(DATASET_DIR / source_file)
    data = archive.read_data_json()
    layers = data["layers"]

    extra_files: dict[str, bytes] = {}
    all_results: list[RenderResult] = []

    for m in file_matches:
        layer_index = m["layer_index"]
        key = (source_file, layer_index)
        label = lettering_by_key.get(key, "machine_set")

        csv_row = csv_rows_by_id.get(m["csv_row_id"])
        if csv_row is None:
            continue

        if label == "hand_lettered":
            target_langs = [
                lang for lang in csv_row.texts if lang not in ALREADY_POPULATED
                and languages.is_known_language(lang)
            ]
            all_results.extend(
                render_handlettered.handle_hand_lettered_balloon(
                    source_file, layer_index, target_langs
                )
            )
            continue

        layer = layers[layer_index]
        new_images, results = render_balloon(archive, layer, csv_row, source_file, layer_index)
        all_results.extend(results)

        if not new_images:
            continue

        images_list = layer.setdefault("images", [])
        max_index = max(new_images)
        while len(images_list) <= max_index:
            images_list.append({})

        # Deterministic, convention-independent naming (deliberately not trying to mimic the
        # original filename, which is wildly inconsistent across the dataset -- see the
        # Specifications survey findings): unique per layer + language, always parseable.
        for lang_index, (img, w, h) in new_images.items():
            lang_code = languages.index_to_lang(lang_index)
            file_template = f"layer{layer_index}_{lang_code}_{{0}}_{{1}}_{{2}}.png"
            tiles = tiling.retile_image(img, file_template)
            for tile_name, tile_bytes in tiles.items():
                extra_files[f"layers/{tile_name}"] = tile_bytes
            images_list[lang_index] = {"file": file_template, "width": w, "height": h}

    if extra_files:
        out_path = out_dir / source_file
        comics_io.write_comics(
            out_path, archive, data_json_override=data, extra_files=extra_files
        )

    return all_results


def package_all(
    matches: list[dict],
    lettering: list[dict],
    csv_rows: list[CsvRow],
    out_dir: Path,
) -> list[RenderResult]:
    lettering_by_key = {(r["source_file"], r["layer_index"]): r["label"] for r in lettering}
    csv_rows_by_id = {r.row_id: r for r in csv_rows}

    by_file: dict[str, list[dict]] = defaultdict(list)
    for m in matches:
        if m["status"] == "matched":
            by_file[m["source_file"]].append(m)

    out_dir.mkdir(parents=True, exist_ok=True)
    all_results: list[RenderResult] = []
    for source_file, file_matches in sorted(by_file.items()):
        results = process_file(
            source_file, file_matches, lettering_by_key, csv_rows_by_id, out_dir
        )
        all_results.extend(results)
    return all_results


def write_jsonl(results: list[RenderResult], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--matches", default=str(DEFAULT_MATCHES))
    ap.add_argument("--lettering", default=str(DEFAULT_LETTERING))
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--renders-out", default=str(DEFAULT_RENDERS_OUT))
    args = ap.parse_args()

    matches = read_jsonl(Path(args.matches))
    lettering = read_jsonl(Path(args.lettering))
    csv_rows = load_csv()

    results = package_all(matches, lettering, csv_rows, Path(args.out_dir))
    render_shaped.shutdown_browser()
    write_jsonl(results, Path(args.renders_out))

    n_rendered = sum(1 for r in results if r.rendered)
    print(
        f"Packaged {len(list(Path(args.out_dir).glob('*.comics')))} files -> {args.out_dir} "
        f"({n_rendered}/{len(results)} language renders succeeded)"
    )


if __name__ == "__main__":
    main()
