#!/usr/bin/env python3
"""Stage 2: extract & stitch. For every balloon discovered in stage 1, stitch each populated
image slot's tiles into a single cached PNG under work/extracted/<source_stem>/.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import comics_io
import languages
import tiling
from models import BalloonLayer

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_DATASET_DIR = REPO_ROOT / "dataset"
WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_BALLOONS_JSONL = WORK_DIR / "balloons.jsonl"
DEFAULT_OUT_ROOT = WORK_DIR / "extracted"
DEFAULT_MANIFEST_OUT = WORK_DIR / "extracted.jsonl"


def read_balloons_jsonl(path: Path) -> list[BalloonLayer]:
    balloons = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            balloons.append(BalloonLayer.from_jsonable(json.loads(line)))
    return balloons


def lang_label(lang_idx: int) -> str:
    if 0 <= lang_idx < len(languages.LANGUAGES):
        return languages.index_to_lang(lang_idx)
    return f"idx{lang_idx}"


def extract_balloon(
    balloon: BalloonLayer,
    archive: comics_io.ComicsArchive,
    out_root: Path,
) -> dict[int, str]:
    """Stitch every populated slot for one balloon; returns {lang_index: relative png path}.
    Slots whose tiles are missing from the archive are skipped (logged by the caller), not fatal.
    """
    file_stem = Path(balloon.source_file).stem
    out_dir = out_root / file_stem
    out_dir.mkdir(parents=True, exist_ok=True)

    result: dict[int, str] = {}
    for lang_idx, slot in sorted(balloon.slots.items()):
        rel_path = f"{file_stem}/layer_{balloon.layer_index}_{lang_label(lang_idx)}.png"
        out_path = out_root / rel_path
        if not out_path.exists():
            try:
                img = tiling.stitch_image(archive, slot.file_template, slot.width, slot.height)
            except FileNotFoundError:
                continue
            img.save(out_path)
        result[lang_idx] = rel_path
    return result


def extract_all(
    balloons: list[BalloonLayer], dataset_dir: Path, out_root: Path
) -> list[dict]:
    manifest: list[dict] = []
    archive_cache: dict[str, comics_io.ComicsArchive] = {}
    missing_tiles = 0

    for balloon in balloons:
        archive = archive_cache.get(balloon.source_file)
        if archive is None:
            archive = comics_io.ComicsArchive(dataset_dir / balloon.source_file)
            archive_cache[balloon.source_file] = archive

        paths = extract_balloon(balloon, archive, out_root)
        if len(paths) < len(balloon.slots):
            missing_tiles += len(balloon.slots) - len(paths)
        manifest.append(
            {
                "source_file": balloon.source_file,
                "layer_index": balloon.layer_index,
                "paths": {str(k): v for k, v in paths.items()},
            }
        )

    if missing_tiles:
        print(f"WARNING: {missing_tiles} image slot(s) had missing tiles and were skipped")
    return manifest


def write_manifest(manifest: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--balloons", default=str(DEFAULT_BALLOONS_JSONL))
    ap.add_argument("--dataset-dir", default=str(DEFAULT_DATASET_DIR))
    ap.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    ap.add_argument("--manifest-out", default=str(DEFAULT_MANIFEST_OUT))
    args = ap.parse_args()

    balloons = read_balloons_jsonl(Path(args.balloons))
    manifest = extract_all(balloons, Path(args.dataset_dir), Path(args.out_root))
    write_manifest(manifest, Path(args.manifest_out))
    n_slots = sum(len(e["paths"]) for e in manifest)
    print(
        f"Extracted {n_slots} image slots from {len(manifest)} balloons -> {args.out_root} "
        f"(manifest: {args.manifest_out})"
    )


if __name__ == "__main__":
    main()
