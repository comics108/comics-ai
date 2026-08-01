"""Thin adapter reusing apps/comics-ai/comics-ai-baloons/scripts/ instead of duplicating it.

comics-ai-baloons is read and invoked here, never modified (flows/sdd-comics-ai-multimodal/
02-specifications.md "Integration Points", "Affected Systems"). Its modules are flat (no package
__init__), imported the same way its own tests do: inject its scripts/ dir onto sys.path.

Note: comics-ai-baloons/scripts/discover.py's own REPO_ROOT (Path(__file__).parents[3]) resolves to
apps/, not the true repo root -- a stale off-by-one from before that app was nested under
apps/comics-ai/. We never rely on its default dataset-dir argument for this reason; every call site
in this bridge passes an explicit path.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "dataset"
BALOONS_APP_DIR = REPO_ROOT / "apps" / "comics-ai" / "comics-ai-baloons"
BALOONS_SCRIPTS_DIR = BALOONS_APP_DIR / "scripts"

# dataset/*.comics files are NOT flat -- they live under nested per-book/per-chapter directories
# (e.g. dataset/boranko/mahabharata/book1/comics_interactive/*.comics). Never assume a flat layout.


def find_comics_files(root: Path = DATASET_DIR) -> list[Path]:
    return sorted(root.rglob("*.comics"))

if not BALOONS_SCRIPTS_DIR.is_dir():
    raise RuntimeError(f"comics-ai-baloons scripts dir not found: {BALOONS_SCRIPTS_DIR}")

# append, not insert(0, ...): a defensive default so our own project's modules stay authoritative
# for any future name collision with comics-ai-baloons' scripts (discovered the hard way in Phase
# 5: comics-ai-baloons/scripts/models.py, a single shared-dataclasses file, collided with what was
# originally our own scripts/models/ package -- fixed by renaming ours to scripts/segmenter_models/
# rather than fighting sys.path/sys.modules ordering, since comics-ai-baloons' match.py itself
# needs its own models.py importable as plain "models"). Whichever "models" resolves first gets
# cached in sys.modules for the rest of the process, so getting this backwards would silently break
# one side or the other.
if str(BALOONS_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(BALOONS_SCRIPTS_DIR))

import comics_io  # noqa: E402  (comics-ai-baloons module, path injected above)
import tiling  # noqa: E402

ComicsArchive = comics_io.ComicsArchive
write_comics = comics_io.write_comics
stitch_image = tiling.stitch_image
retile_image = tiling.retile_image
tile_grid = tiling.tile_grid
tile_filename = tiling.tile_filename
TILE_SIZE = tiling.TILE_SIZE


def is_balloon_layer(layer: dict) -> bool:
    """Single source of truth for the structural balloon rule, shared with discover.py's own
    definition (>=2 non-empty images[] entries) rather than a second, drifting copy of it.
    """
    images = layer.get("images", [])
    populated = [im for im in images if im.get("file")]
    return len(populated) >= 2
