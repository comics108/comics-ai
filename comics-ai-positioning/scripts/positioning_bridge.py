"""Thin adapter reusing apps/comics-ai/comics-multimodal/ outputs instead of duplicating them.

comics-multimodal is read here, never modified (flows/sdd-comics-ai-positioning/
02-specifications.md "Affected Systems"). Its per-episode ground truth (work/canvas/*.gt.json) and
photo alignment (work/alignment.jsonl) are already-materialized JSON -- for the Must-Have path
(Phases 1-4) we read those files directly rather than re-invoking comics-multimodal's Python
functions, which sidesteps the sys.path/module-collision care baloons_bridge.py (comics-multimodal's
own adapter for comics-ai-baloons) had to take. `import_multimodal_module()` below is kept for
Phase 5/7 (learned model, page-number anchor), which may need to call comics-multimodal functions
(e.g. resting_position.resolve_resting_transform) directly for scale/alpha, not just read bbox.

Deviation from Plan Task 1.1 (logged in flows/sdd-comics-ai-positioning/04-implementation-log.md):
the plan described this module as importing render_canvas/resting_position/align_photo/
kind_heuristic directly. In practice their outputs are already serialized to work/canvas/*.gt.json
and work/alignment.jsonl, which is a strictly more robust dependency (no cross-project sys.path
collision risk, works even if comics-multimodal's own heavier deps like torch/opencv aren't
importable in this environment) for everything Phases 1-4 need. The live-import path is kept
available, not deleted, for later phases that genuinely need it.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterator

REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "dataset"
MULTIMODAL_APP_DIR = REPO_ROOT / "apps" / "comics-ai" / "comics-multimodal"
MULTIMODAL_SCRIPTS_DIR = MULTIMODAL_APP_DIR / "scripts"
MULTIMODAL_WORK_DIR = MULTIMODAL_APP_DIR / "work"
CANVAS_DIR = MULTIMODAL_WORK_DIR / "canvas"
ALIGNMENT_JSONL = MULTIMODAL_WORK_DIR / "alignment.jsonl"
REGIONS_JSONL = MULTIMODAL_WORK_DIR / "regions.jsonl"

if not MULTIMODAL_WORK_DIR.is_dir():
    raise RuntimeError(
        f"comics-multimodal work dir not found: {MULTIMODAL_WORK_DIR} -- run that pipeline first"
    )


def require_canvas_ground_truth() -> None:
    """Fail fast, with a clear message, if comics-multimodal's canvas stage hasn't been run."""
    if not CANVAS_DIR.is_dir() or not any(CANVAS_DIR.glob("*.gt.json")):
        raise RuntimeError(
            f"No *.gt.json files under {CANVAS_DIR} -- run comics-multimodal's render_canvas.py "
            "(Phase 2 of that flow) before using this bridge."
        )


def load_canvas_reference(episode_file: str) -> dict:
    """Read one comics-multimodal work/canvas/<stem>.gt.json (CanvasReference) as a plain dict."""
    stem = Path(episode_file).stem
    path = CANVAS_DIR / f"{stem}.gt.json"
    if not path.is_file():
        raise FileNotFoundError(f"No ground-truth canvas for {episode_file!r}: {path}")
    return json.loads(path.read_text())


def load_all_canvas_references() -> dict[str, dict]:
    """Keyed by episode stem (e.g. "096e...d7a", matching Path("096e...d7a.comics").stem) -- NOT
    `Path.stem`, which only strips one suffix and would return "096e...d7a.gt" for a
    "096e...d7a.gt.json" filename (a real bug caught by
    test_spacing_stats.py::test_compute_stats_exclude_reduces_counts: an exclude-by-stem filter
    silently matched nothing because of this exact mismatch).
    """
    require_canvas_ground_truth()
    return {
        p.name.removesuffix(".gt.json"): json.loads(p.read_text())
        for p in sorted(CANVAS_DIR.glob("*.gt.json"))
    }


def iter_alignment_rows(path: Path | None = None) -> Iterator[dict]:
    """Yield each row of comics-multimodal's work/alignment.jsonl (PageAlignmentResult dicts).

    `path` defaults to the *current* value of the module-level ALIGNMENT_JSONL, looked up inside
    the function body rather than bound as a parameter default -- a parameter default would be
    frozen at import time, which would silently ignore a test's `monkeypatch.setattr(pb,
    "ALIGNMENT_JSONL", ...)` (a real mistake caught by test_data_checkpoint.py's fixture test
    during Task 1.2's implementation, not a hypothetical).
    """
    resolved = path if path is not None else ALIGNMENT_JSONL
    if not resolved.is_file():
        raise FileNotFoundError(f"No alignment output: {resolved}")
    with resolved.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def iter_matched_alignment_rows(path: Path | None = None) -> Iterator[dict]:
    """Only rows that actually matched an episode with a non-empty ground_truth_cluster."""
    for row in iter_alignment_rows(path):
        if row.get("status") == "matched" and row.get("ground_truth_cluster"):
            yield row


def import_multimodal_module(name: str):
    """Live-import a comics-multimodal script module (e.g. "resting_position") for phases that
    need to call its functions directly, not just read its already-serialized output. Appends
    (not inserts(0, ...)) so this project's own modules stay authoritative on any name collision --
    same defensive convention comics-multimodal's own baloons_bridge.py uses.
    """
    if str(MULTIMODAL_SCRIPTS_DIR) not in sys.path:
        sys.path.append(str(MULTIMODAL_SCRIPTS_DIR))
    import importlib

    return importlib.import_module(name)
