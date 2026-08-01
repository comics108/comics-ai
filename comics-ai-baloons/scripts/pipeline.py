#!/usr/bin/env python3
"""Stage 7.3: orchestrator -- runs every stage in order, resumable per-stage.

Each stage is invoked as a subprocess (matching how a user would run it standalone -- avoids
cross-stage global-state issues, e.g. render_shaped.py's browser singleton living cleanly inside
package.py's own process). A stage is skipped if its primary output already exists, unless
--force is passed.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
WORK_DIR = SCRIPTS_DIR.parent / "work"

# (stage name, script filename, primary output path to check for resumability)
STAGES: list[tuple[str, str, Path]] = [
    ("discover", "discover.py", WORK_DIR / "balloons.jsonl"),
    ("extract", "extract.py", WORK_DIR / "extracted.jsonl"),
    ("ocr", "ocr.py", WORK_DIR / "ocr.jsonl"),
    ("match", "match.py", WORK_DIR / "matches.jsonl"),
    ("lettering_features", "lettering_features.py", WORK_DIR / "lettering_features.jsonl"),
    ("classify", "classify.py", WORK_DIR / "lettering.jsonl"),
    ("package", "package.py", WORK_DIR / "renders.jsonl"),
    ("report", "report.py", WORK_DIR / "report.jsonl"),
]


def run_stage(name: str, script: str, output: Path, force: bool) -> None:
    if output.exists() and not force:
        print(f"[{name}] SKIP (already exists: {output})")
        return

    print(f"[{name}] running {script} ...")
    result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script)])
    if result.returncode != 0:
        print(f"[{name}] FAILED (exit {result.returncode})")
        sys.exit(result.returncode)
    print(f"[{name}] done")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--force", action="store_true", help="re-run every stage, ignore existing output")
    ap.add_argument(
        "--only", nargs="*", default=None, help="run only these stage names (space-separated)"
    )
    args = ap.parse_args()

    stages = STAGES if not args.only else [s for s in STAGES if s[0] in args.only]
    if not stages:
        print(f"No stages matched --only {args.only}; valid names: {[s[0] for s in STAGES]}")
        sys.exit(1)

    for name, script, output in stages:
        run_stage(name, script, output, args.force)

    print("Pipeline complete.")


if __name__ == "__main__":
    main()
