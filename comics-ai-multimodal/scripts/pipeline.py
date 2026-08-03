#!/usr/bin/env python3
"""Task 11.1: single orchestrator running every pipeline stage in order, resumable per-stage
(skips a stage whose cached output already exists unless --force is passed) -- matches
comics-ai-baloons' orchestrator pattern.

Order: render_canvas -> augment -> train_segmenter (unet) -> align_photo -> infer_segmenter ->
evaluate -> route_balloons -> build_library -> package -> report. `dataset/` is never touched by
any stage.
"""

from __future__ import annotations

import argparse
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parent.parent / "work"


def run_stage(name: str, output_marker: Path, fn, force: bool) -> None:
    if output_marker.exists() and not force:
        print(f"[skip] {name} (found: {output_marker})")
        return
    print(f"[run ] {name}")
    fn()
    print(f"[done] {name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="re-run every stage even if cached output exists")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="skip the (slow) model-training stage and use whatever checkpoint already exists",
    )
    args = parser.parse_args()

    import align_photo
    import augment
    import build_library
    import evaluate
    import infer_segmenter
    import package
    import render_canvas
    import report
    import route_balloons
    import train_segmenter

    run_stage("render_canvas", WORK_DIR / "canvas", render_canvas.render_all, args.force)
    run_stage(
        "augment", WORK_DIR / "train_pairs" / "manifest.jsonl", augment.build_training_pairs, args.force
    )
    if not args.skip_training:
        run_stage(
            "train_segmenter (unet)",
            WORK_DIR / "models" / "unet_baseline.pt",
            train_segmenter.train,
            args.force,
        )
    run_stage("align_photo", WORK_DIR / "alignment.jsonl", align_photo.align_all, args.force)
    run_stage("infer_segmenter", WORK_DIR / "regions.jsonl", infer_segmenter.infer_all, args.force)
    run_stage("evaluate", WORK_DIR / "eval_report.jsonl", evaluate.evaluate_all, args.force)
    run_stage(
        "route_balloons", WORK_DIR / "balloon_handoff.jsonl", route_balloons.route_all, args.force
    )
    run_stage("build_library", WORK_DIR / "library", build_library.build_library, args.force)
    run_stage("package", WORK_DIR / "output", package.package_all, args.force)
    run_stage("report", WORK_DIR / "report.jsonl", report.write_report, args.force)

    print("pipeline complete.")


if __name__ == "__main__":
    main()
