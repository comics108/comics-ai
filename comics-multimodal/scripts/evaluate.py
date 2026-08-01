#!/usr/bin/env python3
"""Task 6.2 (revised): evaluate predicted regions (work/regions.jsonl) against each matched page's
`ground_truth_cluster` (work/alignment.jsonl).

**Why this is not pixel-level IoU**, despite Specifications' original Task 6.2 design saying so:
that design assumed a geometric mapping from a photographed page's pixels into canvas coordinates
would exist (the original whole-page-homography alignment design). Revision 1.1/1.2 (Checkpoint A)
established that no such mapping exists or is practically obtainable -- matching is content-based
(OCR + fuzzy text) and only tells us *which layers* are present in a photographed page, not *where*
in the photo's pixel space they are. This is a direct, foreseeable consequence of the already-
approved Revision 1.2 pivot (page-level, content-based matching), not a new design fork -- surfaced
here explicitly rather than silently reinterpreted.

What this script measures instead: for each matched page, compare the **kind distribution** (how
many regions of each kind) the model predicted against the true kind distribution of the matched
`ground_truth_cluster` -- a coarser but honestly-computable proxy for "did the model produce a
plausible decomposition of this kind of scene," not a claim of precise per-region localization
accuracy.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from render_canvas import GroundTruthRegion

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_ALIGNMENT = WORK_DIR / "alignment.jsonl"
DEFAULT_REGIONS = WORK_DIR / "regions.jsonl"
DEFAULT_CANVAS_DIR = WORK_DIR / "canvas"
DEFAULT_OUT = WORK_DIR / "eval_report.jsonl"

ALL_KINDS = ("art", "background", "character", "balloon")


@dataclass
class PageEvalResult:
    photo_file: str
    page_index: int
    episode_file: str
    predicted_counts: dict[str, int]
    ground_truth_counts: dict[str, int]
    per_kind_agreement: dict[str, float]
    mean_agreement: float


def _count_agreement(predicted: int, truth: int) -> float:
    """1.0 if counts match exactly, decaying toward 0 as they diverge; 1.0 (not undefined) when
    both are zero, since "predicted none, truth has none" is a correct outcome, not a gap.
    """
    if predicted == 0 and truth == 0:
        return 1.0
    return 1.0 - abs(predicted - truth) / max(predicted, truth, 1)


def load_ground_truth_kind_counts(
    episode_file: str, layer_indexes: list[int], canvas_dir: Path = DEFAULT_CANVAS_DIR
) -> dict[str, int]:
    gt_path = canvas_dir / f"{Path(episode_file).stem}.gt.json"
    if not gt_path.is_file():
        return {}
    ref_data = json.loads(gt_path.read_text())
    wanted = set(layer_indexes)
    counts: Counter[str] = Counter()
    for r in ref_data["regions"]:
        if r["layer_index"] in wanted:
            counts[r["kind"]] += 1
    return dict(counts)


def evaluate_page(
    photo_file: str,
    page_index: int,
    episode_file: str,
    ground_truth_layer_indexes: list[int],
    predicted_regions: list[dict],
    canvas_dir: Path = DEFAULT_CANVAS_DIR,
) -> PageEvalResult:
    predicted_counts = Counter(r["predicted_kind"] for r in predicted_regions)
    truth_counts = load_ground_truth_kind_counts(episode_file, ground_truth_layer_indexes, canvas_dir)

    per_kind_agreement = {}
    for kind in ALL_KINDS:
        per_kind_agreement[kind] = _count_agreement(predicted_counts.get(kind, 0), truth_counts.get(kind, 0))
    mean_agreement = sum(per_kind_agreement.values()) / len(per_kind_agreement)

    return PageEvalResult(
        photo_file=photo_file,
        page_index=page_index,
        episode_file=episode_file,
        predicted_counts={k: predicted_counts.get(k, 0) for k in ALL_KINDS},
        ground_truth_counts={k: truth_counts.get(k, 0) for k in ALL_KINDS},
        per_kind_agreement=per_kind_agreement,
        mean_agreement=mean_agreement,
    )


def evaluate_all(
    alignment_path: Path = DEFAULT_ALIGNMENT,
    regions_path: Path = DEFAULT_REGIONS,
    canvas_dir: Path = DEFAULT_CANVAS_DIR,
    out_path: Path = DEFAULT_OUT,
) -> list[PageEvalResult]:
    regions_by_page: dict[tuple[str, int], list[dict]] = defaultdict(list)
    with regions_path.open() as f:
        for line in f:
            r = json.loads(line)
            regions_by_page[(r["photo_file"], r["page_index"])].append(r)

    results = []
    with alignment_path.open() as f:
        for line in f:
            entry = json.loads(line)
            if entry["status"] != "matched":
                continue
            key = (entry["photo_file"], entry["page_index"])
            predicted = regions_by_page.get(key, [])
            result = evaluate_page(
                entry["photo_file"],
                entry["page_index"],
                entry["episode_file"],
                entry["ground_truth_cluster"],
                predicted,
                canvas_dir,
            )
            results.append(result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    results = evaluate_all(out_path=args.out)
    if results:
        mean_overall = sum(r.mean_agreement for r in results) / len(results)
        print(f"{len(results)} pages evaluated -> {args.out}")
        print(f"mean kind-count agreement across all pages: {mean_overall:.3f}")
    else:
        print("no matched pages to evaluate")


if __name__ == "__main__":
    main()
