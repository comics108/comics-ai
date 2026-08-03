#!/usr/bin/env python3
"""Criterion 1: held-out (file-wise) evaluation of the calibrated baseline against real ground
truth -- same file-wise-split discipline as `sdd-comics-ai-positioning`'s
`evaluate_positioning.py`, so stats calibration never leaks a held-out file's own reveal patterns
into its own evaluation.

Metric, per property: occurrence accuracy (did the baseline correctly predict animate-or-not,
against real ground truth for every real layer of the 27 files) -- the only metric that applies to
all four properties uniformly, given translate/rotate's confirmed lack of a predictable direction
(see transform_stats.py). Duration error (predicted vs. real end-start) is reported only for
layers where *both* baseline and ground truth agree the property occurs, since duration isn't
meaningful to compare otherwise.
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from baseline_transform import propose_reveal
from transform_stats import DEFAULT_PAIRS, PROPERTIES, compute_stats, load_pairs

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_OUT = WORK_DIR / "eval_report.json"

HELD_OUT_FRACTION = 0.25


def choose_held_out(episode_stems: list[str], fraction: float = HELD_OUT_FRACTION) -> set[str]:
    """Deterministic (no randomness), same convention as sdd-comics-ai-positioning's own
    choose_held_out: every Nth stem, alphabetically."""
    ordered = sorted(set(episode_stems))
    step = max(1, round(1 / fraction))
    return set(ordered[::step])


def evaluate(pairs_path: Path = DEFAULT_PAIRS) -> dict:
    all_pairs = load_pairs(pairs_path)
    held_out = choose_held_out([p["episode_stem"] for p in all_pairs])
    stats = compute_stats(exclude_episode_stems=held_out, pairs_path=pairs_path)

    held_out_pairs = [p for p in all_pairs if p["episode_stem"] in held_out]

    per_property = {
        prop: {"correct": 0, "strawman_correct": 0, "total": 0, "duration_errors": []}
        for prop in PROPERTIES
    }

    for pair in held_out_pairs:
        proposal = propose_reveal(pair["kind"], stats)
        for prop in PROPERTIES:
            actual = pair["reveal"].get(prop)
            actual_occurs = actual is not None
            predicted = proposal[prop]

            per_property[prop]["total"] += 1
            if predicted.occurs == actual_occurs:
                per_property[prop]["correct"] += 1
            # Strawman: "always predict no animation" -- the trivial comparison this repo's own
            # precedent (sdd-comics-ai-positioning) always checks a calibrated baseline against.
            if not actual_occurs:
                per_property[prop]["strawman_correct"] += 1
            if predicted.occurs and actual_occurs:
                actual_duration = actual["end"] - actual["start"]
                predicted_duration = predicted.end - predicted.start
                per_property[prop]["duration_errors"].append(abs(actual_duration - predicted_duration))

    result = {"held_out_episodes": sorted(held_out), "held_out_layer_count": len(held_out_pairs)}
    for prop, data in per_property.items():
        result[prop] = {
            "occurrence_accuracy": data["correct"] / data["total"] if data["total"] else None,
            "strawman_accuracy": data["strawman_correct"] / data["total"] if data["total"] else None,
            "n": data["total"],
            "median_duration_error_when_both_occur": (
                statistics.median(data["duration_errors"]) if data["duration_errors"] else None
            ),
            "n_both_occur": len(data["duration_errors"]),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    result = evaluate()
    args.out.write_text(json.dumps(result, indent=2))
    print(f"Held-out episodes: {len(result['held_out_episodes'])}, layers: {result['held_out_layer_count']}")
    for prop in PROPERTIES:
        r = result[prop]
        line = (
            f"  {prop}: occurrence_accuracy={r['occurrence_accuracy']*100:.1f}% "
            f"(strawman={r['strawman_accuracy']*100:.1f}%, n={r['n']})"
        )
        if r["median_duration_error_when_both_occur"] is not None:
            line += f", median_duration_error={r['median_duration_error_when_both_occur']:.0f}px (n={r['n_both_occur']})"
        print(line)


if __name__ == "__main__":
    main()
