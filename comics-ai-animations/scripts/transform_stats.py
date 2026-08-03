#!/usr/bin/env python3
"""Criterion 1: mine real per-kind reveal-animation statistics from all 27 files' real
`transform_pairs.jsonl` (Task: build_transform_pairs.py), to calibrate the baseline with real
numbers instead of guessed constants -- same precedent as `sdd-comics-ai-positioning`'s
spacing_stats.py.

Real, checked facts this calibration is built on (2026-08-02, verified against the full real
dataset before writing any baseline code):
- Occurrence varies sharply by kind: e.g. balloons animate alpha 76.8% / scale 75.3% of the time;
  backgrounds animate translate only 32.5% and almost never scale/alpha (1-1.5%).
- Alpha reveals are overwhelmingly a fade-in: real median from/to = 0.0 -> 1.0.
- Scale reveals are overwhelmingly a grow-in: real median from/to = 0.6 -> 1.0.
- Translate/rotate reveals have a real, confident occurrence + duration signal, but **no
  confident direction/magnitude** -- median delta is ~0 with a roughly balanced positive/negative
  split (894 positive vs. 849 negative dy, out of 2368 translate reveals). Disclosed as a real
  limitation, not silently guessed with a fake default direction.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_PAIRS = WORK_DIR / "transform_pairs.jsonl"
DEFAULT_OUT = WORK_DIR / "transform_stats.json"

PROPERTIES = ("translate", "scale", "rotate", "alpha")


def load_pairs(path: Path = DEFAULT_PAIRS) -> list[dict]:
    pairs = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                pairs.append(json.loads(line))
    return pairs


def compute_stats(exclude_episode_stems: set[str] | None = None, pairs_path: Path = DEFAULT_PAIRS) -> dict:
    """`exclude_episode_stems` lets held-out evaluation compute stats from the training portion
    only -- same leak-avoidance discipline as `sdd-comics-ai-positioning`'s spacing_stats.py."""
    pairs = load_pairs(pairs_path)
    if exclude_episode_stems:
        pairs = [p for p in pairs if p["episode_stem"] not in exclude_episode_stems]

    counts_by_kind: dict[str, int] = defaultdict(int)
    occurs_by_kind: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    durations: dict[str, list[int]] = defaultdict(list)
    alpha_values: list[tuple[float, float]] = []
    scale_values: list[tuple[float, float]] = []

    for pair in pairs:
        kind = pair["kind"]
        counts_by_kind[kind] += 1
        reveal = pair["reveal"]
        for prop in PROPERTIES:
            entry = reveal.get(prop)
            if entry is not None:
                occurs_by_kind[kind][prop] += 1
                durations[prop].append(entry["end"] - entry["start"])
                if prop == "alpha":
                    alpha_values.append((entry["from_value"]["alpha"], entry["to_value"]["alpha"]))
                elif prop == "scale":
                    scale_values.append(
                        (entry["from_value"]["scaleX"], entry["to_value"]["scaleX"])
                    )

    occurrence_rate = {
        kind: {prop: occurs_by_kind[kind][prop] / total for prop in PROPERTIES}
        for kind, total in counts_by_kind.items()
    }

    def median_duration(prop: str) -> float:
        vals = durations[prop]
        return statistics.median(vals) if vals else 0.0

    return {
        "counts_by_kind": dict(counts_by_kind),
        "occurrence_rate": occurrence_rate,
        "median_duration": {prop: median_duration(prop) for prop in PROPERTIES},
        "alpha_from_to": (
            [statistics.median(v[0] for v in alpha_values), statistics.median(v[1] for v in alpha_values)]
            if alpha_values
            else [0.0, 1.0]
        ),
        "scale_from_to": (
            [statistics.median(v[0] for v in scale_values), statistics.median(v[1] for v in scale_values)]
            if scale_values
            else [0.6, 1.0]
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    stats = compute_stats()
    args.out.write_text(json.dumps(stats, indent=2))
    print(f"Wrote {args.out}")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
