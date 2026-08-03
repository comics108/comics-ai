#!/usr/bin/env python3
"""Task 1.2 (flows/sdd-comics-ai-positioning/03-plan.md): count real, usable training pairs
available from comics-multimodal's existing alignment output, before building anything that
depends on that count. This number bounds Phase 5 (learned model) -- see Plan's Risk Assessment.
"""

from __future__ import annotations

import argparse
from collections import Counter

import positioning_bridge as pb


def summarize() -> dict:
    matched = list(pb.iter_matched_alignment_rows())
    episodes = Counter(row["episode_file"] for row in matched)
    cluster_sizes = [len(row["ground_truth_cluster"]) for row in matched]
    return {
        "matched_pairs": len(matched),
        "distinct_episodes": len(episodes),
        "episodes": dict(episodes),
        "min_cluster_size": min(cluster_sizes) if cluster_sizes else 0,
        "max_cluster_size": max(cluster_sizes) if cluster_sizes else 0,
        "mean_cluster_size": sum(cluster_sizes) / len(cluster_sizes) if cluster_sizes else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = summarize()
    print(f"Matched photo/page pairs with a non-empty ground_truth_cluster: {summary['matched_pairs']}")
    print(f"Distinct episodes represented: {summary['distinct_episodes']} / 27")
    print(
        f"ground_truth_cluster size: min={summary['min_cluster_size']} "
        f"max={summary['max_cluster_size']} mean={summary['mean_cluster_size']:.1f}"
    )
    print("\nPer-episode matched-pair counts:")
    for episode, count in sorted(summary["episodes"].items(), key=lambda kv: -kv[1]):
        print(f"  {episode}: {count}")


if __name__ == "__main__":
    main()
