#!/usr/bin/env python3
"""Task 4.1 (flows/sdd-comics-ai-positioning/03-plan.md): evaluate the baseline positioner against
a file-wise held-out split of the real training pairs (Task 2.2's output).

Scope note, consistent with Specifications' explicit fallback (no Phase 7 cross-page anchor built
yet): this evaluates **relative Y positioning within each page-cluster**, not absolute canvas
position. A training pair's `target_bbox` is in absolute canvas coordinates, but the baseline (and,
this iteration, everything upstream of it) only predicts a page-cluster's *internal* layout, starting
its own cursor at 0 -- comparing that directly to raw absolute canvas Y would conflate this flow's
actual, in-scope error with the separate, out-of-scope cross-page-anchor problem. So both sides are
rebased to their own cluster's minimum Y before comparing -- an apples-to-apples comparison of what
this phase can actually claim to predict.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

import positioning_bridge as pb
from baseline_position import load_stats, position_page
from spacing_stats import compute_stats
from positioning_models import RegionFeatures

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
PAIRS_DIR = WORK_DIR / "train_pairs"
DEFAULT_STATS_PATH = WORK_DIR / "spacing_stats.json"
DEFAULT_OUT = WORK_DIR / "eval_report.jsonl"

HELD_OUT_FRACTION = 0.25


def _episode_stem(episode_file: str) -> str:
    return Path(episode_file).stem


def choose_held_out(episode_stems: list[str], fraction: float = HELD_OUT_FRACTION) -> set[str]:
    """Deterministic (no randomness) -- every 4th stem, alphabetically, so the same set is chosen
    on every run without needing to persist a seed.
    """
    ordered = sorted(episode_stems)
    step = max(1, round(1 / fraction))
    return set(ordered[::step])


def _load_pairs(episode_stem: str) -> list[dict]:
    path = PAIRS_DIR / f"{episode_stem}.jsonl"
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _group_by_page(pairs: list[dict]) -> dict[tuple[str, int], list[dict]]:
    groups: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for pair in pairs:
        key = (pair["photo_file"], pair["region"]["page_index"])
        groups[key].append(pair)
    return groups


def _spearman(a: list[float], b: list[float]) -> float | None:
    """Rank correlation, no scipy dependency. None if undefined (fewer than 2 points or a
    constant series)."""
    n = len(a)
    if n < 2:
        return None
    rank_a = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: a[i]))}
    rank_b = {i: r for r, i in enumerate(sorted(range(n), key=lambda i: b[i]))}
    d2 = sum((rank_a[i] - rank_b[i]) ** 2 for i in range(n))
    denom = n * (n**2 - 1)
    return 1 - 6 * d2 / denom if denom else None


def evaluate_episode(episode_stem: str, stats: dict, model=None) -> dict:
    """`model`, if given (Task 5.1's trained residual model), adds a directly-comparable
    `learned_model` error series alongside `baseline` -- same held-out pairs, same relative-Y
    scoring, so the two are an apples-to-apples comparison per Requirements' explicit
    baseline-vs-model criterion (Must Have 3): the model is only worth keeping if this comparison
    shows it winning, not assumed.
    """
    pairs = _load_pairs(episode_stem)
    if not pairs:
        return {"episode_stem": episode_stem, "pair_count": 0, "errors": []}

    canvas = pb.load_canvas_reference(pairs[0]["episode_file"])
    baseline_errors_px: list[float] = []
    model_errors_px: list[float] = []
    rank_correlations: list[float] = []

    for (_photo, _page), page_pairs in _group_by_page(pairs).items():
        if len(page_pairs) >= 4:
            order = [p["region"]["reading_order_index"] for p in page_pairs]
            target_ys = [p["target_bbox"][1] for p in page_pairs]
            corr = _spearman(order, target_ys)
            if corr is not None:
                rank_correlations.append(corr)
        regions = [
            RegionFeatures(
                kind=p["region"]["kind"],
                kind_source=p["region"]["kind_source"],
                local_bbox=tuple(p["region"]["local_bbox"]),
                page_index=p["region"]["page_index"],
                reading_order_index=p["region"]["reading_order_index"],
            )
            for p in page_pairs
        ]
        region_ids = [str(i) for i in range(len(page_pairs))]
        proposals = position_page(regions, stats, region_ids=region_ids)
        proposal_by_id = {p.region_id: p for p in proposals}

        if model is not None:
            from infer_positioner import position_page_with_model

            text_context_by_id = {
                str(i): p.get("text_context") for i, p in enumerate(page_pairs)
            }
            model_proposals = position_page_with_model(
                regions,
                stats,
                model,
                region_ids=region_ids,
                text_context_by_id=text_context_by_id,
                match_confidence=page_pairs[0]["match_confidence"] if page_pairs else 0.0,
            )
            model_proposal_by_id = {p.region_id: p for p in model_proposals}

        cluster_min_y = min(p["target_bbox"][1] for p in page_pairs)
        for i, pair in enumerate(page_pairs):
            relative_target_y = pair["target_bbox"][1] - cluster_min_y
            proposed_y = proposal_by_id[str(i)].proposed_y
            baseline_errors_px.append(abs(relative_target_y - proposed_y))
            if model is not None:
                model_y = model_proposal_by_id[str(i)].proposed_y
                model_errors_px.append(abs(relative_target_y - model_y))

    result = {
        "episode_stem": episode_stem,
        "pair_count": len(pairs),
        "canvas_height": canvas["height"],
        "mean_error_px": statistics.mean(baseline_errors_px) if baseline_errors_px else None,
        "median_error_px": statistics.median(baseline_errors_px) if baseline_errors_px else None,
        "mean_error_normalized": (
            statistics.mean(e / canvas["height"] for e in baseline_errors_px)
            if baseline_errors_px
            else None
        ),
        # Diagnostic, not a claim about the baseline's own output: measures whether
        # reading_order_index (the baseline's only ordering signal) correlates with real target Y
        # at all, independent of the baseline's (separately, honestly weak) magnitude calibration.
        # See flows/sdd-comics-ai-positioning/04-implementation-log.md Checkpoint B.
        "reading_order_rank_correlation_mean": (
            statistics.mean(rank_correlations) if rank_correlations else None
        ),
    }
    if model is not None:
        result["learned_model_mean_error_px"] = (
            statistics.mean(model_errors_px) if model_errors_px else None
        )
        result["learned_model_median_error_px"] = (
            statistics.median(model_errors_px) if model_errors_px else None
        )
    return result


def run(out_path: Path = DEFAULT_OUT, model_path: Path | None = None) -> list[dict]:
    episode_stems = [p.stem for p in sorted(PAIRS_DIR.glob("*.jsonl"))]
    held_out = choose_held_out(episode_stems)
    train_stems = [s for s in episode_stems if s not in held_out]

    stats = compute_stats(exclude_episode_stems=held_out)

    model = None
    if model_path is not None:
        from infer_positioner import load_model

        model = load_model(model_path)

    results = [evaluate_episode(stem, stats, model=model) for stem in sorted(held_out)]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--model", type=Path, default=None, help="Also evaluate a trained positioner_model.joblib"
    )
    args = parser.parse_args()
    results = run(out_path=args.out, model_path=args.model)
    print(f"Held-out episodes: {len(results)}")
    for r in results:
        if r["mean_error_px"] is None:
            print(f"  {r['episode_stem']}: no pairs")
            continue
        line = (
            f"  {r['episode_stem']}: {r['pair_count']} pairs, "
            f"baseline_mean={r['mean_error_px']:.0f}px "
            f"({r['mean_error_normalized']*100:.1f}% of canvas height), "
            f"baseline_median={r['median_error_px']:.0f}px"
        )
        if "learned_model_mean_error_px" in r and r["learned_model_mean_error_px"] is not None:
            line += (
                f" | model_mean={r['learned_model_mean_error_px']:.0f}px "
                f"model_median={r['learned_model_median_error_px']:.0f}px"
            )
        print(line)


if __name__ == "__main__":
    main()
