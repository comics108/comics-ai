#!/usr/bin/env python3
"""Task 5.1 (flows/sdd-comics-ai-positioning/03-plan.md): train a small regression model that
predicts the *residual* (baseline_position.py's proposed_y minus this) -- not absolute position
from scratch. Residual learning was Specifications' stated design (easier than predicting absolute
position given ~27 files of signal) and lets a poorly-fitting model degrade gracefully to exactly
the baseline (residual ~= 0) rather than to something arbitrary.

Gated per Plan: only meaningful to run once Task 1.2's real data count and Task 4.2's baseline
sanity check are known. Real numbers as of this task: 392 total training pairs, 314 in the training
split (78 held out across 4 episodes) -- small, but this is a low-dimensional engineered-feature
regression (positioner_features.py), not a from-scratch deep model, so it's a reasonable fit for
this data size. Same honest-risk framing as comics-ai-baloons/comics-ai-multimodal: reported against
the baseline, not assumed to win.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestRegressor

import positioning_bridge as pb
from baseline_position import load_stats, position_page
from evaluate_positioning import _group_by_page, _load_pairs, choose_held_out
from positioner_features import build_features
from positioning_models import RegionFeatures

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
PAIRS_DIR = WORK_DIR / "train_pairs"
DEFAULT_MODEL_PATH = WORK_DIR / "positioner_model.joblib"


def build_training_matrix(train_stems: list[str], stats: dict) -> tuple[list[list[float]], list[float]]:
    X: list[list[float]] = []
    y: list[float] = []

    for stem in train_stems:
        pairs = _load_pairs(stem)
        for (_photo, _page), page_pairs in _group_by_page(pairs).items():
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
            baseline_proposals = {p.region_id: p for p in position_page(regions, stats, region_ids=region_ids)}

            cluster_min_y = min(p["target_bbox"][1] for p in page_pairs)
            page_region_count = len(page_pairs)

            for i, pair in enumerate(page_pairs):
                relative_target_y = pair["target_bbox"][1] - cluster_min_y
                baseline_y = baseline_proposals[str(i)].proposed_y
                residual = relative_target_y - baseline_y

                features = build_features(
                    kind=pair["region"]["kind"],
                    local_bbox=tuple(pair["region"]["local_bbox"]),
                    reading_order_index=pair["region"]["reading_order_index"],
                    page_region_count=page_region_count,
                    match_confidence=pair["match_confidence"],
                    text_context_length=len(pair.get("text_context") or ""),
                )
                X.append(features)
                y.append(float(residual))

    return X, y


def train(model_path: Path = DEFAULT_MODEL_PATH) -> dict:
    episode_stems = [p.stem for p in sorted(PAIRS_DIR.glob("*.jsonl"))]
    held_out = choose_held_out(episode_stems)
    train_stems = [s for s in episode_stems if s not in held_out]

    from spacing_stats import compute_stats

    stats = compute_stats(exclude_episode_stems=held_out)
    X, y = build_training_matrix(train_stems, stats)

    if len(X) < 10:
        raise RuntimeError(f"Too few training examples ({len(X)}) to fit anything meaningful")

    model = RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=3, random_state=0
    )
    model.fit(X, y)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)

    return {"train_examples": len(X), "train_episodes": len(train_stems), "held_out_episodes": len(held_out)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_MODEL_PATH)
    args = parser.parse_args()
    summary = train(model_path=args.out)
    print(json.dumps(summary, indent=2))
    print(f"Model saved to {args.out}")


if __name__ == "__main__":
    main()
