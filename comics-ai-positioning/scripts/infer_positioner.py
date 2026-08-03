#!/usr/bin/env python3
"""Task 5.1 (flows/sdd-comics-ai-positioning/03-plan.md): apply the trained residual model on top
of baseline_position.py's proposals -- final_y = baseline_y + predicted_residual. A model that
learned nothing useful (residual ~= 0 everywhere) degrades to exactly the baseline, by construction.
"""

from __future__ import annotations

from pathlib import Path

import joblib

from baseline_position import position_page
from positioner_features import build_features
from positioning_models import PositionProposal, RegionFeatures

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[1] / "work" / "positioner_model.joblib"


def load_model(path: Path = DEFAULT_MODEL_PATH):
    if not path.is_file():
        raise FileNotFoundError(f"No trained model at {path} -- run train_positioner.py first")
    return joblib.load(path)


def position_page_with_model(
    regions: list[RegionFeatures],
    stats: dict,
    model,
    region_ids: list[str] | None = None,
    text_context_by_id: dict[str, str | None] | None = None,
    match_confidence: float = 0.0,
) -> list[PositionProposal]:
    baseline_proposals = position_page(regions, stats, region_ids=region_ids)
    page_region_count = len(regions)
    text_context_by_id = text_context_by_id or {}

    results: list[PositionProposal] = []
    for region, baseline in zip(
        sorted(regions, key=lambda r: r.reading_order_index), baseline_proposals
    ):
        features = build_features(
            kind=region.kind,
            local_bbox=region.local_bbox,
            reading_order_index=region.reading_order_index,
            page_region_count=page_region_count,
            match_confidence=match_confidence,
            text_context_length=len(text_context_by_id.get(baseline.region_id) or ""),
        )
        residual = float(model.predict([features])[0])
        results.append(
            PositionProposal(
                region_id=baseline.region_id,
                proposed_x=baseline.proposed_x,
                proposed_y=round(baseline.proposed_y + residual),
                source="learned_model",
                confidence=None,
            )
        )
    return results
