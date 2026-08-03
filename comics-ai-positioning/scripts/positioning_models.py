"""Shared dataclasses (flows/sdd-comics-ai-positioning/02-specifications.md "Data Models")."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class RegionFeatures:
    kind: str
    kind_source: str  # "explicit" | "inferred_heuristic" | "predicted"
    local_bbox: tuple[int, int, int, int]  # region's bbox within its own page/photo crop
    page_index: int
    reading_order_index: int  # this region's index among the page's regions, top-to-bottom


@dataclass
class PositionTrainingPair:
    episode_file: str
    photo_file: str
    region: RegionFeatures
    target_layer_index: int
    target_bbox: tuple[int, int, int, int]  # GroundTruthRegion.bbox -- canvas coordinates
    target_transform: dict = field(default_factory=dict)  # x, y always; scale_x/scale_y/alpha optional (Should Have)
    match_confidence: float = 0.0  # from PageAlignmentResult
    text_context: str | None = None  # this page-cluster's own real OCR'd dialogue/caption text
    # (scripts/scene_text.py) -- 100% relevant by construction, broad coverage (all 16 episodes with
    # real training pairs). None only when literally no layer in the cluster has any OCR'd text.
    source_narrative_context: str | None = None  # human-verified spiritual_text excerpt
    # (scripts/text_context.py) -- narrow (a handful of episodes), a bonus signal, never a guess

    def to_jsonable(self) -> dict:
        return asdict(self)


@dataclass
class PositionProposal:
    region_id: str  # matches the corresponding DetectedRegion (comics-multimodal contract)
    proposed_x: int
    proposed_y: int
    proposed_scale_x: float = 1.0
    proposed_scale_y: float = 1.0
    source: str = "baseline"  # "baseline" | "learned_model"
    confidence: float | None = None

    def to_jsonable(self) -> dict:
        return asdict(self)
