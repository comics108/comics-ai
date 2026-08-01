"""Shared dataclasses used across pipeline stages (flows/sdd-comics-ai-baloons/02-specifications.md
"Data Models"). Kept dependency-free (stdlib only) so every stage can import it cheaply.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Literal


@dataclass
class ImageSlot:
    lang_index: int  # position in the layer's images[] array -- 0=en, 1=ru, 2=hi, ...
    file_template: str  # e.g. "b1_eng_{0}_{1}_{2}.png"
    width: int
    height: int


@dataclass
class BalloonLayer:
    source_file: str  # dataset/*.comics filename (basename only)
    layer_index: int  # index into data.json "layers"
    slots: dict[int, ImageSlot]  # populated indices only
    canvas_offset: tuple[int, int] = (0, 0)  # from the layer's base TranslateAnim, if any

    def to_jsonable(self) -> dict:
        d = asdict(self)
        d["slots"] = {str(k): v for k, v in d["slots"].items()}
        return d

    @staticmethod
    def from_jsonable(d: dict) -> "BalloonLayer":
        slots = {int(k): ImageSlot(**v) for k, v in d["slots"].items()}
        return BalloonLayer(
            source_file=d["source_file"],
            layer_index=d["layer_index"],
            slots=slots,
            canvas_offset=tuple(d.get("canvas_offset", (0, 0))),
        )


@dataclass
class OcrResult:
    source_file: str
    layer_index: int
    lang_index: int  # which slot was OCR'd
    text: str
    confidence: float
    needed_crop_fallback: bool = False  # direct OCR was empty; recovered via outline-stripped crop
    # ^ kept as a feature: balloons that need this fallback skew towards short/tight balloons and
    # possibly hand lettering (irregular strokes throw Tesseract off more) -- worth reusing as a
    # signal in stage 5's lettering classifier, not just an OCR implementation detail.


@dataclass
class MatchResult:
    source_file: str
    layer_index: int
    csv_row_id: str | None
    match_score: float
    matched_on: Literal["en", "ru", ""]
    status: Literal[
        "matched", "skipped_no_match", "skipped_ambiguous", "skipped_low_confidence"
    ]
    reason: str = ""


@dataclass
class LetteringClass:
    source_file: str
    layer_index: int
    label: Literal["machine_set", "hand_lettered"]
    confidence: float
    signals: dict[str, float] = field(default_factory=dict)


@dataclass
class RenderResult:
    source_file: str
    layer_index: int
    lang_code: str
    rendered: bool
    image_path: str | None = None
    reason: str = ""
