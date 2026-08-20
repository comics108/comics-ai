#!/usr/bin/env python3
"""Task 2.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): storyboard
dataclasses and the deterministic (no-AI) storyboard builder -- the Must-Have path's actual
storyboard source, per Requirements' "deterministic path is the release gate" decision.

Real design decision, not fully pinned down by 02-specifications.md (which only says the
deterministic mode uses "ordered scene groups based only on contiguous sloka ranges," without
specifying a grouping granularity): **one scene per chapter**, covering that chapter's entire
contiguous sloka-order range. Any sub-chapter scene boundary is inherently a narrative/semantic
judgment call -- exactly what "no AI, no invented structure" means to avoid making. This keeps the
deterministic mode honestly minimal rather than picking an arbitrary batch size (e.g. "5 slokas per
scene") that would imply structure the source data doesn't actually signal. Flagged here for Anton
to redirect if a different default is wanted; nothing downstream depends on this granularity for
the Must-Have render (render_cards.py renders one card per real SlokaSource regardless of how
scenes group them).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from models import CanonicalChapter

SCHEMA_VERSION = 1


@dataclass(frozen=True)
class StoryScene:
    scene_id: str
    title: str
    summary_ru: str | None
    source_sloka_orders: tuple[int, ...]
    characters: tuple[str, ...]
    location: str | None
    visual_prompt: str | None


@dataclass(frozen=True)
class ChapterStoryboard:
    schema_version: int
    mode: Literal["ollama", "deterministic"]
    model: str | None
    prompt_version: str
    chapter_summary_ru: str | None
    scenes: tuple[StoryScene, ...]
    warnings: tuple[str, ...]
    raw_model_output: str | None


def build_deterministic_storyboard(chapter: CanonicalChapter) -> ChapterStoryboard:
    """The Must-Have storyboard: one scene per chapter, covering its full real sloka-order range,
    with no synthetic summary and no invented characters/location. Real chapters always have at
    least one sloka (dataset-integrity-checked upstream in load_dataset.py) except in synthetic
    test fixtures, which this function also handles honestly (zero scenes, not a fabricated one)."""
    if not chapter.slokas:
        return ChapterStoryboard(
            schema_version=SCHEMA_VERSION,
            mode="deterministic",
            model=None,
            prompt_version="n/a",
            chapter_summary_ru=None,
            scenes=(),
            warnings=("chapter has zero slokas; no scene emitted",),
            raw_model_output=None,
        )

    orders = tuple(s.order for s in chapter.slokas)
    scene = StoryScene(
        scene_id=f"ch{chapter.order:02d}-scene01",
        title=chapter.title,
        summary_ru=None,
        source_sloka_orders=orders,
        characters=(),
        location=None,
        visual_prompt=None,
    )
    return ChapterStoryboard(
        schema_version=SCHEMA_VERSION,
        mode="deterministic",
        model=None,
        prompt_version="n/a",
        chapter_summary_ru=None,
        scenes=(scene,),
        warnings=(),
        raw_model_output=None,
    )
