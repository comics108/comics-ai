#!/usr/bin/env python3
"""Task 1.1 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): canonical data
model, exactly as specified in 02-specifications.md's "Canonical Data Model" section. Pure data,
no I/O -- load_dataset.py (Task 1.2) is the only producer of these.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SlokaSource:
    id: int
    chapter_id: int
    order: int
    name: str
    sanskrit: str
    transcription: str
    translation_ru: str
    comment_ru: str
    audio_ref: str
    sanskrit_audio_ref: str


@dataclass(frozen=True)
class CanonicalChapter:
    book_id: int
    chapter_id: int
    order: int
    title: str
    slokas: tuple[SlokaSource, ...]
