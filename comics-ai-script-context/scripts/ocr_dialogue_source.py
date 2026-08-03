#!/usr/bin/env python3
"""Criterion 2 (flows/sdd-comics-ai-transformations/01-requirements.md v0.3): OCR-dialogue
fallback text source for `sdd-comics-ai-script-context`, extending coverage beyond the 6 episodes
with hand-verified `spiritual_text` excerpts.

Real, checked fact (2026-08-02): `comics-ai-baloons`'s `discover.py` scans the whole dataset
structurally (independent of any photo-matching), so `work/ocr.jsonl` already has real OCR'd
dialogue for **all 27** episode files, not just the training-relevant subset. This module builds a
per-episode excerpt by concatenating that episode's own real English balloon dialogue -- broad
coverage, but narrative-shallower than `spiritual_text` (balloon lines, not descriptive prose) --
kept as an explicitly lower-trust `text_source` provenance (see `scene_models.SceneExtraction`),
never merged into `spiritual_text`'s excerpts silently.
"""

from __future__ import annotations

import json
from pathlib import Path

BALOONS_OCR_JSONL = (
    Path(__file__).resolve().parents[3] / "comics-ai" / "comics-ai-baloons" / "work" / "ocr.jsonl"
)

ENGLISH_LANG_INDEX = 0
MAX_EXCERPT_CHARS = 4000  # generous for a whole episode's dialogue; keeps the LLM prompt bounded


def load_ocr_entries(path: Path = BALOONS_OCR_JSONL) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def build_excerpt_for_episode(
    episode_file: str, entries: list[dict], max_chars: int = MAX_EXCERPT_CHARS
) -> str | None:
    """Concatenates an episode's own real English balloon dialogue, in layer-index order (a
    reasonable proxy for reading order within the episode, same assumption `scene_text.py` already
    makes elsewhere in this pipeline). Returns None if the episode has no real English text at
    all (honest absence, not an empty-string guess)."""
    lines = [
        (e["layer_index"], e["text"].strip())
        for e in entries
        if e["source_file"] == episode_file
        and e.get("lang_index") == ENGLISH_LANG_INDEX
        and e.get("text", "").strip()
    ]
    if not lines:
        return None
    lines.sort(key=lambda pair: pair[0])
    excerpt = " ".join(text for _, text in lines)
    return excerpt[:max_chars]
