"""Real, broad-coverage text context: the comic's OWN OCR'd balloon/caption dialogue for a matched
page-cluster (apps/comics-ai/comics-ai-baloons's work/ocr.jsonl -- Tesseract OCR, already proven
throughout this whole project, not a weak local VLM's guess and not a fuzzy match against an
external, differently-worded corpus).

This supersedes the original plan of aligning against spiritual_text as the *primary* text-context
source: spiritual_text alignment (text_context.py) only ever produced 2-3 hand-verified episode
matches (paraphrase-vs-19th-century-prose gap too wide for automated matching, confirmed in
flows/sdd-comics-ai-positioning/04-implementation-log.md Task 6.1). The comic's own dialogue has no
such gap -- it *is* the scene, at 100% relevance by construction, and covers all 27 dataset files
(verified: all 16 episodes with real training pairs have OCR text available). Real finding while
building this: reading a training pair's own cluster dialogue directly (episode
9b76ee4c0f844a86a5a3475831482d7e.comics, "10_the_brahmanas_do_not_have_to_fight") surfaced a
caption -- "THE MIGHTY CITY OF MAHISMATI, THE CAPITAL OF HAYHAYS; THE INVINCIBLE FORTRESS OF THE
KING ARJUNA KARTAVIRYA" -- that visually confirmed this episode belongs to the same Kartavirya/
Parashurama arc as 06/08/09, and OCR grep separately surfaced a 4th episode
(6c690c679511407cb558a0dc347fdebf.comics, "11_sneaky_revenge": "ARJUNA KARTAVIRYA, WAS KILLED
BECAUSE OF HIM!") -- neither found by the earlier spiritual_text-alignment attempts.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import positioning_bridge as pb

OCR_JSONL = pb.REPO_ROOT / "apps" / "comics-ai" / "comics-ai-baloons" / "work" / "ocr.jsonl"

_CACHE: dict[tuple[str, int], str] | None = None


def _load() -> dict[tuple[str, int], str]:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    lookup: dict[tuple[str, int], str] = {}
    with OCR_JSONL.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("lang_index") != 0:  # English only, same convention as spike_text_alignment
                continue
            key = (row["source_file"], row["layer_index"])
            lookup[key] = row["text"]
    _CACHE = lookup
    return lookup


def text_for_layer(episode_file: str, layer_index: int) -> str | None:
    return _load().get((episode_file, layer_index))


def text_for_cluster(episode_file: str, layer_indexes: list[int]) -> str | None:
    """Concatenated OCR'd dialogue/caption text for every layer in a page's ground_truth_cluster
    that actually has any (most cluster layers are background/character art with no text at all).
    Returns None (not "") when nothing in the cluster has text -- so callers can distinguish
    "checked, nothing there" from "never checked".
    """
    lookup = _load()
    texts = [lookup[(episode_file, li)] for li in sorted(layer_indexes) if (episode_file, li) in lookup]
    return " / ".join(texts) if texts else None
