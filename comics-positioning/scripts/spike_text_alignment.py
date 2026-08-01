#!/usr/bin/env python3
"""Task 6.1 (flows/sdd-comics-ai-positioning/03-plan.md): attempt automatic episode <-> spiritual_
text passage alignment, using the same content-based, skip+log-not-guess technique that already
found the real episode-21 ("ambas_plea") match by hand during Requirements drafting -- English
balloon dialogue (comics-ai-baloons' own OCR corpus, work/ocr.jsonl) fuzzy-matched against
spiritual_text, chunked into its own SECTION-delimited passages.

Standalone, time-boxed per Plan: reports real coverage across all 27 episodes. Whether/how its
output feeds into the positioning model itself is decided *after* seeing real coverage, not before
-- see build_pairs.py's optional --text-features flag (added once this spike's real results were
known) for how a match, where found, becomes an actual model input.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from rapidfuzz import fuzz

import positioning_bridge as pb

BALOONS_APP_DIR = pb.REPO_ROOT / "apps" / "comics-ai" / "comics-ai-baloons"
BALOONS_SCRIPTS_DIR = BALOONS_APP_DIR / "scripts"
OCR_JSONL = BALOONS_APP_DIR / "work" / "ocr.jsonl"
SPIRITUAL_TEXT_PATH = (
    pb.DATASET_DIR
    / "boranko"
    / "mahabharata"
    / "book1"
    / "spiritual_text"
    / "The Mahabharata, Volume I., Book 1-3 by Kisari Mohan Ganguli.html"
)

if str(BALOONS_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(BALOONS_SCRIPTS_DIR))
from match import normalize  # noqa: E402  (comics-ai-baloons' own text normalization, reused)

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_OUT = WORK_DIR / "text_alignment.jsonl"

MIN_PHRASE_LENGTH = 12  # normalized chars; same cutoff align_photo.py uses, same reasoning
PARTIAL_MATCH_THRESHOLD = 80.0
MIN_CONFIDENT_PHRASES = 2

SECTION_RE = re.compile(r"(SECTION [A-Z]+(?:\s*\([^)]*\))?)")


@dataclass
class TextSection:
    index: int
    heading: str
    text: str
    normalized: str


@dataclass
class EpisodeTextAlignment:
    episode_file: str
    status: str  # "matched" | "skipped_no_match"
    best_section_index: int | None
    best_section_heading: str | None
    confident_phrase_count: int
    confidence: float
    reason: str


def load_spiritual_text_sections(path: Path = SPIRITUAL_TEXT_PATH) -> list[TextSection]:
    html = path.read_text(encoding="utf-8", errors="ignore")
    text = re.sub(r"<[^>]+>", " ", html)
    text = re.sub(r"\s+", " ", text)

    parts = SECTION_RE.split(text)
    sections: list[TextSection] = []
    # parts[0] is preamble text before the first SECTION heading
    if parts[0].strip():
        norm = normalize(parts[0])
        sections.append(TextSection(0, "(preamble)", parts[0], norm))
    idx = 1
    i = 1
    while i < len(parts) - 1:
        heading, body = parts[i], parts[i + 1]
        sections.append(TextSection(idx, heading, body, normalize(body)))
        idx += 1
        i += 2
    return sections


def load_episode_phrases() -> dict[str, list[str]]:
    """English (lang_index == 0) balloon phrases per episode, deduplicated, length-filtered."""
    phrases_by_episode: dict[str, set[str]] = defaultdict(set)
    with OCR_JSONL.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if row.get("lang_index") != 0:
                continue
            norm = normalize(row["text"])
            if len(norm) >= MIN_PHRASE_LENGTH:
                phrases_by_episode[row["source_file"]].add(norm)
    return {k: sorted(v) for k, v in phrases_by_episode.items()}


def align_episode(phrases: list[str], sections: list[TextSection]) -> EpisodeTextAlignment:
    hits_by_section: dict[int, list[float]] = defaultdict(list)
    for phrase in phrases:
        best_section_idx = None
        best_score = 0.0
        for section in sections:
            score = fuzz.partial_ratio(phrase, section.normalized)
            if score > best_score:
                best_score = score
                best_section_idx = section.index
        if best_score >= PARTIAL_MATCH_THRESHOLD and best_section_idx is not None:
            hits_by_section[best_section_idx].append(best_score)

    if not hits_by_section:
        return EpisodeTextAlignment("", "skipped_no_match", None, None, 0, 0.0, "no confident phrase matched any section")

    best_section = max(hits_by_section, key=lambda s: len(hits_by_section[s]))
    scores = hits_by_section[best_section]
    if len(scores) < MIN_CONFIDENT_PHRASES:
        return EpisodeTextAlignment(
            "", "skipped_no_match", None, None, len(scores), 0.0,
            f"only {len(scores)} confident phrase hit(s) (need >= {MIN_CONFIDENT_PHRASES})",
        )

    return EpisodeTextAlignment(
        "", "matched", best_section, None, len(scores), sum(scores) / len(scores) / 100.0, ""
    )


def run(out_path: Path = DEFAULT_OUT) -> list[EpisodeTextAlignment]:
    sections = load_spiritual_text_sections()
    section_heading_by_index = {s.index: s.heading for s in sections}
    phrases_by_episode = load_episode_phrases()

    results: list[EpisodeTextAlignment] = []
    for episode_file, phrases in sorted(phrases_by_episode.items()):
        result = align_episode(phrases, sections)
        result.episode_file = episode_file
        if result.best_section_index is not None:
            result.best_section_heading = section_heading_by_index.get(result.best_section_index)
        results.append(result)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(asdict(r)) + "\n")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    results = run(out_path=args.out)
    matched = [r for r in results if r.status == "matched"]
    print(f"{len(matched)} / {len(results)} episodes matched a spiritual_text passage")
    for r in results:
        status = "MATCHED" if r.status == "matched" else "skip"
        heading = f" -> {r.best_section_heading}" if r.best_section_heading else ""
        print(f"  [{status}] {r.episode_file}: {r.confident_phrase_count} phrases{heading} ({r.reason})")


if __name__ == "__main__":
    main()
