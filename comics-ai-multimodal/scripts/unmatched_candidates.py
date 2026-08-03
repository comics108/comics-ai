#!/usr/bin/env python3
"""Criterion 4 pilot tool (flows/sdd-comics-ai-transformations/02-specifications.md, "Criterion 4
Pilot" section). For each unmatched page-row in `work/alignment.jsonl`, surfaces:

  (a) an adjacency candidate -- an episode confidently matched immediately before AND after this
      page in physical page order (photo filenames are capture timestamps, so filename order is
      real physical page order) -- and only when both sides agree on the same episode;
  (b) weak text-signal candidates (score >= WEAK_MATCH_THRESHOLD, below the trusted
      `align_photo.PARTIAL_MATCH_THRESHOLD`) -- real but not confident matches;
  (c) the list of known episodes with zero matched pages at all, for context.

Real investigation (2026-08-02) found neither (a) nor (b) is independently confirmed strongly
enough to auto-apply the way criterion 3's margin rule was -- cross-checking 3 adjacency proposals
against the weak text signal found zero corroboration. This tool is explicitly for HUMAN REVIEW,
mirroring `sdd-comics-ai-multimodal`'s own established never-silent-auto-apply pattern -- it does
not write to `alignment.jsonl` or propose anything as a confident match.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import align_photo as ap
from rapidfuzz import fuzz

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_ALIGNMENT = WORK_DIR / "alignment.jsonl"
DEFAULT_EPISODES_CSV = (
    ap.baloons_bridge.REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_interactive"
    / "Comics_Episodes.csv"
)
DEFAULT_OUT = WORK_DIR / "unmatched_candidates.jsonl"

WEAK_MATCH_THRESHOLD = 60.0
WEAK_CANDIDATE_LIMIT = 3


def load_alignment_rows(path: Path = DEFAULT_ALIGNMENT) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def compute_adjacency_candidates(rows: list[dict]) -> dict[tuple[str, int], str | None]:
    """For each unmatched (photo_file, page_index), returns the episode confidently matched
    immediately before AND after it in physical-page order, or None if either side is unmatched
    (sequence edge) or the two sides disagree (a genuine story-transition boundary)."""
    ordered = sorted(rows, key=lambda r: (r["photo_file"], r["page_index"]))
    seq = [
        (r["photo_file"], r["page_index"], r["episode_file"] if r["status"] == "matched" else None)
        for r in ordered
    ]
    n = len(seq)
    result: dict[tuple[str, int], str | None] = {}
    for i, (photo, page, ep) in enumerate(seq):
        if ep is not None:
            continue
        before = next((seq[j][2] for j in range(i - 1, -1, -1) if seq[j][2] is not None), None)
        after = next((seq[j][2] for j in range(i + 1, n) if seq[j][2] is not None), None)
        result[(photo, page)] = before if (before is not None and before == after) else None
    return result


def weak_text_candidates(
    norm_page_text: str, corpus: list[dict], limit: int = WEAK_CANDIDATE_LIMIT
) -> list[tuple[str, float]]:
    """Best score per episode for any corpus phrase scoring >= WEAK_MATCH_THRESHOLD (but not
    necessarily clearing align_photo's trusted PARTIAL_MATCH_THRESHOLD) -- real but not confident
    signal, ranked descending, top `limit` episodes."""
    best_by_episode: dict[str, float] = {}
    for entry in corpus:
        candidate = ap.normalize(entry["text"])
        if len(candidate) < ap.MIN_PHRASE_LENGTH:
            continue
        score = fuzz.partial_ratio(candidate, norm_page_text)
        if score >= WEAK_MATCH_THRESHOLD:
            if entry["source_file"] not in best_by_episode or score > best_by_episode[entry["source_file"]]:
                best_by_episode[entry["source_file"]] = score
    ranked = sorted(best_by_episode.items(), key=lambda kv: -kv[1])
    return ranked[:limit]


def all_episode_files(path: Path = DEFAULT_EPISODES_CSV) -> set[str]:
    files = set()
    with path.open(encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        for row in reader:
            file_field = row.get("File", "")
            if file_field.endswith(".comics"):
                files.add(file_field.split("/")[-1])
    return files


def zero_coverage_episodes(rows: list[dict], episodes_csv: Path = DEFAULT_EPISODES_CSV) -> list[str]:
    matched = {r["episode_file"] for r in rows if r["status"] == "matched"}
    return sorted(all_episode_files(episodes_csv) - matched)


def build_report(
    alignment_path: Path = DEFAULT_ALIGNMENT,
    episodes_csv: Path = DEFAULT_EPISODES_CSV,
    out_path: Path = DEFAULT_OUT,
) -> dict:
    rows = load_alignment_rows(alignment_path)
    unmatched = [r for r in rows if r["status"] != "matched"]
    adjacency = compute_adjacency_candidates(rows)
    corpus = ap.load_ocr_corpus()
    zero_coverage = zero_coverage_episodes(rows, episodes_csv)

    entries = []
    for r in unmatched:
        photo_path = ap.DEFAULT_LOWCAMERA_DIR / r["photo_file"]
        image = ap.cv2.imread(str(photo_path))
        pages = ap.detect_pages(image) if image is not None else []
        text = ""
        if r["page_index"] < len(pages):
            x0, y0, x1, y1 = pages[r["page_index"]].bbox
            crop = image[y0:y1, x0:x1]
            rect = ap.rectify_page(crop)
            text = ap.ocr_page(rect.rectified)
        norm_text = ap.normalize(text)

        key = (r["photo_file"], r["page_index"])
        entries.append(
            {
                "photo_file": r["photo_file"],
                "page_index": r["page_index"],
                "original_reason": r["reason"],
                "adjacency_candidate": adjacency.get(key),
                "weak_text_candidates": (
                    weak_text_candidates(norm_text, corpus) if norm_text else []
                ),
            }
        )

    report = {"zero_coverage_episodes": zero_coverage, "unmatched_pages": entries}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))
    return report


def main() -> None:
    report = build_report()
    n_adjacency = sum(1 for e in report["unmatched_pages"] if e["adjacency_candidate"])
    n_weak = sum(1 for e in report["unmatched_pages"] if e["weak_text_candidates"])
    print(
        f"{len(report['unmatched_pages'])} unmatched pages -> {DEFAULT_OUT}\n"
        f"  {n_adjacency} have an adjacency candidate\n"
        f"  {n_weak} have >=1 weak text candidate\n"
        f"  {len(report['zero_coverage_episodes'])} episodes have zero matched pages at all"
    )


if __name__ == "__main__":
    main()
