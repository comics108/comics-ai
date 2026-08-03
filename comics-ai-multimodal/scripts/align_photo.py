#!/usr/bin/env python3
"""Task 5.2 (revised, Revision 1.2 -- page-level, not panel-level; see detect_panels.py): match
each detected PAGE in a real photo to its source episode + layer content, via OCR + fuzzy substring
matching against comics-ai-baloons' own per-balloon OCR corpus (work/ocr.jsonl) -- the same
content-based, skip+log-not-guess principle already used for CSV matching in that flow, now applied
here against a different corpus and at page (not CSV-row) granularity.

Cross-flow dependency: requires apps/comics-ai/comics-ai-baloons/work/ocr.jsonl to exist (that
pipeline's OCR stage must have been run at least once). Confirmed present and current at the start
of this task: 1650 entries covering all 27 dataset files.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract

import baloons_bridge
from augment import cluster_layers_by_scene
from detect_panels import detect_pages
from rectify import rectify_page
from render_canvas import GroundTruthRegion

if str(baloons_bridge.BALOONS_SCRIPTS_DIR) not in sys.path:
    sys.path.append(str(baloons_bridge.BALOONS_SCRIPTS_DIR))
from match import normalize  # noqa: E402  (comics-ai-baloons' own text normalization, reused)
from rapidfuzz import fuzz  # noqa: E402

WORK_DIR = Path(__file__).resolve().parents[1] / "work"
DEFAULT_LOWCAMERA_DIR = (
    baloons_bridge.REPO_ROOT
    / "dataset"
    / "boranko"
    / "mahabharata"
    / "book1"
    / "comics_book_lowcamera"
)
DEFAULT_OUT = WORK_DIR / "alignment.jsonl"
DEFAULT_CANVAS_DIR = WORK_DIR / "canvas"
BALOONS_OCR_JSONL = baloons_bridge.BALOONS_APP_DIR / "work" / "ocr.jsonl"

# rapidfuzz partial_ratio (0-100): finds the best-matching substring of the (long) page OCR blob
# for each (short) known balloon phrase -- unlike token_sort_ratio (comics-ai-baloons' CSV
# matching), which compares two comparably-sized strings and is a poor fit when one side is a
# whole page's worth of concatenated dialogue.
PARTIAL_MATCH_THRESHOLD = 80.0
MIN_CONFIDENT_PHRASES = 2  # a page with >=2 confident hits for one episode is trusted outright
MIN_PHRASE_LENGTH = 12  # normalized chars; found via real-data verification (2026-07-31): short
# generic corpus phrases ("NO", "NO.") trivially partial_ratio-match almost any OCR'd page text by
# substring containment -- a real photo matched 5/5 "confident" hits that were all just "NO"/"NO."
# variants of the same short word, not 5 independently-meaningful phrase matches. partial_ratio is
# substring-containment-seeking by design, which is exactly why very short candidates are
# dangerous with it (unlike comics-ai-baloons' token_sort_ratio, which naturally penalizes length
# mismatch). ~9% of the real OCR corpus (149/1650 entries) is shorter than this cutoff.
MARGIN_FOR_SINGLE_HIT = 10.0  # confidence points. A page with exactly 1 confident hit used to be
# rejected outright regardless of context (flows/sdd-comics-ai-positioning's original design).
# Real-data investigation (2026-08-01, flows/sdd-comics-ai-transformations/02-specifications.md)
# re-ran real OCR+matching on all 24 real "1 confident hit" pages in the dataset: 21/24 had no
# competing episode at all and were real, trustworthy matches (e.g. a 98.0-score hit on an
# otherwise-garbled page -- the rest of the page just didn't happen to contain a second corpus
# phrase). Only 3/24 had a second episode also at 1 hit -- genuinely ambiguous. A >=10-point score
# margin over the best competing episode's hit cleanly separated the 22 recoverable cases (21 with
# no competitor + 1 with a 13-point margin) from the 2 truly ambiguous ones (2.9 and 7.5-point
# margins) in that real sample. This does not relax MIN_CONFIDENT_PHRASES itself -- a >=2-hit match
# is still trusted outright, unconditionally -- it only recovers single-hit pages that have no real
# competing claim.


@dataclass
class PageAlignmentResult:
    photo_file: str
    page_index: int
    episode_file: str | None
    matched_layer_indexes: list[int]
    ground_truth_cluster: list[int]
    confidence: float
    status: str  # "matched" | "skipped_no_match"
    reason: str


def load_ocr_corpus(path: Path = BALOONS_OCR_JSONL) -> list[dict]:
    entries = []
    with path.open(encoding="utf-8-sig") as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def ocr_page(image: np.ndarray) -> str:
    """OCR a whole page crop, eng+rus jointly (Tesseract supports multi-language OCR in one pass)
    since we don't know in advance which balloons on a given page are in which language.
    """
    return pytesseract.image_to_string(image, lang="eng+rus")


def match_page_to_episode(
    page_text: str, corpus: list[dict]
) -> tuple[str | None, list[int], float, str]:
    norm_page = normalize(page_text)
    if not norm_page:
        return None, [], 0.0, "no OCR text extracted from page"

    hits_by_episode: dict[str, list[tuple[int, float]]] = defaultdict(list)
    for entry in corpus:
        candidate = normalize(entry["text"])
        if len(candidate) < MIN_PHRASE_LENGTH:
            continue
        score = fuzz.partial_ratio(candidate, norm_page)
        if score >= PARTIAL_MATCH_THRESHOLD:
            hits_by_episode[entry["source_file"]].append((entry["layer_index"], score))

    if not hits_by_episode:
        return None, [], 0.0, "no balloon phrase matched confidently"

    max_hit_count = max(len(hits) for hits in hits_by_episode.values())

    if max_hit_count >= MIN_CONFIDENT_PHRASES:
        best_episode = max(hits_by_episode, key=lambda ep: len(hits_by_episode[ep]))
        hits = hits_by_episode[best_episode]
        layer_indexes = sorted({idx for idx, _ in hits})
        confidence = sum(score for _, score in hits) / len(hits) / 100.0
        return best_episode, layer_indexes, confidence, ""

    # max_hit_count == 1 here for every episode present (an episode only enters hits_by_episode on
    # a real hit, so 0 is impossible) -- see MARGIN_FOR_SINGLE_HIT's docstring for the real-data
    # justification of what follows.
    ranked = sorted(
        ((ep, hits[0][1]) for ep, hits in hits_by_episode.items()),
        key=lambda pair: -pair[1],
    )
    top_episode, top_score = ranked[0]
    runner_up_score = ranked[1][1] if len(ranked) > 1 else None

    if runner_up_score is not None and (top_score - runner_up_score) < MARGIN_FOR_SINGLE_HIT:
        return (
            None,
            [],
            0.0,
            f"only 1 confident phrase hit each for {len(ranked)} competing episodes, "
            f"top margin {top_score - runner_up_score:.1f} < {MARGIN_FOR_SINGLE_HIT}",
        )

    hits = hits_by_episode[top_episode]
    layer_indexes = sorted({idx for idx, _ in hits})
    confidence = hits[0][1] / 100.0
    return top_episode, layer_indexes, confidence, ""


def ground_truth_cluster_for(
    episode_file: str, layer_indexes: list[int], canvas_dir: Path = DEFAULT_CANVAS_DIR
) -> list[int]:
    """Expand matched layer_indexes to their full local scene cluster(s) via the same
    background-anchored clustering augment.py uses for synthetic training crops -- this is what
    stage 6 (Task 6.2) evaluates cut regions against, not the bare matched layer_indexes.
    """
    gt_path = canvas_dir / f"{Path(episode_file).stem}.gt.json"
    if not gt_path.is_file():
        return layer_indexes
    ref_data = json.loads(gt_path.read_text())
    regions = [
        GroundTruthRegion(
            layer_index=r["layer_index"],
            kind=r["kind"],
            kind_source=r["kind_source"],
            bbox=tuple(r["bbox"]),
        )
        for r in ref_data["regions"]
    ]
    clusters = cluster_layers_by_scene(regions)
    matched_set = set(layer_indexes)
    result_indexes: set[int] = set()
    for cluster in clusters:
        cluster_indexes = {r.layer_index for r in cluster}
        if cluster_indexes & matched_set:
            result_indexes |= cluster_indexes
    return sorted(result_indexes) if result_indexes else layer_indexes


def align_photo(
    photo_path: Path, corpus: list[dict], canvas_dir: Path = DEFAULT_CANVAS_DIR
) -> list[PageAlignmentResult]:
    image = cv2.imread(str(photo_path))
    if image is None:
        return [
            PageAlignmentResult(
                photo_path.name, 0, None, [], [], 0.0, "skipped_no_match", "failed to read image file"
            )
        ]

    pages = detect_pages(image)
    if not pages:
        return [
            PageAlignmentResult(
                photo_path.name, 0, None, [], [], 0.0, "skipped_no_match", "no page regions detected"
            )
        ]

    results = []
    for i, page in enumerate(pages):
        x0, y0, x1, y1 = page.bbox
        crop = image[y0:y1, x0:x1]
        rect_result = rectify_page(crop)
        text = ocr_page(rect_result.rectified)
        episode, layer_indexes, confidence, reason = match_page_to_episode(text, corpus)

        if episode is None:
            results.append(
                PageAlignmentResult(photo_path.name, i, None, [], [], 0.0, "skipped_no_match", reason)
            )
            continue

        cluster = ground_truth_cluster_for(episode, layer_indexes, canvas_dir)
        results.append(
            PageAlignmentResult(
                photo_path.name, i, episode, layer_indexes, cluster, confidence, "matched", ""
            )
        )
    return results


def align_all(
    lowcamera_dir: Path = DEFAULT_LOWCAMERA_DIR,
    out_path: Path = DEFAULT_OUT,
    canvas_dir: Path = DEFAULT_CANVAS_DIR,
) -> list[PageAlignmentResult]:
    corpus = load_ocr_corpus()
    results = []
    for f in sorted(lowcamera_dir.glob("*.jpg")):
        results.extend(align_photo(f, corpus, canvas_dir))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as fh:
        for r in results:
            fh.write(json.dumps(asdict(r)) + "\n")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    results = align_all(out_path=args.out)
    matched = sum(1 for r in results if r.status == "matched")
    print(f"{matched}/{len(results)} pages matched -> {args.out}")


if __name__ == "__main__":
    main()
