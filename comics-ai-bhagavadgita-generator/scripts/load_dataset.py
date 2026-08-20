#!/usr/bin/env python3
"""Task 1.2 (flows/comics-ai/sdd-comics-ai-bhagavadgita-generator/03-plan.md): load and normalize
the Bhagavad Gita CSV dataset into CanonicalChapter[18] for BookId=1 (Russian), per
02-specifications.md's "Canonical Data Model" CSV parsing rules.

Real, verified CSV shapes (checked directly against dataset/bhagavadgita/spiritual_text/ this
session, not assumed):
- db_books.csv, db_chapters.csv: comma-delimited.
- Gita_Slokas.csv: semicolon-delimited, utf-8-sig (BOM present).
`dataset/bhagavadgita/` is read-only -- this module only ever opens files for reading.
"""

from __future__ import annotations

import csv
from pathlib import Path

from models import CanonicalChapter, SlokaSource

REPO_ROOT = Path(__file__).resolve().parents[4]
DATASET_DIR = REPO_ROOT / "dataset" / "bhagavadgita" / "spiritual_text"
DEFAULT_BOOK_ID = 1  # Russian edition, per Requirements' approved default

EXPECTED_CHAPTER_COUNT = 18
EXPECTED_SLOKA_COUNT = 663


class DatasetIntegrityError(ValueError):
    """Raised when the dataset doesn't match the real, checked shape this generator depends on --
    per Specifications' failure-handling table: stop before generation, no final files replaced."""


def _read_csv(path: Path, delimiter: str) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f, delimiter=delimiter))


def _parse_positive_int(value: str, field: str, row: dict) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise DatasetIntegrityError(f"{field!r} is not an integer: {value!r} (row={row!r})")
    if parsed <= 0:
        raise DatasetIntegrityError(f"{field!r} must be positive: {parsed!r} (row={row!r})")
    return parsed


def _load_chapters_for_book(
    dataset_dir: Path, book_id: int
) -> dict[int, dict]:
    """Returns real db_chapters.csv rows for `book_id`, keyed by chapter Id (int)."""
    rows = _read_csv(dataset_dir / "db_chapters.csv", delimiter=",")
    by_id: dict[int, dict] = {}
    for row in rows:
        if _parse_positive_int(row["BookId"], "BookId", row) != book_id:
            continue
        chapter_id = _parse_positive_int(row["Id"], "Id", row)
        if not row.get("Name", "").strip():
            raise DatasetIntegrityError(f"Empty chapter Name for chapter Id={chapter_id}")
        by_id[chapter_id] = row
    return by_id


def _load_slokas_for_chapters(
    dataset_dir: Path, chapter_ids: set[int]
) -> dict[int, list[SlokaSource]]:
    """Returns real, validated SlokaSource rows grouped by chapter_id."""
    rows = _read_csv(dataset_dir / "Gita_Slokas.csv", delimiter=";")
    by_chapter: dict[int, list[SlokaSource]] = {}
    seen_orders_by_chapter: dict[int, set[int]] = {}

    for row in rows:
        chapter_id = _parse_positive_int(row["ChapterId"], "ChapterId", row)
        if chapter_id not in chapter_ids:
            continue

        sloka_id = _parse_positive_int(row["Id"], "Id", row)
        order = _parse_positive_int(row["Order"], "Order", row)

        for required_field in ("Text", "Transcription", "Translation"):
            if not row.get(required_field, "").strip():
                raise DatasetIntegrityError(
                    f"Empty required field {required_field!r} for sloka Id={sloka_id} "
                    f"(chapter_id={chapter_id})"
                )

        seen = seen_orders_by_chapter.setdefault(chapter_id, set())
        if order in seen:
            raise DatasetIntegrityError(
                f"Duplicate sloka Order={order} within chapter_id={chapter_id}"
            )
        seen.add(order)

        sloka = SlokaSource(
            id=sloka_id,
            chapter_id=chapter_id,
            order=order,
            name=row.get("Name", "").strip(),
            sanskrit=row["Text"],
            transcription=row["Transcription"],
            translation_ru=row["Translation"],
            comment_ru=row.get("Comment", ""),
            audio_ref=row.get("Audio", ""),
            sanskrit_audio_ref=row.get("AudioSanskrit", ""),
        )
        by_chapter.setdefault(chapter_id, []).append(sloka)

    for chapter_id, slokas in by_chapter.items():
        # Sort by numeric Order; Id is only a deterministic tie-breaker (never expected to matter,
        # since Order-within-chapter is already validated unique above).
        slokas.sort(key=lambda s: (s.order, s.id))

    return by_chapter


def load_book_one(dataset_dir: Path = DATASET_DIR, book_id: int = DEFAULT_BOOK_ID) -> tuple[CanonicalChapter, ...]:
    """Loads, joins, sorts, and validates all real chapters for `book_id`. Raises
    DatasetIntegrityError on any structural problem -- never silently drops or guesses."""
    chapter_rows = _load_chapters_for_book(dataset_dir, book_id)
    if not chapter_rows:
        raise DatasetIntegrityError(f"No chapters found for BookId={book_id}")

    slokas_by_chapter = _load_slokas_for_chapters(dataset_dir, set(chapter_rows.keys()))

    seen_orders: dict[int, int] = {}  # order -> chapter_id, to detect duplicates across chapters
    chapters: list[CanonicalChapter] = []
    for chapter_id, row in chapter_rows.items():
        order = _parse_positive_int(row["Order"], "Order", row)
        if order in seen_orders:
            raise DatasetIntegrityError(
                f"Duplicate chapter Order={order}: chapter_id={chapter_id} and "
                f"chapter_id={seen_orders[order]}"
            )
        seen_orders[order] = chapter_id

        chapters.append(
            CanonicalChapter(
                book_id=book_id,
                chapter_id=chapter_id,
                order=order,
                title=row["Name"].strip(),
                slokas=tuple(slokas_by_chapter.get(chapter_id, [])),
            )
        )

    chapters.sort(key=lambda c: (c.order, c.chapter_id))
    return tuple(chapters)


def verify_dataset_integrity(chapters: tuple[CanonicalChapter, ...]) -> None:
    """The real dataset-integrity checkpoint per Specifications: exactly chapter orders 1-18, no
    gaps, and exactly 663 total slokas for the production (BookId=1) set. A real, checked invariant
    of this specific dataset -- not hardcoded discovery logic (the loader itself derives chapters
    from real relationships; this only asserts the *result* matches what's actually there)."""
    orders = sorted(c.order for c in chapters)
    expected_orders = list(range(1, EXPECTED_CHAPTER_COUNT + 1))
    if orders != expected_orders:
        raise DatasetIntegrityError(
            f"Expected chapter orders {expected_orders}, got {orders}"
        )

    total_slokas = sum(len(c.slokas) for c in chapters)
    if total_slokas != EXPECTED_SLOKA_COUNT:
        raise DatasetIntegrityError(
            f"Expected {EXPECTED_SLOKA_COUNT} total slokas, got {total_slokas}"
        )
