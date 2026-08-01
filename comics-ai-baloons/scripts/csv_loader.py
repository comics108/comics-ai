#!/usr/bin/env python3
"""Load dataset/Translation - Mahabharata Book 1.csv (read-only) into structured rows.

CSV layout (verified by direct inspection, not assumed):
- Row 0 (0-indexed): ISO language codes, columns 3-22 (20 languages) -- in the exact same order
  as scripts/languages.py's LANGUAGES table (en, ru, uk, th, zh, ko, kn, es, fr, pt, ja, tr, vi,
  hi, ta, mr, bn, ne, he, ar). This is not a coincidence to maintain by hand -- languages.py's
  order was derived from this same CSV column order.
- Rows 1-7: further header/metadata/attribution rows, not data.
- Data rows: identified by column 1 matching "P<page>_<bubble>" (e.g. "P1_001") -- this is a more
  robust data-row filter than a fixed row offset, since it doesn't depend on the exact header
  block size staying constant if the CSV is regenerated.
- Column 0: chapter/page-range label, usually blank except on a row that starts a new chapter.
- Column 1: the bubble id ("P<page>_<bubble>").
- Column 2: bubble type ("speech", "caption", or occasionally "empthy"/"empty"/blank).
- Columns 3-22: per-language text, sparse (not every row has every language filled).
"""

from __future__ import annotations

import csv
import re
from dataclasses import dataclass, field
from pathlib import Path

import languages

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CSV_PATH = REPO_ROOT / "dataset" / "Translation - Mahabharata Book 1.csv"

ROW_ID_PATTERN = re.compile(r"^P\d+_\d+$")
ISO_ROW_INDEX = 0
LANG_COL_START = 3  # column index of the first language ("en")


@dataclass
class CsvRow:
    row_id: str  # e.g. "P1_001"
    bubble_type: str  # "speech" | "caption" | other/blank
    chapter_label: str
    texts: dict[str, str] = field(default_factory=dict)  # lang code -> text, sparse

    def text_for(self, lang_code: str) -> str | None:
        return self.texts.get(lang_code) or None


def _iso_column_map(iso_row: list[str]) -> dict[int, str]:
    """column index -> ISO code, for the known language columns."""
    col_to_lang: dict[int, str] = {}
    for i, code in enumerate(iso_row):
        code = code.strip()
        if i >= LANG_COL_START and languages.is_known_language(code):
            col_to_lang[i] = code
    return col_to_lang


def load_csv(path: Path = DEFAULT_CSV_PATH) -> list[CsvRow]:
    with path.open(encoding="utf-8") as f:
        rows = list(csv.reader(f))

    col_to_lang = _iso_column_map(rows[ISO_ROW_INDEX])
    # Sanity check: every language in languages.LANGUAGES is present in the CSV header. Column
    # *order* legitimately differs -- languages.py reorders "hi" to index 2 to match the real
    # editor's Cultures enum (En, Ru, Hi), which the CSV's natural column order knows nothing
    # about. Only the *set* of languages needs to agree, not the order.
    found_set = set(col_to_lang.values())
    assert found_set == set(languages.LANGUAGES), (
        "CSV language columns don't match scripts/languages.py -- "
        f"CSV has {sorted(found_set)}, expected {sorted(languages.LANGUAGES)}"
    )

    out: list[CsvRow] = []
    chapter_label = ""
    for row in rows:
        if len(row) <= 1:
            continue
        row_id = row[1].strip()
        if not ROW_ID_PATTERN.match(row_id):
            continue
        if row[0].strip():
            chapter_label = row[0].strip()
        bubble_type = row[2].strip() if len(row) > 2 else ""
        texts = {}
        for col, lang_code in col_to_lang.items():
            if col < len(row) and row[col].strip():
                texts[lang_code] = row[col].strip()
        out.append(
            CsvRow(
                row_id=row_id,
                bubble_type=bubble_type,
                chapter_label=chapter_label,
                texts=texts,
            )
        )
    return out


def main() -> None:
    rows = load_csv()
    print(f"Loaded {len(rows)} rows from {DEFAULT_CSV_PATH}")
    by_type: dict[str, int] = {}
    for r in rows:
        by_type[r.bubble_type] = by_type.get(r.bubble_type, 0) + 1
    print("By type:", by_type)


if __name__ == "__main__":
    main()
