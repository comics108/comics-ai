# comics-ai-baloons

Batch pipeline that reads `dataset/*.comics` (read-only) and
`dataset/Translation - Mahabharata Book 1.csv`, and produces new `.comics` files with balloon text
rendered in every language present in the CSV, wherever a balloon can be confidently matched.

Full design: see `flows/sdd-comics-ai-baloons/` (Requirements, Specifications, Plan) at the repo
root.

**`dataset/` is read-only input and is never modified by anything in this app.** All output goes to
`work/` (gitignored).

## Setup

```bash
cd apps/comics-ai-baloons
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium   # needed for non-Latin/Cyrillic script rendering (stage 6b)
```

System dependencies:

- **Tesseract OCR** with `eng` and `rus` language data (e.g. `brew install tesseract tesseract-lang`
  on macOS).

## Pipeline stages

Each stage is a script under `scripts/`, reading/writing `.jsonl` files (and cached assets) under
`work/`, so any stage can be re-run in isolation:

```
survey_dataset.py   # done — catalogs naming conventions per file -> work/survey.json
discover.py          # structural balloon-layer discovery -> work/balloons.jsonl
extract.py           # stitch tiles per balloon -> work/extracted/
ocr.py                # Tesseract on en/ru -> work/ocr.jsonl
csv_loader.py + match.py   # fuzzy-match balloons to translation CSV rows -> work/matches.jsonl
lettering_features.py + classify.py   # machine-set vs. hand-lettered -> work/lettering.jsonl
erase.py, layout.py, render_latin.py, render_shaped.py, render_handlettered.py
                      # empty-balloon synthesis + per-language rendering -> work/renders/
package.py            # re-tile + write new .comics files -> work/output/
report.py             # work/report.md, work/report.jsonl
pipeline.py           # runs all of the above in order, resumable per-stage
```

Run the full pipeline:

```bash
python scripts/pipeline.py
```

## Directory layout

- `scripts/` — pipeline code (tracked in git)
- `work/` — all generated/intermediate data, including output `.comics` files (gitignored, never
  committed)
