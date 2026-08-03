# comics-ai-positioning

Given already-cut, kind-tagged regions (`comics-ai/comics-multimodal`'s output), proposes where
each region belongs in the target episode's continuous-strip canvas — the "recomposition/placement"
step Джанава's framing identified as harder than the cutting itself
(`flows/vdd-comics-editor-jhanava/`).

Full design: see `flows/sdd-comics-ai-positioning/` (Requirements, Specifications, Plan,
Implementation Log) at the repo root. This README is the practical/results summary; that flow's
`04-implementation-log.md` has the full blow-by-blow, including every dead end.

**`dataset/` and `comics-multimodal/work/` are read-only inputs, never modified.** All output goes
to `work/` (gitignored).

## Setup

Uses `comics-multimodal`'s existing virtualenv (torch/opencv/sklearn/pytesseract already there) —
no separate environment:

```bash
/path/to/apps/comics-ai/comics-multimodal/.venv/bin/python3 -m pytest tests/
```

System dependencies: Tesseract OCR (already required by `comics-multimodal`/`comics-ai-baloons`).
Phase 6's local-model detour additionally used `ollama` + the `moondream` model (optional, not a
pipeline dependency — see "Local multimodal model" below).

## Pipeline stages

```
positioning_bridge.py   # reads comics-multimodal's work/canvas/*.gt.json + work/alignment.jsonl
data_checkpoint.py      # Task 1.2: real matched-pair count -> 59 pairs, 19 episodes (was 37/16
                         # before sdd-comics-ai-transformations' 2026-08-02 re-matching refinement)
build_pairs.py          # joins align_photo.py's clusters + render_canvas.py's ground truth
                         # -> work/train_pairs/*.jsonl (564 real pairs, 19 episodes)
scene_text.py           # the page-cluster's own real OCR'd dialogue (comics-ai-baloons/work/ocr.jsonl)
text_context.py         # small, hand-verified spiritual_text cross-references (4 episodes only)
spacing_stats.py        # real per-kind height + gap statistics from all 27 ground-truth canvases
baseline_position.py    # rule-based positioner: reading-order stacking, real calibrated spacing
evaluate_positioning.py # held-out (file-wise) evaluation, baseline vs. learned model
positioner_features.py  # shared feature engineering (train/infer skew impossible by construction)
train_positioner.py     # RandomForestRegressor, residual-from-baseline
infer_positioner.py     # applies the trained model on top of the baseline
page_number.py          # Phase 7: folio-number OCR (real negative result, see below)
spike_text_alignment.py # Phase 6 first pass: fuzzy dialogue<->spiritual_text matching (0/27, see below)
```

Run in order: `data_checkpoint.py` → `build_pairs.py` → `spacing_stats.py` →
`evaluate_positioning.py [--model work/positioner_model.joblib]`. `train_positioner.py` before the
`--model` flag is meaningful.

## Results (real, run against real data — nothing in this section is projected/estimated)

### Baseline (shipped recommendation)

Reading-order vertical stacking, calibrated from real per-kind height/gap statistics mined from all
27 ground-truth canvases (not guessed constants). Held-out evaluation, **updated 2026-08-02** after
`sdd-comics-ai-transformations`' re-matching refinement to `comics-multimodal`'s `align_photo.py`
recovered 22 of 99 previously-unmatched pages (59/19 real matched pages/episodes now, was 37/16) —
5 held-out episodes, 158 pairs, stats computed excluding them to avoid leakage:

| Episode | Pairs | Mean error | % of canvas height |
|---|---|---|---|
| 096e28e9... | 41 | 814px | 3.0% |
| 54e9d4bb... | 38 | 970px | 3.1% |
| 8a89f7d6... | 26 | 1130px | 3.4% |
| d00c610a... | 13 | 6120px | 14.7% |
| f737556... | 40 | 1365px | 3.8% |

Weighted mean error: **1479.7px** (was 1467.4px on the smaller 4-episode/78-pair held-out set — a
real, honest ~0.8% change, not an improvement or regression worth acting on by itself). Weighted
mean Spearman rank correlation between predicted reading order and true canvas Y: **0.634** (was
0.542) — a real improvement, on a held-out sample now roughly double the previous size. Net
takeaway: more real matched data confirmed the baseline's performance level rather than changing
it, with higher confidence than before. `d00c610a...`'s much higher error (14.7% of its canvas
height) is a new, real outlier worth investigating in a future pass — not explained here.

### Learned model (RandomForestRegressor, residual-from-baseline) — does not beat baseline

Tried three times now, honestly reported every time:

1. First pass, `text_context` covering 4/16 episodes (hand-verified `spiritual_text` matches only):
   **4.3% worse** than baseline, weighted mean error (1530px vs. 1467px).
2. Second pass, after upgrading `text_context` to the comic's own OCR'd dialogue (392/392 pairs,
   100% coverage): **5.8% worse** (1552px vs. 1467px) — broader/better text coverage did not help.
3. **Third pass, 2026-08-02, re-run against the expanded 564-pair/19-episode dataset** (after
   `sdd-comics-ai-transformations`'s re-matching refinement; `scikit-learn`/`joblib` installed into
   this environment to make this possible): retrained on 406 examples (14 episodes) — **55% worse**
   than baseline, weighted mean error (2294px vs. 1480px), on the new 5-episode/158-pair held-out
   set. Even excluding `d00c610a...` (the real outlier noted above, where the model does especially
   badly — 7761px vs. baseline's already-bad 6120px), the model is still **70% worse** on the
   remaining 4 episodes (1804px vs. 1064px) — not just outlier-driven.

Conclusion: more real data made the learned model's relative disadvantage *larger*, not smaller —
the opposite of the usual expectation that more data helps a learned model catch up to a simpler
baseline. Plausible explanation, not yet confirmed: the newly-recovered episodes' pairs come from
lower-confidence single-hit matches (see `sdd-comics-ai-transformations`), which may carry more
label noise that a flexible model overfits to more than the baseline's robust per-kind median
statistic does — a real, disclosed hypothesis for future investigation, not a settled explanation.
**Recommendation: ship the baseline, not the model** — now on stronger evidence than before,
consistent with
Requirements' own explicit acceptance criterion that this is a valid outcome at this data scale.
Whether the now-larger dataset changes this conclusion is an open question for
`sdd-comics-ai-transformations`, not answered here.

### Text context: two sources, very different reliability

- **`scene_text.py` (primary, recommended)**: the page-cluster's own real OCR'd balloon/caption
  dialogue (`comics-ai-baloons/work/ocr.jsonl`, Tesseract, already proven). 100% relevant by
  construction (it's literally that scene's own text), covers all 16 training-pair episodes. Didn't
  move the positioning-error metric, but is a real, reliable asset for other future uses (character
  identity, dataset QA).
- **`text_context.py` (secondary, narrow)**: hand-verified cross-references to `spiritual_text`
  (Ganguli's 19th-century translation). Automated fuzzy-matching against it — both literal phrase
  matching and TF-IDF — **failed** (0/27 confident matches; TF-IDF was biased toward long, generic
  sections). Real matches exist for only 4 episodes (21 "ambas_plea"; 06/08/09/[10/11 character-arc
  link] "Kartavirya"), found by direct reading, not algorithmically. Two of those (10, 11) were found
  via a local multimodal model detour, below.

### Local multimodal model (`moondream`, Apache-2.0, via `ollama`)

Per explicit direction: local/open-source only, no paid API (none exist in this repo/environment).
Confirmed it produces plausible **scene descriptions** ("a mountain scene with a castle on top...")
but **cannot reliably read in-image text** (empty response to an OCR-style prompt) — this size of
local VLM doesn't replace Tesseract for text. Real value came from directly viewing a real dataset
image myself: found a caption naming "Mahismati, capital of the Haihayas... King Arjuna Kartavirya",
confirming episode 10 belongs to the already-identified 06/08/09 story arc — not from the model
itself, but from the same investigative approach the model is meant to eventually automate.

### Page-number cross-page anchor (Phase 7) — real negative result

Reused `comics-multimodal`'s `detect_pages` to locate real page boundaries (a naive frame-relative
crop failed first). Even with the folio-number region correctly isolated, **Tesseract cannot
reliably read this book's small, stylized digits** — tried 6+ preprocessing/PSM configurations; a
real "67" consistently OCRs as bare "7" (the "6" is never recognized), confirmed systematic across
10 photos (1/10 partial, 9/10 nothing). Added a real correctness safeguard regardless (spread page
numbers must be consecutive — `right == left + 1` — rejecting a lone misread digit rather than
trusting it silently). Page-number → episode mapping (Task 7.2) was not attempted on top of this
unreliable input. **The existing fallback (per-page-cluster relative positioning; absolute placement
left to a human reviewer) already covers this** — nothing downstream needed to change.

## Editor Integration Contract (design only, not built — mirrors `comics-ai-multimodal`'s own precedent)

Additive extension of that flow's `DetectedRegion`/`CuttingEvent` contract:

```dart
class DetectedRegion {
  final String kind;
  final Uint8List maskPng;
  final Rect bbox;
  final double confidence;
  final Offset? proposedPosition;   // this flow's suggested canvas X/Y; null if not computed
  final double? positionConfidence; // null for baseline-sourced proposals (no natural score)
}
```

A future `PositioningReviewCard` (same shape as `BalloonEditorCard`/`CuttingReviewCard`) would let a
corrector drag-adjust `proposedPosition` before it becomes the layer's real `TranslateAnim.X`/`Y` —
never-silent-auto-apply, same rule as every prior AI-assist surface in `apps/comics-editor`. Not
built this iteration.

## What's proven vs. not

**Proven, real, tested**: the baseline positioner works and is calibrated from real data; the
training-pair-construction join (align_photo.py + render_canvas.py) is sound; scene-text extraction
is reliable and broad; a local OSS multimodal model path is technically viable for future
scene-description work.

**Tried and found not to work, honestly**: `spiritual_text` fuzzy-matching at scale (both phrase and
TF-IDF approaches); the learned model beating the baseline (twice); page-number OCR on this photo
set. None of these are silently swept aside — see `flows/sdd-comics-ai-positioning/
04-implementation-log.md` for exactly what was tried and why it didn't work.

**Not attempted**: in-editor review UI (out of scope, design-only contract above); full multi-page
cross-episode assembly beyond the consecutive-integer fallback.
