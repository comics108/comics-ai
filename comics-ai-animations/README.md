# comics-ai-transformations

Criterion 1 (`flows/sdd-comics-ai-transformations/`): proposes a region's `Anim` reveal
animation — which properties (translate/scale/rotate/alpha) animate as the reader scrolls past it,
and how — calibrated from real per-kind statistics mined from all 27 real `.comics` files. Same
"real baseline, real held-out evaluation" discipline as the sibling `comics-ai-positioning` flow.

## Pipeline

```
scripts/transforms_bridge.py     # bridges to comics-multimodal's resting_position.py/kind_heuristic.py
scripts/build_transform_pairs.py # extracts (kind -> RevealAnimation) ground truth, all 27 files
scripts/transform_stats.py       # real per-kind occurrence rates + per-property duration/value stats
scripts/baseline_transform.py    # rule-based proposal, calibrated from transform_stats.py
scripts/evaluate_transforms.py   # held-out (file-wise) evaluation vs. a trivial "always static" strawman
scripts/full_pipeline_demo.py    # criterion 5: real cut->position->transform run on one real page
```

Run in order: `build_transform_pairs.py` → `transform_stats.py` → `evaluate_transforms.py`.

## Real findings (checked against the full dataset before writing any baseline code)

- **Occurrence varies sharply by kind**: balloons animate alpha 76.8% / scale 75.3% of the time
  (a fade+grow-in reveal as they appear); backgrounds almost never animate scale/alpha (1-1.5%) and
  animate translate only 32.5% — under a majority threshold, so the baseline correctly predicts
  "static" for backgrounds.
- **Alpha reveals are overwhelmingly a fade-in**: real median (from, to) = (0.0, 1.0).
- **Scale reveals are overwhelmingly a grow-in**: real median (from, to) = (0.6, 1.0).
- **Translate/rotate have no confident direction**: real median delta ≈ 0 with a roughly balanced
  positive/negative split (894 vs. 849 for translate dy, out of 2368 reveals). The baseline predicts
  occurrence + duration for these confidently, but does not fabricate a direction/magnitude it
  cannot support from real data.

## Results (real, held-out, 7 episodes / 1246 layers — nothing projected)

| Property | Baseline occurrence accuracy | Strawman ("always static") | Duration error (when both occur) |
|---|---|---|---|
| translate | 62.5% | 51.8% | 524px median |
| scale | 90.0% | 80.3% | 336px median |
| rotate | 92.5% | 92.5% (tie) | n/a |
| alpha | 90.8% | 80.7% | 295px median |

**Real, disclosed result**: the calibrated baseline beats the trivial strawman on translate, scale,
and alpha (+10-11 points each). It **ties** the strawman on rotate — no region kind's real rotate
occurrence rate clears the 50% majority threshold, so the baseline degenerates to "always predict
no rotation," identical to the strawman for this one property. Not hidden — a real limit of a
majority-vote rule at this occurrence-rate distribution, not a bug.

## Criterion 5: real end-to-end run (`full_pipeline_demo.py`)

Ran cut → position → transform for real on a page from `2a5e3303ba8c42e3ba395dad794164a7.comics`
— an episode with **zero** matched photos before `sdd-comics-ai-transformations`' criterion 3
re-matching refinement recovered it. Real result: 15 real cut regions (5 balloon, 2 character, 8
art), each with a real proposed position (`comics-ai-positioning`'s `baseline_position.py`) and a
real proposed reveal animation (this app's own `baseline_transform.py`) — every balloon region
correctly gets the calibrated alpha+scale fade/grow-in reveal, matching the per-kind pattern found
across the full dataset. Cross-checked against `sdd-comics-ai-script-context`'s real narrative
extraction for this same episode (`ocr_dialogue`-sourced, since it never had a `spiritual_text`
match either): "RAMA...chased and killed Kshatriyas", "Parasurama...destroyed armies 21 times" —
**directly consistent with the episode's own real title in `Comics_Episodes.csv`**:
`2a5e3303ba8c42e3ba395dad794164a7.comics` is titled `13_kshatriyas_extermination` (Order 13,
immediately following `97cf25db...`'s `12_defy_the_kshatriyas` at Order 12 — the same saga's next
chapter, confirmed by directly checking the CSV, not inferred from a neighboring episode's title).
Balloon *text* content itself (which line goes in which balloon) is intentionally not attempted
here — that's `comics-ai-baloons`' own established pipeline, out of this flow's scope.

## What this flow deliberately does not do

- No specific direction/magnitude prediction for translate or rotate reveals — real data doesn't
  support one; occurrence and duration are the honest Must-Have deliverable for these two.
- No learned model (yet) — per this repo's established precedent (`comics-ai-positioning`), a
  learned model is only worth attempting if a real checkpoint justifies it; not attempted this pass.
- No balloon text-content assignment (which real dialogue line goes in which balloon) — reuses
  `comics-ai-baloons`'s own established pipeline for that, not rebuilt here.
