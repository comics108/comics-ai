# Deferred / follow-up items

Tracked per the SDD flow's Requirements decisions (`flows/sdd-comics-ai-baloons/01-requirements.md`,
"Deferred" section) plus items discovered during implementation. None of these block the current
pipeline; all are explicit scope cuts, not oversights.

## From Requirements (deferred at the start)

1. **Arbitrary new balloon input.** This iteration only handles balloons that already exist in
   `dataset/` (erase existing text → insert different text). Accepting a brand-new balloon
   contour/mask with no existing example was deferred. Revisit if it turns out cheap given the
   current architecture — `erase.py`'s single-image approach and `layout.py`'s interior-rect
   detection don't actually depend on the balloon pre-existing in the dataset, so supporting a
   fresh hand-drawn outline may already be close to free; the missing piece would be a matching
   *target* language text (no CSV row to key off) and an entry point that isn't the discover/match
   pipeline.
2. **Automatic font matching.** The future editor should eventually auto-pick/match a font from a
   balloon's existing lettering examples, rather than the fixed manually-chosen Shantell Sans used
   here. Deferred; `render_latin.py`'s `font_path` parameter is already pluggable (not hardcoded
   inline), so swapping in an auto-selected font later shouldn't require restructuring.

## Discovered during implementation

3. **Track 6b (hand-lettered balloon rendering) is flag-only, not a trained model.** Real count in
   this dataset: 2/825 balloons, both of which also fail CSV matching. Revisit if a future dataset
   surfaces meaningfully more hand-lettered examples — `render_handlettered.py` is an intentional
   no-op stub, not a partial implementation.
4. **CSV translation coverage gaps.** Two distinct gaps, both content issues in
   `dataset/Translation - Mahabharata Book 1.csv`, not pipeline defects:
   - Only ~90% of balloons *within the content this CSV covers* match; the CSV appears to not
     cover 5 of the 27 `.comics` files at all (the 2022-production-era batch — 0% match rate, byte
     evidence in `04-implementation-log.md`).
   - 10 of the 20 target languages have **zero** rows anywhere in the CSV: kn, fr, pt, tr, vi, ta,
     mr, bn, ne, he, ar (including both RTL languages). The rendering mechanism was validated to
     work correctly for all of these using placeholder text — there's simply no real text to
     render yet. If/when translations for these languages (or for the missing 5 files) become
     available, the pipeline needs no changes to pick them up.
5. **`apps/comics-editor-v2.9` integration.** Out of scope this iteration by design. Output
   `.comics` files are schema-compatible today (verified against the real C# model) for en/ru/hi,
   and forward-compatible for the other 17 languages pending an additive `Cultures` enum extension
   in the editor — see Specifications' "Editor Schema Ground Truth" section for the exact index
   table to extend by.
6. **Verification gap**: could not open an output `.comics` file in the actual
   `apps/comics-editor-v2.9` GUI (a Flutter/WPF desktop app) inside this environment. Structural
   validation against the real schema was substituted; a real open-in-editor check is worth doing
   before treating this pipeline's output as production-ready.
