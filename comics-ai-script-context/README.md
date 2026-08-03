# comics-ai-script-context

Turns an episode's real narrative/dialogue text into a structured record of named characters,
props, and locations, using a local Ollama model (`qwen2.5-coder:32b`, chosen by real comparison —
see `flows/sdd-comics-ai-script-context/01-requirements.md`). No paid API, no generated art — text
in, structured text out.

## Pipeline

```
scripts/scene_models.py         # CharacterMention / SceneExtraction dataclasses (incl. text_source provenance)
scripts/extract_scene.py        # prompt building, JSON parsing (+ brace-matching fallback), ollama call
scripts/ocr_dialogue_source.py  # criterion-2 fallback: an episode's own OCR'd balloon dialogue
scripts/run_all.py              # full run across all 27 real episode files + honest coverage report
```

Run: `python3 scripts/run_all.py` (from this directory). Output goes to `work/scenes/*.json` and
`work/report.md` (both gitignored, real pipeline output, not committed).

## Real coverage (as of 2026-08-02) — 27 of 27 episodes, two source tiers

**All 27 episodes now extract successfully — 0 failed, 0 "no source text"**, per
`flows/sdd-comics-ai-transformations`' criterion 2 (extending this flow's original 6/27 ceiling).
Two source tiers, kept explicit via `SceneExtraction.text_source`, never silently merged:

- **`spiritual_text` (6 episodes, higher trust)**: hand-verified narrative prose (reused from
  `apps/comics-ai/comics-positioning/scripts/text_context.py::VERIFIED`) — episode 21 and the
  06/08/09/10/11 Kartavirya cluster.
- **`ocr_dialogue` (21 episodes, broader but shallower)**: the episode's own real OCR'd balloon
  dialogue (`comics-ai-baloons/work/ocr.jsonl`, which — checked directly — already covers all 27
  episodes structurally, independent of any photo-matching), concatenated in layer-index order.
  Real, generally plausible extractions across previously-uncovered episodes (e.g. `f8614207...`:
  Sri Krsna, Yashoda, Utkacha; `c4f04778...`: Karna, Parashurama, Bhumi, Arjun, Krishna), including
  one independent confirmation of an earlier hypothesis: `97cf25db...` (titled
  "12_defy_the_kshatriyas" in `Comics_Episodes.csv`) extracts "RAM" — consistent with it being part
  of the same Parashurama-vs-Kshatriyas arc as the already-known Kartavirya cluster, guessed from
  dialogue-style clues alone during `sdd-comics-ai-transformations`' criterion 4 investigation
  before this run confirmed it independently.

Of the original 6 `spiritual_text` episodes: 1 (`09_magic_cow_kamadhenu`) genuinely has no named
character in its excerpt (correctly returns zero characters, flagged in the report) — its excerpt
only mentions a possessed object ("Jamadagni's cow"), not a person acting.

## Known, disclosed limitations

Found during the real model-comparison spike (Requirements) and reproduced in the full run — not
fixed, and not silently hidden:

- **Coreference misses**: an epithet ("the mighty ruler of the Haihaya tribe") is not always linked
  back to the character's proper name mentioned elsewhere in the same excerpt ("Kartavirya"). Real
  example: `10_the_brahmanas_do_not_have_to_fight`'s extraction names the character "mighty ruler",
  not "Kartavirya", even though the source excerpt establishes they're the same person.
- **Occasional non-person entries**: e.g. "four boys" (a plural/collective reference) sometimes
  appears in `characters` alongside real named individuals. Not filtered out — the schema doesn't
  distinguish "named individual" from "described group" today.
- Every `work/scenes/*.json` file keeps the model's full `raw_model_output` alongside the parsed
  result specifically so these failure modes can be spot-checked by a human, not just trusted.

## Adoption contract for consumer flows (documented here, not built)

This flow does not modify any of the following — each is that flow's own future work:

- **`sdd-comics-ai-multimodal`** (character identity): a `CharacterMention.name` for an episode
  already known to contain that episode's cut regions is a candidate replacement for the current
  episode-name-token + visual-clustering heuristic (which produces generic labels like "the-2").
- **`sdd-comics-ai-positioning`** (`text_context` feature): `positioner_features.py`'s
  `text_context_length` (currently just a character count of OCR'd dialogue) could be replaced with
  a semantic feature derived from a `SceneExtraction` — e.g. keyword overlap between
  `CharacterMention.action_or_state` and a region's own OCR'd dialogue.
- **`vdd-comics-editor-systematization-uiux`** (variant tag): `CharacterMention.action_or_state`
  (e.g. "reflecting on actions", "choosing husband") is a direct candidate source for that flow's
  proposed pose/emotion/action tag per character-library crop — needs that flow's own taxonomy
  design to map free text to a controlled vocabulary.
- **`sdd-comics-ai-positioning`** (reading-order cross-check): for source pages with genuinely
  ambiguous panel order (irregular/staggered layouts, not simple grids), the narrative's implied
  action sequence is a candidate independent signal to confirm/correct a geometric guess — see that
  flow's `_status.md` Blockers for the specific `reading_order_index` bug this could help validate.

## What this flow deliberately does not do

- No image/video generation — text/structured-data output only.
- No claim that `ocr_dialogue`-sourced extractions are as reliable as `spiritual_text` ones — broad
  coverage was a real, separate goal (criterion 2) from narrative depth; the two tiers are kept
  distinguishable via `text_source`, not blended into one undifferentiated signal.
- Does not adopt `vendors/anima`'s generation-oriented DSL fields (camera framing, zone placement,
  TTS/lipsync) — only its entity/action decomposition idea, since this codebase processes existing
  hand-drawn art rather than generating new visuals.
