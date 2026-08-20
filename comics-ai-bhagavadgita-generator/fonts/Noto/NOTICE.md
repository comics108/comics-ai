# Noto Sans fonts (vendored)

Source: [Google Noto Fonts](https://fonts.google.com/noto), obtained via Homebrew casks
(`font-noto-sans`, `font-noto-sans-devanagari`), matching `comics-ai-baloons`'s vendoring
precedent exactly.

License: **SIL Open Font License, Version 1.1** (all Noto fonts are OFL-1.1 licensed by Google;
see <https://openfontlicense.org> for the license text). Free to use, modify, and redistribute,
including embedded in this repository, per the OFL terms.

## Language coverage in this pipeline

| File | Covers |
|------|--------|
| `NotoSans-Regular.ttf` / `NotoSans-Bold.ttf` | Cyrillic (Russian translation/comment text), Latin (labels, source markers) |
| `NotoSansDevanagari[wdth,wght].ttf` | Devanagari (Sanskrit `Text`, `Transcription` where applicable) |

See `scripts/render_cards.py` for how these are combined in a single `@font-face` stack per card.
