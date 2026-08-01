# Noto Sans fonts (vendored)

Source: [Google Noto Fonts](https://fonts.google.com/noto), obtained via Homebrew casks
(`font-noto-sans-thai`, `font-noto-sans-sc`, `font-noto-sans-kr`, `font-noto-sans-kannada`,
`font-noto-sans-jp`, `font-noto-sans-devanagari`, `font-noto-sans-tamil`, `font-noto-sans-bengali`,
`font-noto-sans-hebrew`, `font-noto-sans-arabic`).

License: **SIL Open Font License, Version 1.1** (all Noto fonts are OFL-1.1 licensed by Google;
see <https://openfontlicense.org> for the license text). Free to use, modify, and redistribute,
including embedded in this repository, per the OFL terms.

## Language coverage in this pipeline

| File | Covers (this pipeline's languages) |
|------|-------------------------------------|
| `NotoSansThai[wdth,wght].ttf` | th |
| `NotoSansSC[wght].ttf` | zh |
| `NotoSansKR[wght].ttf` | ko |
| `NotoSansKannada[wdth,wght].ttf` | kn |
| `NotoSansJP[wght].ttf` | ja |
| `NotoSansDevanagari[wdth,wght].ttf` | hi, mr, ne (all Devanagari script) |
| `NotoSansTamil[wdth,wght].ttf` | ta |
| `NotoSansBengali[wdth,wght].ttf` | bn |
| `NotoSansHebrew[wdth,wght].ttf` | he (RTL) |
| `NotoSansArabic[wdth,wght].ttf` | ar (RTL) |

See `scripts/render_shaped.py` for the language -> font-file mapping used at render time.
