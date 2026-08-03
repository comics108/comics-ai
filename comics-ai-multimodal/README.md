# comics-ai-multimodal

From-scratch-trained segmentation ("cutting") pipeline: decomposes a flattened comic page image
(starting with real camera photos of the printed Mahabharata book) back into its constituent,
kind-tagged layer regions (background/character/balloon/sound-visual/motion-fx), at a quality
sufficient to reassemble a valid `.comics` file.

Full design: see `flows/sdd-comics-ai-multimodal/` (Requirements, Specifications, Plan) at the repo
root.

**`dataset/` is read-only input and is never modified.** All output — rendered canvas references,
trained model checkpoints, cut regions, the character/environment library, packaged `.comics`
files, and reports — goes to `work/` (gitignored).

`apps/comics-ai/comics-ai-baloons/` is **invoked, never modified** — balloon-region deep processing
(OCR, translation matching, erase/render) is reused from that pipeline via
`scripts/baloons_bridge.py`, not reimplemented here.

## Setup

```bash
cd apps/comics-ai/comics-multimodal
python3.13 -m venv --system-site-packages .venv   # --system-site-packages reuses an existing
                                                    # global torch install if present, to save disk
source .venv/bin/activate
pip install -r requirements.txt
```

No system dependencies beyond Python 3.13 (torch/torchvision provide their own MPS/CPU backends on
Apple Silicon — no CUDA/Tesseract/Playwright needed here, unlike `comics-ai-baloons`).

## Pipeline stages

Each stage is a script under `scripts/`, reading/writing cached files under `work/`, so any stage
can be re-run in isolation:

```
render_canvas.py      # composite each dataset .comics file + emit ground-truth layer map -> work/canvas/
augment.py             # synthetic camera-realism degradation of canvas crops -> work/train_pairs/
train_segmenter.py     # train the segmentation model on synthetic pairs -> work/models/
align_photo.py         # auto-align a real photo to its source episode/page, no manual mapping -> work/alignment.jsonl
infer_segmenter.py     # run the trained model on an aligned photo -> work/regions.jsonl
route_balloons.py       # hand off balloon regions to comics-ai-baloons (reused, not reimplemented)
build_library.py       # cluster character/environment crops into a gallery -> work/library/
package.py              # assemble a new .comics file -> work/output/
report.py               # per-photo match/cut/kind/IoU summary -> work/report.md, work/report.jsonl
pipeline.py             # runs all of the above in order, resumable per-stage
```

## Running tests

```bash
.venv/bin/python -m pytest tests/ -v
```
