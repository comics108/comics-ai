#!/usr/bin/env python3
"""Read-only survey of dataset/*.comics — catalogs layer-naming conventions,
language suffixes, tile counts. Never writes to dataset/; output goes to
apps/comics-ai-baloons/work/.
"""
import json
import re
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
DATASET_DIR = REPO_ROOT / "dataset"
OUT_DIR = Path(__file__).resolve().parents[1] / "work"
OUT_DIR.mkdir(parents=True, exist_ok=True)

BALLOON_PATTERNS = [
    # (name, regex over the layer basename without extension)
    ("b_prefixed", re.compile(r"^b(\d+)_([a-zA-Z]+)_(\d+)_(\d+)_(\d+)$")),
    ("Text_lang_prefixed", re.compile(r"^Text_([a-zA-Z]+)(\d+)_(\d+)_(\d+)_(\d+)$")),
]


def classify_layer(basename: str):
    for kind, rx in BALLOON_PATTERNS:
        m = rx.match(basename)
        if m:
            return kind, m
    return None, None


def survey_file(path: Path):
    result = {
        "file": path.name,
        "size_bytes": path.stat().st_size,
        "data_json_ok": False,
        "width": None,
        "height": None,
        "num_layers_in_json": None,
        "layer_name_kinds": {},
        "balloon_ids": {},  # kind -> set of ids (as list) -> langs
        "unmatched_samples": [],
        "error": None,
    }
    try:
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            try:
                raw = zf.read("data.json").decode("utf-8-sig")
                data = json.loads(raw)
                result["data_json_ok"] = True
                result["width"] = data.get("width")
                result["height"] = data.get("height")
                result["num_layers_in_json"] = len(data.get("layers", []))
            except Exception as e:  # noqa: BLE001
                result["error"] = f"data.json: {e}"

            balloon_ids = {}
            kind_counts = {}
            unmatched = []
            for n in names:
                if not n.startswith("layers/") or n.endswith("/"):
                    continue
                base = Path(n).stem  # strip .png/.jpg
                kind, m = classify_layer(base)
                if kind is None:
                    unmatched.append(base)
                    continue
                kind_counts[kind] = kind_counts.get(kind, 0) + 1
                if kind == "b_prefixed":
                    bid, lang = m.group(1), m.group(2)
                elif kind == "Text_lang_prefixed":
                    lang, bid = m.group(1), m.group(2)
                else:
                    continue
                balloon_ids.setdefault(kind, {}).setdefault(bid, set()).add(lang)

            result["layer_name_kinds"] = kind_counts
            result["balloon_ids"] = {
                kind: {bid: sorted(langs) for bid, langs in ids.items()}
                for kind, ids in balloon_ids.items()
            }
            result["unmatched_samples"] = sorted(set(unmatched))[:15]
            result["unmatched_total"] = len(set(unmatched))
    except Exception as e:  # noqa: BLE001
        result["error"] = str(e)
    return result


def main():
    comics_files = sorted(DATASET_DIR.glob("*.comics"))
    survey = [survey_file(p) for p in comics_files]

    out_path = OUT_DIR / "survey.json"
    out_path.write_text(json.dumps(survey, indent=2, ensure_ascii=False))

    print(f"Surveyed {len(survey)} files -> {out_path}")
    for s in survey:
        kinds = s["layer_name_kinds"]
        langs_all = set()
        for kind, ids in s["balloon_ids"].items():
            for bid, langs in ids.items():
                langs_all.update(langs)
        n_balloons = sum(len(ids) for ids in s["balloon_ids"].values())
        print(
            f"{s['file'][:12]}... "
            f"dims={s['width']}x{s['height']} "
            f"kinds={kinds} "
            f"balloons={n_balloons} "
            f"langs={sorted(langs_all)} "
            f"unmatched={s.get('unmatched_total')} "
            f"err={s['error']}"
        )


if __name__ == "__main__":
    main()
