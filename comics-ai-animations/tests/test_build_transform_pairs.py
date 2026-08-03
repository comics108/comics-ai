import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import transforms_bridge as tb
from build_transform_pairs import build_all, build_pairs_for_file


def test_build_pairs_for_file_on_a_real_known_file():
    path = next(p for p in tb.find_comics_files() if p.name == "8a89f7d689fb441ea280cd782276bd7a.comics")
    pairs = build_pairs_for_file(path)
    assert len(pairs) == 200  # real total layer count for this file (confirmed earlier session)
    kinds = {p.kind for p in pairs}
    assert kinds <= {"background", "art", "character", "balloon"}
    # every pair covers all four reveal slots, even if None
    assert all(set(p.reveal.keys()) == {"translate", "scale", "rotate", "alpha"} for p in pairs)


def test_build_all_covers_all_27_files_with_real_layer_counts(tmp_path):
    out = tmp_path / "pairs.jsonl"
    counts = build_all(out_path=out)
    assert len(counts) == 27
    assert sum(counts.values()) == 4594  # real total dataset layer count, confirmed this session
    assert out.is_file()
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 4594
