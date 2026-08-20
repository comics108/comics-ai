import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from build_independent_panorama_consensus import maximum_pairs


def test_consensus_matching_is_one_to_one_and_thresholded():
    pairs = maximum_pairs([(.9, 0, 0), (.85, 0, 1), (.82, 1, 0), (.79, 1, 1)], .8)
    assert pairs == [(.9, 0, 0)]


def test_fragment_completeness_thresholds_are_not_weakened():
    source = Path(__file__).resolve().parent.parent / "scripts/build_independent_panorama_consensus.py"
    text = source.read_text()
    assert 'sam_proposal["coverage"] >= .01' in text
    assert "width_fraction >= .08" in text
    assert "height_fraction >= .15" in text
    assert "ink_edge_density < .01" in text
    assert "dark_pixel_density < .02" in text
