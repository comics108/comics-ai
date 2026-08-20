import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from register_panorama_pairs import PairEvidence, select_confident_pairs


def _evidence(colour, bw, inliers, ratio=.9):
    return PairEvidence(colour, bw, inliers + 5, inliers, ratio, tuple(range(9)))


def test_registration_selects_unique_high_margin_geometric_matches():
    selected = select_confident_pairs([
        _evidence(1, 3, 180), _evidence(1, 4, 5, .2),
        _evidence(2, 4, 160), _evidence(2, 3, 4, .2),
    ])
    assert [(item.colour_page, item.bw_page) for item in selected] == [(1, 3), (2, 4)]


def test_registration_abstains_on_ambiguous_or_duplicate_match():
    with pytest.raises(ValueError, match="ambiguous"):
        select_confident_pairs([_evidence(1, 3, 60), _evidence(1, 4, 30)])
    with pytest.raises(ValueError, match="same B&W"):
        select_confident_pairs([
            _evidence(1, 3, 180), _evidence(1, 4, 4, .2),
            _evidence(2, 3, 170), _evidence(2, 4, 3, .2),
        ])
