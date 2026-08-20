import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate_gold_maskrcnn import macro_f1, per_class_f1


def test_macro_f1_counts_missing_prediction_as_false_negative():
    assert macro_f1([1, 1, 2, 2], [1, None, 2, None]) == 2 / 3
    assert macro_f1([1, 2], [None, None]) == 0
    assert per_class_f1([1, 1, 2], [1, 1, 1])[2] == 0
