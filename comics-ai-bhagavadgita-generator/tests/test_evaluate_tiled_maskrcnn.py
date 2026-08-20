import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def test_tiled_predictor_rejects_bad_geometry_without_loading_model():
    pytest.importorskip("torch")
    from evaluate_tiled_maskrcnn import predict_page
    from PIL import Image

    class EmptyModel:
        def __call__(self, images):
            import torch
            return [{"scores": torch.tensor([]), "boxes": torch.empty((0, 4))}]

    raw, merged = predict_page(EmptyModel(), Image.new("RGB", (700, 100), "white"))
    assert raw == merged == []
