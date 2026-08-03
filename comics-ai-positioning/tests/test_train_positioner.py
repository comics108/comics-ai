import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from train_positioner import build_training_matrix, train
from spacing_stats import compute_stats
from evaluate_positioning import choose_held_out
from pathlib import Path as P


def test_build_training_matrix_on_real_data():
    pairs_dir = P(__file__).resolve().parents[1] / "work" / "train_pairs"
    episode_stems = [p.stem for p in sorted(pairs_dir.glob("*.jsonl"))]
    held_out = choose_held_out(episode_stems)
    train_stems = [s for s in episode_stems if s not in held_out]
    stats = compute_stats(exclude_episode_stems=held_out)

    X, y = build_training_matrix(train_stems, stats)
    assert len(X) == len(y)
    assert len(X) > 100  # real data: 392 total pairs, most in the training split
    assert all(isinstance(v, float) for row in X for v in row)


def test_train_produces_a_loadable_model(tmp_path):
    model_path = tmp_path / "model.joblib"
    summary = train(model_path=model_path)
    assert model_path.is_file()
    assert summary["train_examples"] > 100

    import joblib

    model = joblib.load(model_path)
    assert hasattr(model, "predict")
