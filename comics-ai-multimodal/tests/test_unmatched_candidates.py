import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import unmatched_candidates as uc


def test_compute_adjacency_candidates_agrees_on_both_sides():
    rows = [
        {"photo_file": "a.jpg", "page_index": 0, "status": "matched", "episode_file": "ep1.comics"},
        {"photo_file": "b.jpg", "page_index": 0, "status": "skipped_no_match", "episode_file": None},
        {"photo_file": "c.jpg", "page_index": 0, "status": "matched", "episode_file": "ep1.comics"},
    ]
    result = uc.compute_adjacency_candidates(rows)
    assert result[("b.jpg", 0)] == "ep1.comics"


def test_compute_adjacency_candidates_rejects_disagreement():
    rows = [
        {"photo_file": "a.jpg", "page_index": 0, "status": "matched", "episode_file": "ep1.comics"},
        {"photo_file": "b.jpg", "page_index": 0, "status": "skipped_no_match", "episode_file": None},
        {"photo_file": "c.jpg", "page_index": 0, "status": "matched", "episode_file": "ep2.comics"},
    ]
    result = uc.compute_adjacency_candidates(rows)
    assert result[("b.jpg", 0)] is None


def test_compute_adjacency_candidates_none_at_sequence_edge():
    rows = [
        {"photo_file": "a.jpg", "page_index": 0, "status": "skipped_no_match", "episode_file": None},
        {"photo_file": "b.jpg", "page_index": 0, "status": "matched", "episode_file": "ep1.comics"},
    ]
    result = uc.compute_adjacency_candidates(rows)
    assert result[("a.jpg", 0)] is None


def test_compute_adjacency_candidates_spans_multiple_unmatched_rows():
    # Real-shape case: a whole unmatched run between two matched runs of the same episode.
    rows = [
        {"photo_file": "a.jpg", "page_index": 0, "status": "matched", "episode_file": "ep1.comics"},
        {"photo_file": "b.jpg", "page_index": 0, "status": "skipped_no_match", "episode_file": None},
        {"photo_file": "c.jpg", "page_index": 0, "status": "skipped_no_match", "episode_file": None},
        {"photo_file": "d.jpg", "page_index": 0, "status": "matched", "episode_file": "ep1.comics"},
    ]
    result = uc.compute_adjacency_candidates(rows)
    assert result[("b.jpg", 0)] == "ep1.comics"
    assert result[("c.jpg", 0)] == "ep1.comics"


def test_weak_text_candidates_ranks_by_score_above_threshold():
    corpus = [
        {"source_file": "ep1.comics", "layer_index": 1, "text": "a fairly distinctive phrase here"},
        {"source_file": "ep2.comics", "layer_index": 2, "text": "a somewhat similar phrase around"},
        {"source_file": "ep3.comics", "layer_index": 3, "text": "totally unrelated content zzz"},
    ]
    page = uc.ap.normalize("noise a fairly distinctive phrase here more noise")
    result = uc.weak_text_candidates(page, corpus)
    assert result
    assert result[0][0] == "ep1.comics"
    assert all(score >= uc.WEAK_MATCH_THRESHOLD for _, score in result)


def test_weak_text_candidates_empty_when_nothing_clears_threshold():
    corpus = [{"source_file": "ep1.comics", "layer_index": 1, "text": "totally unrelated content here"}]
    page = uc.ap.normalize("garbled nonsense zzz qqq")
    result = uc.weak_text_candidates(page, corpus)
    assert result == []


def test_all_episode_files_reads_real_csv():
    files = uc.all_episode_files()
    assert len(files) == 27
    assert all(f.endswith(".comics") for f in files)
    assert "d00c610a6f4647dcbd8116014674d255.comics" in files


def test_zero_coverage_episodes_on_fixture(tmp_path):
    csv_path = tmp_path / "episodes.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Id", "File"])
        writer.writerow([1, "/Files/ep1.comics"])
        writer.writerow([2, "/Files/ep2.comics"])
        writer.writerow([3, "/Files/ep3.comics"])

    rows = [
        {"status": "matched", "episode_file": "ep1.comics"},
        {"status": "skipped_no_match", "episode_file": None},
    ]
    result = uc.zero_coverage_episodes(rows, episodes_csv=csv_path)
    assert result == ["ep2.comics", "ep3.comics"]


def test_zero_coverage_episodes_on_real_data_finds_the_known_8():
    rows = uc.load_alignment_rows()
    result = uc.zero_coverage_episodes(rows)
    # Real count as of 2026-08-02 (flows/sdd-comics-ai-transformations/_status.md's Criterion 4
    # Pilot finding) -- 8 of 27 known episodes still have zero matched pages after criterion 3.
    assert len(result) == 8
    assert "97cf25db04534eccbc3495c0fc6fb251.comics" in result
