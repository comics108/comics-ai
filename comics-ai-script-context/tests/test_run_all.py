import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from extract_scene import ExtractionFailed
from scene_models import CharacterMention, SceneExtraction
import run_all as run_all_module


def test_write_report_buckets_all_three_categories(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_module, "WORK_DIR", tmp_path)

    extracted = [
        SceneExtraction(
            episode_file="ep21.comics",
            source_excerpt="...",
            characters=(CharacterMention(name="Amba", action_or_state="pleading"),),
            props=(),
            locations=("Kasi",),
            raw_model_output="{...}",
        ),
    ]
    failed = [("ep99.comics", "malformed JSON")]
    no_source_text = ["ep05.comics", "ep07.comics"]

    run_all_module.write_report(extracted, failed, no_source_text)

    report = (tmp_path / "report.md").read_text()
    assert "Total episodes: 4" in report
    assert "ep21.comics` (spiritual_text): Amba" in report
    assert "ep99.comics`: malformed JSON" in report
    assert "ep05.comics" in report and "ep07.comics" in report


def test_write_report_flags_placeholder_name_and_zero_characters(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_module, "WORK_DIR", tmp_path)

    extracted = [
        SceneExtraction(
            episode_file="ep_heroine.comics",
            source_excerpt="...",
            characters=(CharacterMention(name="Heroine", action_or_state="choosing"),),
            props=(),
            locations=(),
            raw_model_output="{...}",
        ),
        SceneExtraction(
            episode_file="ep_empty.comics",
            source_excerpt="...",
            characters=(),
            props=(),
            locations=(),
            raw_model_output="{...}",
        ),
    ]

    run_all_module.write_report(extracted, [], [])

    report = (tmp_path / "report.md").read_text()
    assert "ep_heroine.comics` (spiritual_text): Heroine **[PLACEHOLDER NAME FLAGGED]**" in report
    assert "ep_empty.comics` (spiritual_text): (none found) **[ZERO CHARACTERS]**" in report


def test_run_all_buckets_episodes_via_stubbed_extract(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_module, "WORK_DIR", tmp_path)
    monkeypatch.setattr(run_all_module, "SCENES_DIR", tmp_path / "scenes")
    monkeypatch.setattr(
        run_all_module, "all_episode_files", lambda: ["ep_ok.comics", "ep_fail.comics", "ep_none.comics"]
    )

    class FakeVerified:
        excerpt = "some real excerpt"

    monkeypatch.setattr(run_all_module, "VERIFIED", {"ep_ok.comics": FakeVerified(), "ep_fail.comics": FakeVerified()})

    def fake_extract(excerpt, episode_file, model=None, text_source="spiritual_text"):
        if episode_file == "ep_ok.comics":
            return SceneExtraction(
                episode_file=episode_file,
                source_excerpt=excerpt,
                characters=(CharacterMention(name="Amba", action_or_state="pleading"),),
                props=(),
                locations=(),
                raw_model_output="{...}",
            )
        raise ExtractionFailed(episode_file, "simulated failure", "raw")

    monkeypatch.setattr(run_all_module, "extract", fake_extract)

    summary = run_all_module.run_all()

    assert [e.episode_file for e in summary["extracted"]] == ["ep_ok.comics"]
    assert summary["failed"] == [("ep_fail.comics", "simulated failure")]
    assert summary["no_source_text"] == ["ep_none.comics"]
    assert (tmp_path / "scenes" / "ep_ok.comics.json").exists()
    assert not (tmp_path / "scenes" / "ep_fail.comics.json").exists()


def test_run_all_falls_back_to_ocr_dialogue_when_not_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(run_all_module, "WORK_DIR", tmp_path)
    monkeypatch.setattr(run_all_module, "SCENES_DIR", tmp_path / "scenes")
    monkeypatch.setattr(run_all_module, "all_episode_files", lambda: ["ep_ocr.comics"])
    monkeypatch.setattr(run_all_module, "VERIFIED", {})
    monkeypatch.setattr(run_all_module, "load_ocr_entries", lambda: [{"source_file": "ep_ocr.comics", "layer_index": 1, "lang_index": 0, "text": "real dialogue"}])

    captured = {}

    def fake_extract(excerpt, episode_file, model=None, text_source="spiritual_text"):
        captured["excerpt"] = excerpt
        captured["text_source"] = text_source
        return SceneExtraction(
            episode_file=episode_file,
            source_excerpt=excerpt,
            characters=(),
            props=(),
            locations=(),
            raw_model_output="{...}",
            text_source=text_source,
        )

    monkeypatch.setattr(run_all_module, "extract", fake_extract)

    summary = run_all_module.run_all()

    assert captured["excerpt"] == "real dialogue"
    assert captured["text_source"] == "ocr_dialogue"
    assert summary["extracted"][0].text_source == "ocr_dialogue"
