import json
from io import StringIO
from pathlib import Path

from langfeat_analysis.pipeline.progress import TerminalReporter
from langfeat_analysis.pipeline.model_cache import required_model_assets
from langfeat_analysis.pipeline.runner import (
    BatchPipeline,
    select_transcript_matches,
    transcript_filename_score,
)
from langfeat_analysis.registry import EventRegistry


def _pipeline(tmp_path, audio_dir, text_dir, output_dir):
    process = {
        "enabled": True,
        "input": {
            "audio": {"directory": str(audio_dir), "pattern": "*.wav"},
            "existing_transcripts": {
                "directory": str(text_dir),
                "patterns": ["*.csv", "*.txt", "*.TextGrid"],
            },
            "matching": {"threshold": 0.72, "min_margin": 0.05},
        },
        "output": {
            "directory": str(output_dir),
            "file_template": "{stimulus}-annotations.csv",
        },
        "settings": {},
    }
    config = {"version": 1, "processes": {"transcripts": process}}
    terminal = StringIO()
    registry = EventRegistry(tmp_path / "logs", filename="events.jsonl")
    pipeline = BatchPipeline(config, tmp_path, registry, TerminalReporter(terminal))
    return pipeline, process, terminal, registry


def test_collector_names_and_section_numbers_select_best_transcript(tmp_path):
    audio = tmp_path / "alice_audio_lpp-section-1-EN_12345.wav"
    correct = tmp_path / "alice_text_lppEN_section1_98765.TextGrid"
    wrong = tmp_path / "alice_text_lppEN_section2_54321.TextGrid"

    assert transcript_filename_score(audio, correct) >= 0.72
    assert transcript_filename_score(audio, correct) > transcript_filename_score(
        audio, wrong
    )


def test_equal_alias_counts_enable_one_to_one_matches_below_threshold(tmp_path):
    audio = [
        tmp_path / "cohort_audio_speaker_section_1_alpha.wav",
        tmp_path / "cohort_audio_speaker_section_2_beta.wav",
    ]
    text = [
        tmp_path / "cohort_text_session_section_2_blue.txt",
        tmp_path / "cohort_text_session_section_1_red.txt",
    ]

    matches, ambiguities = select_transcript_matches(
        audio,
        text,
        threshold=0.95,
        alias_min_score=0.35,
    )

    assert ambiguities == []
    assert len(matches) == 2
    assert matches[audio[0]].transcript == text[1]
    assert matches[audio[1]].transcript == text[0]
    assert all(match.score < 0.95 for match in matches.values())
    assert {match.method for match in matches.values()} == {"equal_alias_count"}


def test_unequal_alias_counts_do_not_enable_fallback(tmp_path):
    audio = [
        tmp_path / "cohort_audio_speaker_section_1_alpha.wav",
        tmp_path / "cohort_audio_speaker_section_2_beta.wav",
    ]
    text = [tmp_path / "cohort_text_session_section_1_red.txt"]

    matches, _ = select_transcript_matches(
        audio,
        text,
        threshold=0.95,
        alias_min_score=0.35,
    )

    assert matches == {}


def test_only_unmatched_audio_is_transcribed(tmp_path):
    audio_dir = tmp_path / "audio"
    text_dir = tmp_path / "text"
    output_dir = tmp_path / "output"
    audio_dir.mkdir()
    text_dir.mkdir()
    matched_audio = audio_dir / "alice_audio_story_111.wav"
    unmatched_audio = audio_dir / "bob_audio_chapter_1_222.wav"
    matched_text = text_dir / "alice_text_story_999.csv"
    matched_audio.touch()
    unmatched_audio.touch()
    matched_text.write_text("word,onset,offset\nhello,0,1\n", encoding="utf-8")
    pipeline, process, terminal, registry = _pipeline(
        tmp_path, audio_dir, text_dir, output_dir
    )
    calls = []

    class Transcriber:
        def transcribe_audio(self, path):
            calls.append(Path(path))
            output_dir.mkdir(exist_ok=True)
            destination = output_dir / f"{Path(path).stem}-annotations.csv"
            destination.write_text("word,onset,offset\nnew,0,1\n", encoding="utf-8")
            return str(destination)

        def unload_models(self):
            return None

    pipeline._transcriber = Transcriber()
    result = pipeline._run_transcripts(process)

    assert calls == [unmatched_audio]
    assert result.failed == 0
    assert result.succeeded == 1
    assert result.skipped == 1
    skipped = next(item for item in result.items if item.status == "skipped")
    assert skipped.input_path == str(matched_audio)
    assert skipped.output_paths == [str(matched_text)]
    assert "[SKIPPED] transcripts: alice_audio_story_111.wav" in terminal.getvalue()
    events = [json.loads(line) for line in registry.path.read_text().splitlines()]
    event = next(item for item in events if item["event"] == "item_skipped")
    assert event["output_path"] == str(matched_text)
    assert event["details"]["match_score"] >= 0.72


def test_transcription_model_preload_is_skipped_when_every_audio_has_text(tmp_path):
    audio_dir = tmp_path / "audio"
    text_dir = tmp_path / "text"
    output_dir = tmp_path / "output"
    audio_dir.mkdir()
    text_dir.mkdir()
    (audio_dir / "story_audio_part_1_123.wav").touch()
    (text_dir / "story_text_part_1_987.txt").write_text("transcript", encoding="utf-8")
    pipeline, _process, _terminal, registry = _pipeline(
        tmp_path, audio_dir, text_dir, output_dir
    )

    pipeline._preload_transcription_models(["transcripts"])

    assert pipeline._transcriber is None
    events = [json.loads(line) for line in registry.path.read_text().splitlines()]
    skipped = next(item for item in events if item["event"] == "model_load_skipped")
    assert skipped["details"]["matched_audio"] == 1

    assert required_model_assets(pipeline.config, config_dir=tmp_path) == []
