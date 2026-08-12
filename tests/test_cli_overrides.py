from langfeat_analysis.cli import apply_directory_overrides
from langfeat_analysis.pipeline.runner import collector_match_keys
from langfeat_analysis.pipeline.runner import BatchPipeline
from langfeat_analysis.registry import EventRegistry


def base_config(tmp_path):
    old_outputs = tmp_path / "old-outputs"
    return {
        "version": 1,
        "processes": {
            "audio_features": {
                "enabled": True,
                "input": {"directory": "audio", "pattern": "*.wav"},
                "output": {"directory": str(old_outputs / "audio/features")},
                "settings": {"features": ["vad"]},
            },
            "transcripts": {
                "enabled": True,
                "input": {"directory": "audio", "pattern": "*.wav"},
                "output": {"directory": str(old_outputs / "transcripts")},
            },
            "text_embeddings": {
                "enabled": True,
                "input": {
                    "annotations": {"directory": str(old_outputs / "transcripts")},
                    "stimuli": {"directory": "audio", "pattern": "*.wav"},
                },
                "output": {"directory": str(old_outputs / "text/embeddings")},
                "settings": {"model_name": "test-model"},
            },
        },
    }


def test_storage_root_uses_collector_audio_and_text_layout(tmp_path):
    storage = tmp_path / "features"
    (storage / "audio").mkdir(parents=True)
    (storage / "text").mkdir()
    output = tmp_path / "results"

    updated = apply_directory_overrides(
        base_config(tmp_path), tmp_path, input_dir=storage, output_dir=output
    )

    assert updated["processes"]["audio_features"]["input"]["directory"] == str(
        storage / "audio"
    )
    annotations = updated["processes"]["text_embeddings"]["input"]["annotations"]
    assert annotations["directory"] == str(storage / "text")
    assert "*.csv" in annotations["patterns"]
    assert updated["processes"]["text_embeddings"]["input"]["matching"][
        "strategy"
    ] == "auto"
    assert updated["processes"]["audio_features"]["output"]["directory"] == str(
        output / "audio/features"
    )
    assert updated["logging"]["directory"] == str(output / "logs")


def test_output_override_preserves_transcript_dependency(tmp_path):
    output = tmp_path / "new-results"
    updated = apply_directory_overrides(
        base_config(tmp_path), tmp_path, output_dir=output
    )
    annotations = updated["processes"]["text_embeddings"]["input"]["annotations"]
    assert annotations["directory"] == str(output / "transcripts")


def test_collector_names_pair_despite_different_checksums():
    audio_keys = collector_match_keys("alice_audio_chapter_1_12345", "audio")
    text_keys = collector_match_keys("alice_text_chapter_1_98765", "text")
    assert audio_keys & text_keys


def test_batch_pairing_understands_collector_filenames(tmp_path):
    audio_dir = tmp_path / "audio"
    text_dir = tmp_path / "text"
    audio_dir.mkdir()
    text_dir.mkdir()
    audio = audio_dir / "alice_audio_chapter_1_12345.wav"
    annotation = text_dir / "alice_text_chapter_1_98765.csv"
    audio.touch()
    annotation.touch()
    process = {
        "input": {
            "annotations": {"directory": str(text_dir), "pattern": "*.csv"},
            "stimuli": {"directory": str(audio_dir), "pattern": "*.wav"},
            "matching": {"strategy": "auto", "annotation_suffix": "-annotations"},
        },
        "output": {"directory": str(tmp_path / "output")},
        "settings": {"model_name": "test"},
    }
    config = {
        "version": 1,
        "processes": {"text_embeddings": {"enabled": True, **process}},
    }
    pipeline = BatchPipeline(
        config, tmp_path, EventRegistry(tmp_path / "logs", filename="test.jsonl")
    )
    pairs, errors = pipeline._match_text_inputs(process)
    assert pairs == [(annotation, audio)]
    assert errors == []


def test_auto_pairing_normalizes_audio_text_markers_and_separators(tmp_path):
    audio_dir = tmp_path / "audio"
    text_dir = tmp_path / "text"
    audio_dir.mkdir()
    text_dir.mkdir()
    audio = audio_dir / "lpp_audio_lppEN-section1.wav"
    annotation = text_dir / "lpp_text_lppEN_section1.TextGrid"
    audio.touch()
    annotation.touch()
    process = {
        "input": {
            "annotations": {"directory": str(text_dir), "pattern": "*.TextGrid"},
            "stimuli": {"directory": str(audio_dir), "pattern": "*.wav"},
            "matching": {"strategy": "auto"},
        },
        "output": {"directory": str(tmp_path / "output")},
        "settings": {"model_name": "test"},
    }
    config = {"version": 1, "processes": {"text_embeddings": {"enabled": True, **process}}}
    pipeline = BatchPipeline(
        config, tmp_path, EventRegistry(tmp_path / "logs", filename="test.jsonl")
    )

    pairs, errors = pipeline._match_text_inputs(process)

    assert pairs == [(annotation, audio)]
    assert errors == []
