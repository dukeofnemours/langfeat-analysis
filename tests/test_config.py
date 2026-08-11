import pytest

from langfeat_analysis.pipeline.config import discover_files, validate_config
from langfeat_analysis.pipeline.errors import ConfigurationError, InputError


def test_workers_error_explains_model_safety():
    config = {
        "version": 1,
        "batch": {"workers": 2},
        "processes": {
            "transcripts": {
                "enabled": True,
                "input": {"directory": "."},
                "output": {"directory": "out"},
            }
        },
    }
    with pytest.raises(ConfigurationError, match="not safe to share"):
        validate_config(config)


def test_explicit_missing_file_is_not_silently_dropped(tmp_path):
    missing = tmp_path / "missing.wav"
    with pytest.raises(InputError, match="missing or non-file"):
        discover_files(
            {"files": [str(missing)]},
            tmp_path,
            label="test.inputs",
        )


def test_multiple_patterns_are_discovered_without_duplicates(tmp_path):
    (tmp_path / "words.csv").touch()
    (tmp_path / "words.tsv").touch()
    paths = discover_files(
        {"directory": str(tmp_path), "patterns": ["*.csv", "*.tsv", "words.*"]},
        tmp_path,
        label="test.annotations",
    )
    assert [path.name for path in paths] == ["words.csv", "words.tsv"]


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("tensorflow_device", "mps", "tensorflow_device"),
        ("inference_batch_size", 0, "inference_batch_size"),
        ("stimulus_batch_size", True, "stimulus_batch_size"),
        ("max_stimulus_batch_seconds", -1, "max_stimulus_batch_seconds"),
        ("memory_fallback", "yes", "memory_fallback"),
    ],
)
def test_openl3_controls_are_validated(key, value, message):
    config = {
        "version": 1,
        "processes": {
            "audio_embeddings": {
                "enabled": True,
                "input": {"directory": "."},
                "output": {"directory": "out"},
                "settings": {"embedding_options": {key: value}},
            }
        },
    }
    with pytest.raises(ConfigurationError, match=message):
        validate_config(config)


@pytest.mark.parametrize(
    ("matching", "message"),
    [
        ({"alias_count_heuristic": "yes"}, "alias_count_heuristic"),
        ({"alias_min_score": -0.1}, "alias_min_score"),
        ({"alias_min_score": "low"}, "alias_min_score"),
    ],
)
def test_transcript_alias_matching_controls_are_validated(matching, message):
    config = {
        "version": 1,
        "processes": {
            "transcripts": {
                "enabled": True,
                "input": {"directory": ".", "matching": matching},
                "output": {"directory": "out"},
            }
        },
    }

    with pytest.raises(ConfigurationError, match=message):
        validate_config(config)
