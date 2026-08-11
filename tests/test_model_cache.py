import json
from io import StringIO

from langfeat_analysis.pipeline.model_cache import (
    cache_models,
    configure_model_cache_environment,
    required_model_assets,
)
from langfeat_analysis.pipeline.progress import TerminalReporter
from langfeat_analysis.registry import EventRegistry


def _config(tmp_path):
    common = {
        "enabled": True,
        "input": {"directory": str(tmp_path)},
        "output": {"directory": str(tmp_path / "out")},
    }
    return {
        "version": 1,
        "models": {"cache_directory": str(tmp_path / "models"), "offline": False},
        "processes": {
            "audio_features": {**common, "settings": {"features": ["vad"]}},
            "acoustic_phonetics": {
                **common,
                "settings": {
                    "extractor_options": {
                        "posterior_backend": "ctc",
                        "ctc_model": "example/ctc",
                    }
                },
            },
            "audio_embeddings": {
                **common,
                "settings": {
                    "embedding_options": {
                        "input_repr": "linear",
                        "content_type": "env",
                    }
                },
            },
            "transcripts": {
                **common,
                "settings": {
                    "pytorch_asr_model": "example/asr",
                    "pytorch_aligner_model": "example/aligner",
                },
            },
            "text_embeddings": {
                "enabled": True,
                "input": {
                    "annotations": {"directory": str(tmp_path)},
                    "stimuli": {"directory": str(tmp_path)},
                },
                "output": {"directory": str(tmp_path / "out")},
                "settings": {"model_name": "example/sentence"},
            },
        },
    }


def test_model_inventory_uses_active_backend_and_skips_signal_features(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        "langfeat_analysis.pipeline.model_cache._uses_mlx_backend", lambda: False
    )
    assets = required_model_assets(_config(tmp_path))

    assert [(asset.process, asset.model) for asset in assets] == [
        ("acoustic_phonetics", "example/ctc"),
        ("audio_embeddings", "linear:env"),
        ("transcripts", "example/asr"),
        ("transcripts", "example/aligner"),
        ("text_embeddings", "example/sentence"),
    ]


def test_cache_mode_records_each_asset_without_starting_pipeline(tmp_path, monkeypatch):
    config = _config(tmp_path)
    monkeypatch.setattr(
        "langfeat_analysis.pipeline.model_cache._uses_mlx_backend", lambda: False
    )
    cached_path = tmp_path / "snapshot"
    cached_path.mkdir()
    calls = []

    def fake_cache(asset, config_dir, cache_root, *, download):
        calls.append((asset.model, config_dir, cache_root, download))
        return cached_path

    monkeypatch.setattr(
        "langfeat_analysis.pipeline.model_cache._cache_asset", fake_cache
    )
    registry = EventRegistry(tmp_path / "logs", filename="cache.jsonl")
    terminal = StringIO()
    results = cache_models(
        config,
        tmp_path,
        registry,
        TerminalReporter(terminal),
        download=False,
    )

    assert len(results) == 5
    assert all(call[-1] is False for call in calls)
    events = [json.loads(line) for line in registry.path.read_text().splitlines()]
    assert sum(event["event"] == "model_cache_completed" for event in events) == 5
    assert terminal.getvalue().count("[SUCCESS]") == 5


def test_cache_environment_resolves_relative_directory_and_offline_flags(
    tmp_path, monkeypatch
):
    monkeypatch.delenv("HF_HOME", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    config = {"models": {"cache_directory": "cache", "offline": True}}

    root = configure_model_cache_environment(config, tmp_path)

    assert root == (tmp_path / "cache").resolve()
    assert root.is_dir()
    assert __import__("os").environ["HF_HOME"] == str(root)
    assert __import__("os").environ["HF_HUB_OFFLINE"] == "1"
    assert __import__("os").environ["TRANSFORMERS_OFFLINE"] == "1"
