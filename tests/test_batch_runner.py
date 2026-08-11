import json
from pathlib import Path

from langfeat_analysis.pipeline.config import validate_config
from langfeat_analysis.pipeline.runner import BatchPipeline
from langfeat_analysis.registry import EventRegistry


def test_batch_continues_after_one_item_fails(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    (inputs / "bad.wav").touch()
    (inputs / "good.wav").touch()
    config = {
        "version": 1,
        "defaults": {"tr_s": 1.0},
        "batch": {"continue_on_error": True, "workers": 1},
        "processes": {
            "audio_features": {
                "enabled": True,
                "input": {"directory": str(inputs), "pattern": "*.wav"},
                "output": {"directory": str(outputs), "file_template": "{title}.json"},
                "settings": {"features": ["vad"]},
            }
        },
    }
    validate_config(config)
    registry = EventRegistry(tmp_path / "logs", filename="events.jsonl")
    pipeline = BatchPipeline(config, tmp_path, registry)

    def fake_process(name, process, path):
        if path.name == "bad.wav":
            raise ValueError("audio header is corrupt")
        destination = outputs / "good.json"
        destination.parent.mkdir()
        destination.write_text("{}", encoding="utf-8")
        return [destination]

    monkeypatch.setattr(pipeline, "_run_audio_item", fake_process)
    report = pipeline.run()

    assert report.failed_items == 1
    assert [item.status for item in report.processes[0].items] == ["failed", "success"]
    assert (outputs / "good.json").is_file()
    events = [
        json.loads(line)
        for line in registry.path.read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "item_failed" for event in events)
    assert any(event["event"] == "item_completed" for event in events)
