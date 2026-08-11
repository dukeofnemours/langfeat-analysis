import json
from io import StringIO

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from langfeat_analysis.pipeline.progress import TerminalReporter
from langfeat_analysis.pipeline.runner import BatchPipeline
from langfeat_analysis.registry import EventRegistry
from natural_features.core.stimulus import AudioStimulus
from natural_features.features.speech.phonology import (
    CTCModelRuntime,
    ctc_phone_posteriors,
)


class _Tokenizer:
    def convert_ids_to_tokens(self, index):
        return ("AA", "B", "S")[index]


class _Processor:
    tokenizer = _Tokenizer()

    def __call__(self, chunks, **_kwargs):
        longest = max(len(chunk) for chunk in chunks)
        values = torch.zeros((len(chunks), longest), dtype=torch.float32)
        mask = torch.zeros((len(chunks), longest), dtype=torch.int64)
        for index, chunk in enumerate(chunks):
            values[index, : len(chunk)] = torch.from_numpy(chunk)
            mask[index, : len(chunk)] = 1
        return {"input_values": values, "attention_mask": mask}


class _Model:
    def __init__(self):
        self.calls = 0

    def __call__(self, input_values, attention_mask):
        self.calls += 1
        frames = max(1, input_values.shape[1] // 2)
        logits = torch.ones((input_values.shape[0], frames, 3))
        return type("Output", (), {"logits": logits})()

    @staticmethod
    def _get_feat_extract_output_lengths(lengths):
        return torch.clamp(lengths // 2, min=1)


def test_ctc_chunks_are_batched_and_report_progress():
    model = _Model()
    runtime = CTCModelRuntime(_Processor(), model, torch, "cpu", "fake", 10)
    stimulus = AudioStimulus.from_array(np.ones(250, dtype=np.float32), sr_hz=10)
    progress = []

    result = ctc_phone_posteriors(
        stimulus,
        model="fake",
        runtime=runtime,
        chunk_seconds=10,
        batch_size=2,
        progress_callback=lambda completed, total: progress.append((completed, total)),
    )

    assert model.calls == 2
    assert progress == [(0, 3), (2, 3), (3, 3)]
    assert len(result.times_s) == result.values.shape[0]
    assert np.all(np.diff(result.times_s) >= 0)
    assert result.metadata["chunk_count"] == 3
    assert result.metadata["batch_size"] == 2
    assert result.metadata["device"] == "cpu"


def test_pipeline_preloads_ctc_model_once(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    (inputs / "one.wav").touch()
    (inputs / "two.wav").touch()
    config = {
        "version": 1,
        "defaults": {"tr_s": 1.0},
        "batch": {"continue_on_error": True, "workers": 1},
        "processes": {
            "acoustic_phonetics": {
                "enabled": True,
                "input": {"directory": str(inputs), "pattern": "*.wav"},
                "output": {"directory": str(outputs), "file_template": "{title}.json"},
                "settings": {
                    "extractor_options": {
                        "posterior_backend": "ctc",
                        "ctc_model": "fake-model",
                        "ctc_device": "auto",
                    }
                },
            }
        },
    }
    calls = []

    class Runtime:
        device = "mps"

        def close(self):
            calls.append("close")

    def fake_load(**kwargs):
        calls.append(kwargs)
        return Runtime()

    monkeypatch.setattr(
        "natural_features.features.speech.phonology.load_ctc_runtime", fake_load
    )
    registry = EventRegistry(tmp_path / "logs", filename="events.jsonl")
    terminal = StringIO()
    pipeline = BatchPipeline(config, tmp_path, registry, TerminalReporter(terminal))

    def fake_item(_name, _process, path):
        assert pipeline._ctc_runtime is not None
        output = outputs / f"{path.stem}.json"
        output.parent.mkdir(exist_ok=True)
        output.write_text("{}", encoding="utf-8")
        return [output]

    monkeypatch.setattr(pipeline, "_run_audio_item", fake_item)
    report = pipeline.run()

    assert report.status == "success"
    assert calls == [
        {"model": "fake-model", "local_files_only": True, "device": "auto"},
        "close",
    ]
    feedback = terminal.getvalue()
    assert "[PENDING] acoustic_phonetics: loading model fake-model" in feedback
    assert "[SUCCESS] acoustic_phonetics: model fake-model — ready on mps" in feedback
    events = [
        json.loads(line)
        for line in registry.path.read_text(encoding="utf-8").splitlines()
    ]
    assert sum(event["event"] == "model_load_started" for event in events) == 1
    assert sum(event["event"] == "model_load_completed" for event in events) == 1
