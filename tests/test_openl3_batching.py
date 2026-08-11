import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from langfeat_analysis.preprocessing.audio import AudioVectorizer, OpenL3Runtime
from langfeat_analysis.pipeline.runner import BatchPipeline


class _TensorFlow:
    class keras:
        class backend:
            @staticmethod
            def clear_session():
                return None


def test_openl3_batch_falls_back_to_sequential_with_same_model(tmp_path, monkeypatch):
    inputs = tmp_path / "inputs"
    outputs = tmp_path / "outputs"
    inputs.mkdir()
    paths = [inputs / "one.wav", inputs / "two.wav"]
    for path in paths:
        sf.write(path, np.zeros(32000, dtype=np.float32), 16000)

    model = object()
    calls = []

    def fake_embedding(audio, sample_rates, *, model, batch_size, **_options):
        calls.append((len(audio), model, batch_size, sample_rates))
        if len(audio) > 1:
            raise MemoryError("synthetic OpenL3 OOM")
        return [np.ones((2, 3), dtype=np.float32)], [np.array([0.0, 1.0])]

    monkeypatch.setitem(
        sys.modules,
        "openl3",
        SimpleNamespace(get_audio_embedding=fake_embedding),
    )
    runtime = OpenL3Runtime(model=model, tensorflow=_TensorFlow(), device="cpu")
    vectorizer = AudioVectorizer(
        inputs,
        outputs,
        tr_s=1.0,
        runtime=runtime,
        embedding_options={
            "stimulus_batch_size": 2,
            "max_stimulus_batch_seconds": 10,
            "inference_batch_size": 16,
            "memory_fallback": True,
        },
    )

    assert vectorizer.compatible_groups(paths) == [paths]
    result = vectorizer.process_paths(paths)

    assert [call[0] for call in calls] == [2, 1, 1]
    assert all(call[1] is model for call in calls)
    assert all(call[2] == 16 for call in calls)
    assert all(result[path][0].is_file() for path in paths)


def test_model_preload_order_places_openl3_after_ctc(monkeypatch):
    pipeline = object.__new__(BatchPipeline)
    order = []
    monkeypatch.setattr(pipeline, "_preload_ctc_model", lambda _names: order.append("ctc"))
    monkeypatch.setattr(
        pipeline, "_preload_openl3_model", lambda _names: order.append("openl3")
    )
    monkeypatch.setattr(
        pipeline,
        "_preload_transcription_models",
        lambda _names: order.append("transcripts"),
    )
    monkeypatch.setattr(
        pipeline, "_preload_sentence_model", lambda _names: order.append("sentence")
    )

    pipeline._preload_models(
        ["acoustic_phonetics", "audio_embeddings", "transcripts", "text_embeddings"]
    )

    assert order == ["ctc", "openl3", "transcripts", "sentence"]
