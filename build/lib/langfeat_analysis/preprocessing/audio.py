"""Audio preprocessing components used by the YAML-driven pipeline.

All paths and extraction settings are supplied by the caller.  In particular,
the number of time frames is derived from each stimulus duration and the TR;
it is never encoded as a dataset-specific constant.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import soundfile as sf


from natural_features.workflows.acoustic_phonetics import (  # noqa: E402
    extract_acoustic_phonetics,
)
from natural_features.features.speech.phonology import CTCModelRuntime  # noqa: E402
from natural_features.workflows.audio_batch import extract_audio_dir  # noqa: E402
from langfeat_analysis.registry import Action, EventRegistry, register_event
from langfeat_analysis.preprocessing.common import align_time_matrix, stimulus_frame_grid
from langfeat_analysis.io import atomic_json_dump, safe_output_path


PIPELINE_NAME = "audio_preproc"
PIPELINE_VERSION = "1.2.0"


@dataclass
class OpenL3Runtime:
    """Loaded OpenL3 model and its independently selected TensorFlow device."""

    model: Any
    tensorflow: Any
    device: str

    def close(self) -> None:
        """Release Keras state after all audio stimuli have been embedded."""
        self.tensorflow.keras.backend.clear_session()


def load_openl3_runtime(
    embedding_options: dict[str, Any] | None = None,
) -> OpenL3Runtime:
    """Load one OpenL3 model using a TensorFlow-specific device policy.

    Args:
        embedding_options: YAML options including model characteristics and
            ``tensorflow_device`` (``auto``, ``gpu``, or ``cpu``). PyTorch's
            CUDA/MPS selection is deliberately not reused here.
    """
    options = dict(embedding_options or {})
    requested = str(options.get("tensorflow_device", "auto")).casefold()
    if requested not in {"auto", "gpu", "cpu"}:
        raise ValueError("tensorflow_device must be auto, gpu, or cpu")

    import tensorflow as tf
    import openl3

    physical_gpus = tf.config.list_physical_devices("GPU")
    if requested == "gpu" and not physical_gpus:
        raise RuntimeError(
            "TensorFlow GPU was requested for OpenL3 but no GPU device is visible. "
            "On Apple Silicon this requires a TensorFlow Metal installation."
        )
    selected = "gpu" if physical_gpus and requested != "cpu" else "cpu"
    try:
        if selected == "cpu":
            tf.config.set_visible_devices([], "GPU")
        else:
            for gpu in physical_gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as error:
        raise RuntimeError(
            "TensorFlow device configuration happened after runtime initialization. "
            "OpenL3 must be preloaded before any TensorFlow inference."
        ) from error

    model = openl3.models.load_audio_embedding_model(
        str(options.get("input_repr", "linear")),
        str(options.get("content_type", "env")),
        int(options.get("embedding_size", 512)),
        frontend=str(options.get("frontend", "kapre")),
    )
    return OpenL3Runtime(model=model, tensorflow=tf, device=selected)


@dataclass
class AudioPreprocessedData:
    """Serializable feature matrix and provenance for one audio stimulus."""

    title: str
    stimulus: str
    method: str
    feature_list: list[str]
    data: list[list[float]]
    pipeline: str
    dimensions: list[int]
    metadata: dict[str, Any]
    create_at: str


def build_tr_aligned_stimulus(
    filename: str,
    method: str,
    data: np.ndarray,
    feature_list: Sequence[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> AudioPreprocessedData:
    """Build one JSON-ready audio feature record.

    Args:
        filename: Stimulus identifier (usually the audio filename stem).
        method: Extraction method represented by the matrix.
        data: Feature-by-time matrix.
        feature_list: Names corresponding to the feature axis.
        metadata: Extractor provenance to preserve in the output.
    """
    array = np.asarray(data)
    title = re.sub("-", "_", f"{method}_{filename}_preproc")
    return AudioPreprocessedData(
        title=title,
        stimulus=filename,
        method=method,
        feature_list=[str(item) for item in (feature_list or [])],
        data=array.tolist(),
        pipeline=f"{PIPELINE_NAME}_{PIPELINE_VERSION}",
        dimensions=list(array.shape),
        metadata=dict(metadata or {}),
        create_at=datetime.now(timezone.utc).isoformat(),
    )


def store_stimuli(
    stimuli_list: Sequence[dict[str, Any]],
    output_dir: str | Path,
    filename_template: str = "{title}.json",
    *,
    registry: EventRegistry | None = None,
    process: str = "audio_preprocessing",
    input_path: str | Path | None = None,
) -> list[Path]:
    """Write feature records to ``output_dir`` and return created paths.

    Args:
        stimuli_list: JSON-ready records containing a unique ``title`` field.
        output_dir: Directory in which output JSON files will be created.
        filename_template: Format string accepting ``title``, ``method``,
            ``feature``, and ``stimulus``.
        registry: Shared run registry. When omitted a compatibility registry is
            created for each event.
    """
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    created: list[Path] = []

    for stimulus in stimuli_list:
        output_name = filename_template.format(
            title=stimulus["title"],
            method=stimulus["method"],
            feature=stimulus["method"],
            stimulus=stimulus["stimulus"],
        )
        output_file = safe_output_path(destination, output_name)
        atomic_json_dump(output_file, stimulus)
        created.append(output_file)
        register_event(
            path=__file__,
            action=Action(name="file_created", status="success"),
            func_name=__name__,
            out_path=str(output_file),
            input_path=str(input_path or stimulus["stimulus"]),
            process=process,
            registry=registry,
        )
    return created


class AudioFeaturePreprocessor:
    """Extract low-level audio features and align them to stimulus frames."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        features: Sequence[str],
        tr_s: float,
        pattern: str = "*.wav",
        filename_template: str = "{title}.json",
        registry: EventRegistry | None = None,
    ) -> None:
        """Configure input/output directories, features, TR, and file pattern."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.features = list(features)
        self.tr_s = float(tr_s)
        self.pattern = pattern
        self.filename_template = filename_template
        self.registry = registry

    def _extract_features(self, file_result: Any) -> list[tuple[str, np.ndarray]]:
        """Split a natural-features result into named feature-by-time matrices."""
        feature_names = list(getattr(file_result, "feature_names", []) or [])
        matrix = np.asarray(getattr(file_result, "matrix", np.empty((0, 0))))
        if not feature_names or matrix.ndim != 2:
            return []

        extracted: list[tuple[str, np.ndarray]] = []
        for feature in self.features:
            indexes = [
                index
                for index, name in enumerate(feature_names)
                if str(name).startswith(f"{feature}.")
            ]
            if indexes:
                extracted.append((feature, matrix[:, indexes].T))
        return extracted

    def process(self) -> list[Path]:
        """Process all matching audio files and return output JSON paths."""
        results = extract_audio_dir(
            directory=self.input_dir,
            pattern=self.pattern,
            selected_features=self.features,
            resolution_s=self.tr_s,
            as_dataframe=False,
        )
        if not results.files:
            raise RuntimeError(f"No audio files matched {self.input_dir / self.pattern}")

        records: list[dict[str, Any]] = []
        for filename, file_result in results.files.items():
            audio_path = Path(file_result.path)
            target_grid = stimulus_frame_grid(audio_path, self.tr_s)
            matrix = align_time_matrix(
                file_result.matrix, file_result.times_s, target_grid
            )
            file_result.matrix = matrix

            for feature, feature_data in self._extract_features(file_result):
                indexes = [
                    i
                    for i, name in enumerate(file_result.feature_names)
                    if str(name).startswith(f"{feature}.")
                ]
                names = [str(file_result.feature_names[i]) for i in indexes]
                metadata = {
                    "extractor_name": "audio.features_to_tr",
                    "source_extractor": "natural_features.workflows.audio_batch",
                    "tr_s": self.tr_s,
                    "frame_count_source": str(audio_path),
                }
                record = build_tr_aligned_stimulus(
                    filename, feature, feature_data, names, metadata
                )
                records.append(asdict(record))
        return store_stimuli(
            records,
            self.output_dir,
            self.filename_template,
            registry=self.registry,
            process="audio_features",
            input_path=next(iter(results.files.values())).path,
        )


class AudioPhonemePreprocessor:
    """Extract posteriorgram and articulatory features from audio stimuli."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        tr_s: float,
        pattern: str = "*.wav",
        extractor_options: dict[str, Any] | None = None,
        ctc_runtime: CTCModelRuntime | None = None,
        progress_callback: Callable[[int, int], None] | None = None,
        filename_template: str = "{title}.json",
        registry: EventRegistry | None = None,
    ) -> None:
        """Configure paths, TR, extractor options, and an optional shared CTC model.

        Args:
            input_dir: Directory containing stimulus WAV files.
            output_dir: Destination directory for feature JSON files.
            tr_s: Output frame duration in seconds.
            pattern: Input filename or glob pattern.
            extractor_options: Acoustic workflow keyword arguments from YAML.
            ctc_runtime: Preloaded model runtime shared across batch items.
            progress_callback: Callback receiving completed and total CTC chunks.
            filename_template: Output filename template accepting ``title``.
            registry: Shared JSONL event registry.
        """
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tr_s = float(tr_s)
        self.pattern = pattern
        self.extractor_options = dict(extractor_options or {})
        self.ctc_runtime = ctc_runtime
        self.progress_callback = progress_callback
        self.filename_template = filename_template
        self.registry = registry

    def process(self) -> list[Path]:
        """Process all matching stimuli and return output JSON paths."""
        paths = sorted(self.input_dir.glob(self.pattern))
        if not paths:
            raise RuntimeError(f"No audio files matched {self.input_dir / self.pattern}")

        records: list[dict[str, Any]] = []
        for path in paths:
            options = dict(self.extractor_options)
            if self.ctc_runtime is not None:
                options["ctc_runtime"] = self.ctc_runtime
            if self.progress_callback is not None:
                options["ctc_progress_callback"] = self.progress_callback
            result = extract_acoustic_phonetics(path, resolution_s=self.tr_s, **options)
            target_grid = stimulus_frame_grid(path, self.tr_s)
            for method, feature_series in (
                ("posteriorgrams", result.posteriorgrams),
                ("articulatory", result.articulatory),
            ):
                aligned = align_time_matrix(
                    feature_series.values, feature_series.times_s, target_grid
                ).T
                metadata = dict(feature_series.metadata)
                metadata.update(
                    {"tr_s": self.tr_s, "frame_count_source": str(path)}
                )
                record = build_tr_aligned_stimulus(
                    path.stem,
                    method,
                    aligned,
                    feature_series.coords.get("feature", []),
                    metadata,
                )
                records.append(asdict(record))
        return store_stimuli(
            records,
            self.output_dir,
            self.filename_template,
            registry=self.registry,
            process="acoustic_phonetics",
            input_path=paths[0],
        )


class AudioVectorizer:
    """Create OpenL3 embeddings aligned to stimulus-derived TR frames."""

    def __init__(
        self,
        input_dir: str | Path,
        output_dir: str | Path,
        tr_s: float,
        pattern: str = "*.wav",
        embedding_options: dict[str, Any] | None = None,
        runtime: OpenL3Runtime | None = None,
        filename_template: str = "{title}.json",
        registry: EventRegistry | None = None,
    ) -> None:
        """Configure paths, TR, OpenL3 options, and an optional shared model."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tr_s = float(tr_s)
        self.pattern = pattern
        self.embedding_options = dict(embedding_options or {})
        self.runtime = runtime
        self.filename_template = filename_template
        self.registry = registry

    def process(self) -> list[Path]:
        """Embed all matching stimuli and return output JSON paths."""
        paths = sorted(self.input_dir.glob(self.pattern))
        if not paths:
            raise RuntimeError(f"No audio files matched {self.input_dir / self.pattern}")
        outputs = self.process_paths(paths)
        return [output for path in paths for output in outputs[path]]

    def compatible_groups(self, paths: Sequence[Path]) -> list[list[Path]]:
        """Group stimuli without exceeding configured file or duration limits."""
        max_files = int(self.embedding_options.get("stimulus_batch_size", 2))
        max_seconds = float(
            self.embedding_options.get("max_stimulus_batch_seconds", 300.0)
        )
        groups: list[list[Path]] = []
        current: list[Path] = []
        current_seconds = 0.0
        for path in paths:
            try:
                info = sf.info(str(path))
                duration = info.frames / float(info.samplerate)
            except RuntimeError:
                # Keep an unreadable stimulus isolated so the runner can report
                # that item accurately without failing otherwise valid files.
                duration = max_seconds + 1.0
            if current and (
                len(current) >= max_files or current_seconds + duration > max_seconds
            ):
                groups.append(current)
                current = []
                current_seconds = 0.0
            current.append(path)
            current_seconds += duration
        if current:
            groups.append(current)
        return groups

    def process_paths(self, paths: Sequence[Path]) -> dict[Path, list[Path]]:
        """Embed explicit stimuli, batching compatible files through one model."""
        if not paths:
            return {}
        owns_runtime = self.runtime is None
        runtime = self.runtime or load_openl3_runtime(self.embedding_options)
        try:
            return self._process_paths_with_runtime(list(paths), runtime)
        finally:
            if owns_runtime:
                runtime.close()

    def _process_paths_with_runtime(
        self, paths: list[Path], runtime: OpenL3Runtime
    ) -> dict[Path, list[Path]]:
        import openl3

        options = {
            "content_type": "env",
            "input_repr": "linear",
            "center": False,
            "embedding_size": 512,
            "verbose": False,
            "frontend": "kapre",
        }
        control_names = {
            "tensorflow_device",
            "stimulus_batch_size",
            "max_stimulus_batch_seconds",
            "inference_batch_size",
            "memory_fallback",
        }
        options.update(
            {
                key: value
                for key, value in self.embedding_options.items()
                if key not in control_names
            }
        )
        options["hop_size"] = self.tr_s
        inference_batch_size = int(
            self.embedding_options.get("inference_batch_size", 16)
        )
        audio_list: list[np.ndarray] = []
        sample_rates: list[int] = []
        for path in paths:
            audio, sample_rate = sf.read(
                str(path), dtype="float32", always_2d=False
            )
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            audio_list.append(audio)
            sample_rates.append(sample_rate)

        try:
            embedding_list, timestamp_list = openl3.get_audio_embedding(
                audio_list,
                sample_rates,
                model=runtime.model,
                batch_size=inference_batch_size,
                **options,
            )
        except Exception as error:
            if not self._is_memory_error(error) or not bool(
                self.embedding_options.get("memory_fallback", True)
            ):
                raise
            if len(paths) > 1:
                outputs: dict[Path, list[Path]] = {}
                self._log_memory_retry(paths, inference_batch_size, error, "one stimulus at a time")
                for path in paths:
                    outputs.update(self._process_paths_with_runtime([path], runtime))
                return outputs
            if inference_batch_size <= 1:
                raise RuntimeError(
                    f"OpenL3 exhausted memory for {paths[0]} even with batch_size=1. "
                    "Use tensorflow_device: cpu or process a shorter stimulus."
                ) from error
            reduced = max(1, inference_batch_size // 2)
            self._log_memory_retry(paths, inference_batch_size, error, f"batch_size={reduced}")
            original = self.embedding_options.get("inference_batch_size")
            self.embedding_options["inference_batch_size"] = reduced
            try:
                return self._process_paths_with_runtime(paths, runtime)
            finally:
                if original is None:
                    self.embedding_options.pop("inference_batch_size", None)
                else:
                    self.embedding_options["inference_batch_size"] = original

        outputs: dict[Path, list[Path]] = {}
        for path, embeddings, timestamps in zip(
            paths, embedding_list, timestamp_list
        ):
            target_grid = stimulus_frame_grid(path, self.tr_s)
            aligned = align_time_matrix(embeddings, timestamps, target_grid).T
            feature_names = [f"openl3_{index}" for index in range(aligned.shape[0])]
            metadata = {
                "extractor_name": "audio.vectorize_to_tr",
                "source_extractor": "openl3",
                "tr_s": self.tr_s,
                "frame_count_source": str(path),
                "options": {
                    **options,
                    "inference_batch_size": inference_batch_size,
                    "tensorflow_device": runtime.device,
                },
            }
            record = build_tr_aligned_stimulus(
                path.stem, "embedding", aligned, feature_names, metadata
            )
            outputs[path] = store_stimuli(
                [asdict(record)],
                self.output_dir,
                self.filename_template,
                registry=self.registry,
                process="audio_embeddings",
                input_path=path,
            )
        return outputs

    @staticmethod
    def _is_memory_error(error: BaseException) -> bool:
        """Recognize Python and TensorFlow accelerator memory failures."""
        message = str(error).casefold()
        return isinstance(error, MemoryError) or type(error).__name__ == "ResourceExhaustedError" or any(
            marker in message for marker in ("out of memory", "oom", "resource exhausted")
        )

    def _log_memory_retry(
        self,
        paths: Sequence[Path],
        batch_size: int,
        error: BaseException,
        retry: str,
    ) -> None:
        """Record an OpenL3 memory fallback in the shared JSONL registry."""
        if self.registry is not None:
            self.registry.emit(
                "inference_retry",
                "warning",
                process="audio_embeddings",
                input_path=paths[0],
                error=error,
                details={
                    "framework": "tensorflow",
                    "batch_size": batch_size,
                    "stimulus_count": len(paths),
                    "retry": retry,
                },
            )


if __name__ == "__main__":
    raise SystemExit(
        "Run `lafa --config configs/preproc.yaml` instead."
    )
