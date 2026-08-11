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
from typing import Any, Sequence

import numpy as np
import soundfile as sf


from natural_features.workflows.acoustic_phonetics import (  # noqa: E402
    extract_acoustic_phonetics,
)
from natural_features.workflows.audio_batch import extract_audio_dir  # noqa: E402
from langfeat_analysis.registry import Action, EventRegistry, register_event
from langfeat_analysis.preprocessing.common import align_time_matrix, stimulus_frame_grid
from langfeat_analysis.io import atomic_json_dump, safe_output_path


PIPELINE_NAME = "audio_preproc"
PIPELINE_VERSION = "1.2.0"


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
        filename_template: str = "{title}.json",
        registry: EventRegistry | None = None,
    ) -> None:
        """Configure paths, TR, glob pattern, and acoustic extractor options."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tr_s = float(tr_s)
        self.pattern = pattern
        self.extractor_options = dict(extractor_options or {})
        self.filename_template = filename_template
        self.registry = registry

    def process(self) -> list[Path]:
        """Process all matching stimuli and return output JSON paths."""
        paths = sorted(self.input_dir.glob(self.pattern))
        if not paths:
            raise RuntimeError(f"No audio files matched {self.input_dir / self.pattern}")

        records: list[dict[str, Any]] = []
        for path in paths:
            result = extract_acoustic_phonetics(
                path, resolution_s=self.tr_s, **self.extractor_options
            )
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
        filename_template: str = "{title}.json",
        registry: EventRegistry | None = None,
    ) -> None:
        """Configure paths, TR, file pattern, and OpenL3 keyword options."""
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.tr_s = float(tr_s)
        self.pattern = pattern
        self.embedding_options = dict(embedding_options or {})
        self.filename_template = filename_template
        self.registry = registry

    def process(self) -> list[Path]:
        """Embed all matching stimuli and return output JSON paths."""
        import openl3  # Expensive dependency; load only for this process.

        paths = sorted(self.input_dir.glob(self.pattern))
        if not paths:
            raise RuntimeError(f"No audio files matched {self.input_dir / self.pattern}")

        options = {
            "content_type": "env",
            "input_repr": "linear",
            "center": False,
            "embedding_size": 512,
            "verbose": False,
        }
        options.update(self.embedding_options)
        options["hop_size"] = self.tr_s

        records: list[dict[str, Any]] = []
        for path in paths:
            audio, sample_rate = sf.read(str(path), always_2d=False)
            if audio.ndim > 1:
                audio = audio.mean(axis=1)
            embeddings, timestamps = openl3.get_audio_embedding(
                audio, sample_rate, **options
            )
            target_grid = stimulus_frame_grid(path, self.tr_s)
            aligned = align_time_matrix(embeddings, timestamps, target_grid).T
            feature_names = [f"openl3_{index}" for index in range(aligned.shape[0])]
            metadata = {
                "extractor_name": "audio.vectorize_to_tr",
                "source_extractor": "openl3",
                "tr_s": self.tr_s,
                "frame_count_source": str(path),
                "options": options,
            }
            record = build_tr_aligned_stimulus(
                path.stem, "embedding", aligned, feature_names, metadata
            )
            records.append(asdict(record))
        return store_stimuli(
            records,
            self.output_dir,
            self.filename_template,
            registry=self.registry,
            process="audio_embeddings",
            input_path=paths[0],
        )


if __name__ == "__main__":
    raise SystemExit(
        "Run `lafa --config configs/preproc.yaml` instead."
    )
