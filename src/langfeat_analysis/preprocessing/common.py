"""Dependency-light stimulus time-grid utilities shared across modalities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import soundfile as sf


def stimulus_frame_grid(audio_path: str | Path, tr_s: float) -> np.ndarray:
    """Return frame starts inferred from the audio sample count and ``tr_s``."""
    if tr_s <= 0:
        raise ValueError(f"tr_s must be greater than zero, got {tr_s}.")
    path = Path(audio_path)
    if not path.is_file():
        raise FileNotFoundError(f"Stimulus audio does not exist: {path}")
    try:
        info = sf.info(str(path))
    except RuntimeError as error:
        raise ValueError(f"Unable to decode stimulus audio {path}: {error}") from error
    if info.frames <= 0 or info.samplerate <= 0:
        raise ValueError(f"Stimulus audio has no usable samples: {path}")
    duration_s = info.frames / float(info.samplerate)
    return np.arange(0.0, duration_s, tr_s, dtype=np.float64)


def align_time_matrix(
    values: np.ndarray,
    source_times_s: Sequence[float],
    target_times_s: Sequence[float],
) -> np.ndarray:
    """Interpolate a ``time x feature`` matrix onto a target time grid."""
    matrix = np.asarray(values)
    source_times = np.asarray(source_times_s, dtype=np.float64)
    target_times = np.asarray(target_times_s, dtype=np.float64)
    if matrix.ndim != 2:
        raise ValueError(
            f"Feature values must be 2-D time x feature; got shape {matrix.shape}."
        )
    if len(matrix) != len(source_times):
        raise ValueError(
            f"Feature rows ({len(matrix)}) do not match timestamps "
            f"({len(source_times)})."
        )
    if len(source_times) == 0:
        raise ValueError("Cannot align an empty feature matrix.")
    if np.any(np.diff(source_times) < 0):
        raise ValueError("Source feature timestamps must be monotonic.")

    aligned = np.empty((len(target_times), matrix.shape[1]), dtype=np.float32)
    for feature_index in range(matrix.shape[1]):
        aligned[:, feature_index] = np.interp(
            target_times,
            source_times,
            matrix[:, feature_index],
            left=matrix[0, feature_index],
            right=matrix[-1, feature_index],
        )
    return aligned
