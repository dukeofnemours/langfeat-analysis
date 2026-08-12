"""Transcript generation and stimulus-aligned text embedding components."""

from __future__ import annotations

import pandas as pd
import logging
import numpy as np
import re
import csv
import platform
import tempfile
import soundfile as sf
import librosa
import torch


from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Any, Literal, Sequence
from datetime import datetime, timezone


from langfeat_analysis.preprocessing.common import stimulus_frame_grid
from langfeat_analysis.registry import Action, EventRegistry, register_event
from langfeat_analysis.io import atomic_json_dump, atomic_text_writer, safe_output_path


PIPELINE_NAME = "text_preproc"
PIPELINE_VERSION = "1.2.0"


Backend = Literal["cuda", "mps", "cpu"]
_SENTENCE_MODEL_CACHE: dict[str, Any] = {}


def preload_sentence_model(model_name: str) -> Any:
    """Load and cache one SentenceTransformer model before item processing."""
    model = _SENTENCE_MODEL_CACHE.get(model_name)
    if model is None:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(model_name)
        _SENTENCE_MODEL_CACHE[model_name] = model
    return model


def unload_sentence_model(model_name: str) -> None:
    """Remove a cached SentenceTransformer and release PyTorch caches."""
    model = _SENTENCE_MODEL_CACHE.pop(model_name, None)
    if model is not None and hasattr(model, "to"):
        model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()



@dataclass
class TextProcessedData:
    title: str
    stimulus: str
    method: str
    feature_list: list[str]
    data: list[list[float]]
    rawdata: list[Any]
    pipeline: str
    dimensions: list[int]
    metadata: dict[str, Any]
    create_at: str


class TranscriptGenerator:
    PYTORCH_ASR_MODEL = "Qwen/Qwen3-ASR-0.6B"
    PYTORCH_ALIGNER_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"

    MLX_ASR_MODEL = "mlx-community/Qwen3-ASR-0.6B-8bit"
    MLX_ALIGNER_MODEL = "mlx-community/Qwen3-ForcedAligner-0.6B-8bit"

    def __init__(
        self,
        language: str = "English",
        max_new_tokens: int = 4096,
        output_path: str | Path | None = None,
        chunk_seconds: float = 270.0,
        target_sample_rate: int = 16000,
        pytorch_asr_model: str | None = None,
        pytorch_aligner_model: str | None = None,
        mlx_asr_model: str | None = None,
        mlx_aligner_model: str | None = None,
        output_filename_template: str = "{stimulus}-annotations.csv",
        registry: EventRegistry | None = None,
    ) -> None:
        if chunk_seconds <= 0:
            raise ValueError("chunk_seconds must be greater than zero.")

        # Keep chunks safely below the aligner's five-minute limit.
        if chunk_seconds >= 300:
            raise ValueError(
                "chunk_seconds must be below 300 seconds for the forced aligner."
            )

        self.language = language
        self.max_new_tokens = max_new_tokens
        self.output_path = Path(output_path) if output_path is not None else None
        self.chunk_seconds = float(chunk_seconds)
        self.target_sample_rate = int(target_sample_rate)
        self.pytorch_asr_model = pytorch_asr_model or self.PYTORCH_ASR_MODEL
        self.pytorch_aligner_model = (
            pytorch_aligner_model or self.PYTORCH_ALIGNER_MODEL
        )
        self.mlx_asr_model = mlx_asr_model or self.MLX_ASR_MODEL
        self.mlx_aligner_model = mlx_aligner_model or self.MLX_ALIGNER_MODEL
        self.output_filename_template = output_filename_template
        self.registry = registry

        self.backend: Backend = self._detect_backend()

        self._asr_model: Any | None = None
        self._aligner_model: Any | None = None

        logging.info("Selected transcription backend: %s", self.backend)

    @staticmethod
    def _detect_backend() -> Backend:
        if torch.cuda.is_available():
            return "cuda"

        # MLX requires Apple Silicon, not merely an MPS-capable configuration.
        is_apple_silicon = (
            platform.system() == "Darwin"
            and platform.machine() in {"arm64", "aarch64"}
        )

        if is_apple_silicon and torch.backends.mps.is_available():
            return "mps"

        return "cpu"

   
    def _load_audio(
        self,
        audio_path: Path,
        target_sr: int | None = None,
    ) -> tuple[np.ndarray, int]:
        target_sr = target_sr or self.target_sample_rate

        audio, sr = sf.read(
            str(audio_path),
            dtype="float32",
            always_2d=False,
        )

        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # collapse to mono

        if sr != target_sr:
            audio = librosa.resample(
                audio,
                orig_sr=sr,
                target_sr=target_sr,
            )
            sr = target_sr

        return np.asarray(audio, dtype=np.float32), sr

    def _iter_audio_chunks(
        self,
        audio: np.ndarray,
        sr: int,
    ):
        """Yield `(index, start_seconds, chunk_audio)` tuples."""
        samples_per_chunk = max(1, int(round(self.chunk_seconds * sr)))

        for chunk_index, start_sample in enumerate(
            range(0, len(audio), samples_per_chunk)
        ):
            end_sample = min(start_sample + samples_per_chunk, len(audio))
            chunk_audio = audio[start_sample:end_sample]

            if chunk_audio.size == 0:
                continue

            yield chunk_index, start_sample / sr, chunk_audio

    @staticmethod
    def _offset_words(
        aligned_items: Any,
        offset_seconds: float,
        chunk_index: int,
    ) -> list[dict[str, Any]]:
        words = []

        for item in aligned_items:
            start_time = float(item.start_time) + offset_seconds
            end_time = float(item.end_time) + offset_seconds

            words.append(
                {
                    "word": str(item.text),
                    "onset": start_time,
                    "offset": end_time,
                    "chunk_index": chunk_index,
                }
            )

        return words

    @staticmethod
    def _validate_chunk_alignment(
        words: list[dict[str, Any]],
        chunk_duration: float,
        chunk_offset: float,
        *,
        tolerance: float = 0.20,
        max_terminal_words: int = 3,
    ) -> None:
        """Detect the repeated-final-timestamp failure mode."""
        if not words:
            raise RuntimeError(
                f"Forced aligner returned no words for chunk at "
                f"{chunk_offset:.3f}s."
            )

        local_end = chunk_offset + chunk_duration
        terminal_words = [
            word
            for word in words
            if word["onset"] >= local_end - tolerance
            and word["offset"] >= local_end - tolerance
        ]

        if len(terminal_words) > max_terminal_words:
            preview = " ".join(
                word["word"] for word in terminal_words[:15]
            )
            raise RuntimeError(
                "Forced alignment saturated at the end of a chunk. "
                f"{len(terminal_words)} words were assigned near "
                f"{local_end:.3f}s. First affected words: {preview!r}"
            )

    def _load_models(self) -> None:
        """Load and cache the ASR and forced-alignment models."""
        if self._asr_model is not None and self._aligner_model is not None:
            return

        logging.info("Initializing transcription and alignment models.")

        if self.backend == "mps":
            self._load_mlx_models()
        else:
            self._load_pytorch_models()

    def preload_models(self) -> str:
        """Load ASR and alignment models once and return the selected backend."""
        self._load_models()
        return self.backend

    def _load_mlx_models(self) -> None:
        """
        Load quantized MLX models.

        Imports are local so Linux/Windows installations do not need MLX.
        """
        try:
            from mlx_audio.stt import load
        except ImportError as exc:
            raise RuntimeError(
                "Apple Silicon support requires mlx-audio. Install it with:\n"
                "    pip install -U mlx-audio"
            ) from exc

        self._asr_model = load(self.mlx_asr_model)
        self._aligner_model = load(self.mlx_aligner_model)

    def _load_pytorch_models(self) -> None:
        try:
            from qwen_asr import Qwen3ASRModel, Qwen3ForcedAligner
        except ImportError as exc:
            raise RuntimeError(
                "CUDA and CPU support requires qwen-asr. Install it with:\n"
                "    pip install -U qwen-asr"
            ) from exc

        if self.backend == "cuda":
            # BF16 is preferable when the GPU supports it. FP16 is the fallback.
            dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
            device_map = "cuda:0"
        else:
            dtype = torch.float32
            device_map = "cpu"

        common_options = {
            "dtype": dtype,
            "device_map": device_map,
        }

        self._asr_model = Qwen3ASRModel.from_pretrained(
            self.pytorch_asr_model,
            max_inference_batch_size=1,
            max_new_tokens=self.max_new_tokens,
            **common_options,
        )

        self._aligner_model = Qwen3ForcedAligner.from_pretrained(
            self.pytorch_aligner_model,
            **common_options,
        )

    def transcribe_audio(
        self,
        path: str | Path,
        output_path: str | Path | None = None,
    ) -> str:
        audio_path = Path(path).expanduser().resolve()

        if not audio_path.is_file():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        self._load_models()

        logging.info("Starting transcription for %s.", audio_path)

        if self.backend == "mps":
            result = self._transcribe_with_mlx(audio_path)
        else:
            result = self._transcribe_with_pytorch(audio_path)

        destination_dir = (
            Path(output_path) if output_path is not None else self.output_path
        )
        if destination_dir is None:
            raise ValueError("An output directory is required for transcription.")
        destination = safe_output_path(
            destination_dir,
            self.output_filename_template.format(stimulus=audio_path.stem),
        )

        headers=["word", "onset", "offset", "chunk_index"]
        data=result['words'] if result['words'] is not None else result.get("words")

        with atomic_text_writer(destination, newline="") as file:
            writer = csv.DictWriter(file, fieldnames=headers)

            writer.writeheader()

            writer.writerows(data)

        register_event(
            path=__file__,
            action=Action(name="file_created", status="success"),
            func_name=__name__,
            out_path=str(destination),
            input_path=str(audio_path),
            process="transcripts",
            registry=self.registry,
        )
        return str(destination)

    def _transcribe_with_mlx(self, audio_path: Path) -> dict[str, Any]:
        """
        Normalize the source once, write temporary WAV chunks, and run ASR plus
        forced alignment on each chunk. Timestamps are shifted back into the
        full recording timeline.
        """
        logging.info("Transcribing with MLX.")

        audio, sr = self._load_audio(audio_path)
        duration = len(audio) / sr

        logging.info(
            "Normalized sample rate=%d, audio duration=%.3fs, chunk duration=%.3fs.",
            sr,
            duration,
            self.chunk_seconds,
        )

        all_text: list[str] = []
        all_words: list[dict[str, Any]] = []
        detected_language = self.language

        with tempfile.TemporaryDirectory(
            prefix="qwen_audio_chunks_"
        ) as temp_dir:
            temp_dir_path = Path(temp_dir)

            for chunk_index, chunk_offset, chunk_audio in self._iter_audio_chunks(
                audio,
                sr,
            ):
                chunk_duration = len(chunk_audio) / sr
                chunk_path = temp_dir_path / f"chunk_{chunk_index:04d}.wav"

                # PCM_16 gives both models the same stable, broadly supported
                # representation.
                sf.write(
                    str(chunk_path),
                    chunk_audio,
                    sr,
                    subtype="PCM_16",
                )

                logging.info(
                    f"Processing chunk {chunk_index + 1}: "
                    f"{chunk_offset:.3f}s–"
                    f"{chunk_offset + chunk_duration:.3f}s"
                )

                transcription = self._asr_model.generate(
                    str(chunk_path),
                    language=self.language,
                    max_tokens=self.max_new_tokens,
                )

                chunk_text = self._extract_text(transcription).strip()

                if not chunk_text:
                    logging.info(
                        f"Chunk {chunk_index + 1} returned no speech; skipping."
                    )
                    continue

                detected_language = self._extract_language(
                    transcription,
                    default=detected_language,
                )

                alignment = self._aligner_model.generate(
                    str(chunk_path),
                    text=chunk_text,
                    language=detected_language,
                )

                chunk_words = self._offset_words(
                    alignment,
                    offset_seconds=chunk_offset,
                    chunk_index=chunk_index,
                )

                self._validate_chunk_alignment(
                    chunk_words,
                    chunk_duration=chunk_duration,
                    chunk_offset=chunk_offset,
                )

                all_text.append(chunk_text)
                all_words.extend(chunk_words)

        if not all_text:
            raise RuntimeError("MLX ASR returned an empty transcript.")

        text = " ".join(all_text).strip()



        return {
            "backend": "mlx",
            "device": "mps",
            "language": detected_language,
            "text": text,
            "words": all_words,
            "audio_duration": duration,
            "chunk_seconds": self.chunk_seconds,
            "chunk_count": int(
                np.ceil(duration / self.chunk_seconds)
            ) if duration > 0 else 0,
        }

    def _transcribe_with_pytorch(
        self,
        audio_path: Path,
    ) -> dict[str, Any]:
        """
        Normalize the source once, write temporary WAV chunks, and process each
        chunk independently so the forced aligner never receives five minutes
        or more of audio.
        """
        logging.info("Transcribing with PyTorch on %s.", self.backend)

        audio, sr = self._load_audio(audio_path)
        duration = len(audio) / sr

        logging.info(
            "Normalized sample rate=%d, audio duration=%.3fs, chunk duration=%.3fs.",
            sr,
            duration,
            self.chunk_seconds,
        )

        all_text: list[str] = []
        all_words: list[dict[str, Any]] = []
        detected_language = self.language

        with tempfile.TemporaryDirectory(
            prefix="qwen_audio_chunks_"
        ) as temp_dir:
            temp_dir_path = Path(temp_dir)

            for chunk_index, chunk_offset, chunk_audio in self._iter_audio_chunks(
                audio,
                sr,
            ):
                chunk_duration = len(chunk_audio) / sr
                chunk_path = temp_dir_path / f"chunk_{chunk_index:04d}.wav"

                sf.write(
                    str(chunk_path),
                    chunk_audio,
                    sr,
                    subtype="PCM_16",
                )

                logging.info(
                    f"Processing chunk {chunk_index + 1}: "
                    f"{chunk_offset:.3f}s–"
                    f"{chunk_offset + chunk_duration:.3f}s"
                )

                # Preserve qwen-asr's array input contract, but decode the
                # exact temporary WAV used by the aligner.
                normalized_chunk, normalized_sr = sf.read(
                    str(chunk_path),
                    dtype="float32",
                    always_2d=False,
                )
                if normalized_sr != sr:
                    raise RuntimeError(
                        f"Temporary chunk sample rate changed from {sr} "
                        f"to {normalized_sr}."
                    )

                transcription_results = self._asr_model.transcribe(
                    audio=normalized_chunk,
                    language=self.language,
                )

                if not transcription_results:
                    logging.info(
                        f"Chunk {chunk_index + 1} returned no speech; skipping."
                    )
                    continue

                transcription = transcription_results[0]
                chunk_text = self._extract_text(transcription).strip()

                if not chunk_text:
                    logging.info(
                        f"Chunk {chunk_index + 1} returned empty text; skipping."
                    )
                    continue

                detected_language = self._extract_language(
                    transcription,
                    default=detected_language,
                )

                alignment_results = self._aligner_model.align(
                    audio=str(chunk_path),
                    text=chunk_text,
                    language=detected_language,
                )

                if not alignment_results:
                    raise RuntimeError(
                        f"Qwen forced aligner returned no result for chunk "
                        f"{chunk_index + 1}."
                    )

                chunk_words = self._offset_words(
                    alignment_results[0],
                    offset_seconds=chunk_offset,
                    chunk_index=chunk_index,
                )

                self._validate_chunk_alignment(
                    chunk_words,
                    chunk_duration=chunk_duration,
                    chunk_offset=chunk_offset,
                )

                all_text.append(chunk_text)
                all_words.extend(chunk_words)

        if not all_text:
            raise RuntimeError("Qwen ASR returned an empty transcript.")

        text = " ".join(all_text).strip()

        return {
            "backend": "qwen-asr",
            "device": self.backend,
            "language": detected_language,
            "text": text,
            "words": all_words,
            "audio_duration": duration,
            "chunk_seconds": self.chunk_seconds,
            "chunk_count": int(
                np.ceil(duration / self.chunk_seconds)
            ) if duration > 0 else 0,
        }

    @staticmethod
    def _extract_text(result: Any) -> str:
        """
        Normalize result text across qwen-asr and mlx-audio versions.
        """
        if isinstance(result, str):
            return result.strip()

        if isinstance(result, dict):
            value = result.get("text") or result.get("transcription")
            return str(value or "").strip()

        value = getattr(result, "text", None)

        if value is None:
            value = getattr(result, "transcription", None)

        return str(value or "").strip()

    @staticmethod
    def _extract_language(result: Any, default: str) -> str:
        if isinstance(result, dict):
            return str(result.get("language") or default)

        return str(getattr(result, "language", None) or default)

    def unload_models(self) -> None:
        """
        Explicitly release model references and clear accelerator caches.
        """
        self._asr_model = None
        self._aligner_model = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if self.backend == "mps":
            try:
                import mlx.core as mx

                mx.clear_cache()
            except (ImportError, AttributeError):
                pass


class TextEmbedderGrid:
    """Convert word annotations into contextual embeddings on an audio TR grid."""

    def __init__(
        self,
        annotation_path: str | Path,
        stimulus_audio: str | Path,
        output_dir: str | Path,
        tr_s: float,
        model_name: str,
        context_window: int = 3,
        output_filename_template: str = "{title}.json",
        strict_annotations: bool = False,
        registry: EventRegistry | None = None,
    ) -> None:
        """Configure annotation/audio inputs and embedding output settings.

        Args:
            annotation_path: CSV, TSV, or Praat TextGrid word annotations.
            stimulus_audio: Audio stimulus used to infer the exact frame count.
            output_dir: Directory for the embedding JSON file.
            tr_s: Frame interval in seconds.
            model_name: SentenceTransformer model identifier or local path.
            context_window: Number of consecutive frame phrases per embedding.
            output_filename_template: Format string accepting ``title``,
                ``annotation``, and ``stimulus``.
            strict_annotations: Raise on invalid intervals when true; otherwise
                clamp a bad offset to its onset and log a warning.
        """
        if context_window <= 0:
            raise ValueError("context_window must be greater than zero.")
        self.annotation_path = Path(annotation_path)
        self.stimulus_audio = Path(stimulus_audio)
        self.output_dir = Path(output_dir)
        self.tr_s = float(tr_s)
        self.model_name = model_name
        self.context_window = int(context_window)
        self.output_filename_template = output_filename_template
        self.strict_annotations = bool(strict_annotations)
        self.registry = registry
        self.frame_grid = stimulus_frame_grid(self.stimulus_audio, self.tr_s)

    @staticmethod
    def _column(df: pd.DataFrame, target: str) -> str:
        """Find an annotation column by normalized exact name, then substring."""
        normalized = {str(column).strip().casefold(): str(column) for column in df}
        aliases = {
            "word": ("word", "words", "token", "tokens"),
            "onset": ("onset", "word onset", "word_onset"),
            "offset": ("offset", "word offset", "word_offset"),
        }
        for alias in aliases[target]:
            if alias in normalized:
                return normalized[alias]
        candidates = [original for name, original in normalized.items() if target in name]
        if target == "word":
            candidates = [
                name
                for name in candidates
                if "onset" not in name.casefold() and "offset" not in name.casefold()
            ]
        if len(candidates) != 1:
            raise ValueError(
                f"Expected one {target!r} column; found {candidates or 'none'}."
            )
        return candidates[0]

    def _scan_file(self) -> list[list[Any]]:
        """Read word, onset, and offset lists from the configured annotation."""
        extension = self.annotation_path.suffix.lower()
        if extension in {".csv", ".tsv"}:
            delimiter = "," if extension == ".csv" else "\t"
            dataframe = pd.read_csv(self.annotation_path, delimiter=delimiter)
            word_column = self._column(dataframe, "word")
            onset_column = self._column(dataframe, "onset")
            offset_column = self._column(dataframe, "offset")
            words = dataframe[word_column].fillna("").astype(str).tolist()
            onsets = pd.to_numeric(dataframe[onset_column], errors="raise").tolist()
            offsets = pd.to_numeric(dataframe[offset_column], errors="raise").tolist()
            return self._validate_annotations(words, onsets, offsets)
        if extension == ".textgrid":
            return self._read_textgrid()
        raise ValueError(
            f"Unsupported annotation type {extension!r}; use CSV, TSV, or TextGrid."
        )

    def _validate_annotations(
        self,
        words: Sequence[str], onsets: Sequence[float], offsets: Sequence[float]
    ) -> list[list[Any]]:
        """Validate parallel annotation columns and return JSON-safe lists."""
        if not (len(words) == len(onsets) == len(offsets)) or not words:
            raise ValueError("Annotation columns must be non-empty and equal length.")
        clean_words = [str(word).strip() for word in words]
        clean_onsets = [float(value) for value in onsets]
        clean_offsets = [float(value) for value in offsets]
        invalid = [
            index
            for index, (onset, offset) in enumerate(zip(clean_onsets, clean_offsets))
            if offset < onset
        ]
        if invalid and self.strict_annotations:
            raise ValueError(
                f"Annotation offsets occur before onsets at rows: {invalid[:10]}"
            )
        for index in invalid:
            logging.warning(
                "Clamping annotation offset to onset at row %d in %s.",
                index,
                self.annotation_path,
            )
            if self.registry is not None:
                self.registry.emit(
                    "annotation_clamped",
                    "warning",
                    process="text_embeddings",
                    input_path=self.annotation_path,
                    details={
                        "row_index": index,
                        "onset": clean_onsets[index],
                        "invalid_offset": clean_offsets[index],
                    },
                )
            clean_offsets[index] = clean_onsets[index]
        if any(b < a for a, b in zip(clean_onsets, clean_onsets[1:])):
            raise ValueError("Annotation onsets must be monotonically increasing.")
        return [clean_words, clean_onsets, clean_offsets]

    def _read_textgrid(self) -> list[list[Any]]:
        """Read the ``text words`` tier, or first IntervalTier, from TextGrid."""
        raw_content = self.annotation_path.read_bytes()
        # Praat can export long-text TextGrids as UTF-16.  ``utf-16`` consumes
        # either byte-order mark; UTF-8 remains the default for ordinary grids.
        encoding = (
            "utf-16"
            if raw_content.startswith((b"\xff\xfe", b"\xfe\xff"))
            else "utf-8-sig"
        )
        content = raw_content.decode(encoding)
        if 'Object class = "TextGrid"' not in content:
            raise ValueError(f"Not a Praat TextGrid file: {self.annotation_path}")

        tier_pattern = re.compile(
            r"(?ms)^\s*item\s*\[\d+\]:\s*(?P<body>.*?)"
            r"(?=^\s*item\s*\[\d+\]:|\Z)"
        )
        interval_pattern = re.compile(
            r"(?ms)^\s*intervals\s*\[\d+\]:\s*"
            r"\s*xmin\s*=\s*(?P<onset>[-+0-9.eE]+)\s*"
            r"\s*xmax\s*=\s*(?P<offset>[-+0-9.eE]+)\s*"
            r'\s*text\s*=\s*"(?P<word>(?:""|[^"])*)"'
        )
        tiers: list[tuple[str, str]] = []
        for match in tier_pattern.finditer(content):
            body = match.group("body")
            if not re.search(r'^\s*class\s*=\s*"IntervalTier"', body, re.M):
                continue
            name = re.search(r'^\s*name\s*=\s*"(?P<name>[^"]*)"', body, re.M)
            tiers.append((name.group("name") if name else "", body))
        if not tiers:
            raise ValueError(f"No IntervalTier found in {self.annotation_path}")

        _, selected = next(
            (tier for tier in tiers if tier[0].strip().casefold() == "text words"),
            tiers[0],
        )
        words: list[str] = []
        onsets: list[float] = []
        offsets: list[float] = []
        for interval in interval_pattern.finditer(selected):
            word = interval.group("word").replace('""', '"').strip()
            if word:
                words.append(word)
                onsets.append(float(interval.group("onset")))
                offsets.append(float(interval.group("offset")))
        return self._validate_annotations(words, onsets, offsets)

    def _resample_annotations(self, annotations: list[list[Any]]) -> list[list[Any]]:
        """Bin words by onset into frames inferred from the stimulus audio."""
        words, onsets, _ = annotations
        phrases: list[str] = []
        audio_info = sf.info(str(self.stimulus_audio))
        audio_duration = audio_info.frames / float(audio_info.samplerate)
        frame_offsets = np.minimum(self.frame_grid + self.tr_s, audio_duration)
        for index, (start, end) in enumerate(zip(self.frame_grid, frame_offsets)):
            selected = [
                words[word_index]
                for word_index, onset in enumerate(onsets)
                if onset >= start
                and (onset < end or (index == len(self.frame_grid) - 1 and onset <= end))
            ]
            phrases.append(" ".join(filter(None, selected)))
        return [phrases, self.frame_grid.tolist(), frame_offsets.tolist()]

    def _contextual_embeddings(self, phrases: Sequence[str]) -> np.ndarray:
        """Embed rolling windows of frame-level phrases as feature-by-time data."""
        sentences = [
            " ".join(phrases[max(0, i - self.context_window + 1) : i + 1]).strip()
            for i in range(len(phrases))
        ]
        model = preload_sentence_model(self.model_name)
        return np.asarray(model.encode(sentences)).T

    def _build_record(
        self, embeddings: np.ndarray, resampled: list[list[Any]]
    ) -> TextProcessedData:
        """Build a JSON-ready text embedding record with provenance metadata."""
        filename = self.annotation_path.stem
        title = re.sub("-", "_", f"embedding_{filename}_preproc")
        metadata = {
            "extractor_name": "text.embeddings_to_tr",
            "source_extractor": "langfeat_analysis.preprocessing.text",
            "model": self.model_name,
            "context_window": self.context_window,
            "strict_annotations": self.strict_annotations,
            "tr_s": self.tr_s,
            "frame_count_source": str(self.stimulus_audio),
        }
        return TextProcessedData(
            title=title,
            stimulus=self.stimulus_audio.stem,
            method="embedding",
            feature_list=[f"embedding_{i}" for i in range(embeddings.shape[0])],
            data=embeddings.tolist(),
            rawdata=resampled,
            pipeline=f"{PIPELINE_NAME}_{PIPELINE_VERSION}",
            dimensions=list(embeddings.shape),
            metadata=metadata,
            create_at=datetime.now(timezone.utc).isoformat(),
        )

    def process(self) -> Path:
        """Create one stimulus-aligned text embedding JSON file."""
        annotations = self._scan_file()
        resampled = self._resample_annotations(annotations)
        embeddings = self._contextual_embeddings(resampled[0])
        record = asdict(self._build_record(embeddings, resampled))

        self.output_dir.mkdir(parents=True, exist_ok=True)
        output_name = self.output_filename_template.format(
            title=record["title"],
            annotation=self.annotation_path.stem,
            stimulus=self.stimulus_audio.stem,
        )
        output_file = safe_output_path(self.output_dir, output_name)
        atomic_json_dump(output_file, record)
        register_event(
            path=__file__,
            action=Action(name="file_created", status="success"),
            func_name=__name__,
            out_path=str(output_file),
            input_path=str(self.annotation_path),
            process="text_embeddings",
            registry=self.registry,
        )
        return output_file


# Backward-compatible alias for the original misspelled public class name.
TextEmbbederGrid = TextEmbedderGrid


class TranscriptClassifier:
    """Route homogeneous input paths through transcription or text embedding."""

    def __init__(
        self,
        paths: Sequence[str | Path],
        output_dir: str | Path,
        *,
        stimulus_audio: str | Path | None = None,
        tr_s: float = 1.0,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        transcript_options: dict[str, Any] | None = None,
    ) -> None:
        """Configure routed inputs plus all output and alignment settings."""
        self.paths = [Path(path) for path in paths]
        self.output_dir = Path(output_dir)
        self.stimulus_audio = Path(stimulus_audio) if stimulus_audio else None
        self.tr_s = float(tr_s)
        self.embedding_model = embedding_model
        self.transcript_options = dict(transcript_options or {})

    def classify(self) -> list[Path]:
        """Run the process selected by the common input file extension."""
        file_types = {path.suffix.lower() for path in self.paths}
        if file_types == {".wav"}:
            transcriber = TranscriptGenerator(
                output_path=self.output_dir, **self.transcript_options
            )
            annotations = [
                Path(transcriber.transcribe_audio(path)) for path in self.paths
            ]
            return [
                TextEmbedderGrid(
                    annotation,
                    audio,
                    self.output_dir,
                    self.tr_s,
                    self.embedding_model,
                ).process()
                for annotation, audio in zip(annotations, self.paths)
            ]
        if file_types <= {".csv", ".tsv", ".textgrid"} and len(file_types) == 1:
            if self.stimulus_audio is None or len(self.paths) != 1:
                raise ValueError(
                    "A single stimulus_audio is required for annotation input."
                )
            return [
                TextEmbedderGrid(
                    self.paths[0],
                    self.stimulus_audio,
                    self.output_dir,
                    self.tr_s,
                    self.embedding_model,
                ).process()
            ]
        raise ValueError("Inputs must use one supported, homogeneous file type.")


if __name__ == "__main__":
    raise SystemExit(
        "Run `lafa --config configs/preproc.yaml` instead."
    )
