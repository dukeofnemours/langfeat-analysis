"""Batch-safe orchestration for all preprocessing processes."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Callable, Sequence

from langfeat_analysis.registry import EventRegistry

from .config import discover_files, resolve_path, selected_processes
from .errors import InputError, ProcessError
from .progress import TerminalReporter


AUDIO_PROCESSES = {"audio_features", "acoustic_phonetics", "audio_embeddings"}


def collector_match_keys(
    stem: str, modality: str, *, case_sensitive: bool = False
) -> set[str]:
    """Return pairing keys for names produced by ``collect-stimuli.sh``.

    Collector names end in a path checksum and include ``_audio_`` or
    ``_text_``. Multiple keys are returned when those marker words also occur
    in an alias or source stem; the caller must require a unique file match.
    """
    without_checksum = re.sub(r"_\d+$", "", stem)
    searchable = without_checksum if case_sensitive else without_checksum.casefold()
    marker = f"_{modality}_"
    keys: set[str] = set()
    start = 0
    while True:
        index = searchable.find(marker, start)
        if index < 0:
            break
        key = without_checksum[:index] + "\0" + without_checksum[index + len(marker) :]
        keys.add(key if case_sensitive else key.casefold())
        start = index + 1
    return keys


@dataclass
class ItemResult:
    """Outcome for one input item within a batch process."""

    process: str
    input_path: str
    status: str
    output_paths: list[str] = field(default_factory=list)
    error_type: str = ""
    error_message: str = ""


@dataclass
class ProcessResult:
    """Aggregated item outcomes for one process."""

    process: str
    status: str
    items: list[ItemResult] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(item.status == "success" for item in self.items)

    @property
    def failed(self) -> int:
        return sum(item.status == "failed" for item in self.items)

    @property
    def skipped(self) -> int:
        return sum(item.status == "skipped" for item in self.items)


@dataclass(frozen=True)
class TranscriptMatch:
    """Validated best filename match between an audio and existing text file."""

    audio: Path
    transcript: Path
    score: float
    method: str = "similarity_threshold"


def normalized_stimulus_name(stem: str) -> tuple[str, set[str], set[str]]:
    """Normalize collector markers, checksums, suffixes, and token boundaries."""
    value = stem.casefold()
    value = re.sub(r"[_-]\d+$", "", value)
    value = re.sub(r"(^|[_-])(audio|text)(?=[_-])", r"\1", value)
    value = re.sub(
        r"([_-](annotations?|transcripts?|transcriptions?|words?))+$", "", value
    )
    value = re.sub(r"(?<=[a-z])(?=\d)|(?<=\d)(?=[a-z])", " ", value)
    tokens = set(re.findall(r"[a-z]+|\d+", value))
    numbers = {token for token in tokens if token.isdigit()}
    return "".join(re.findall(r"[a-z0-9]+", value)), tokens, numbers


def transcript_filename_score(audio: Path, transcript: Path) -> float:
    """Score filename similarity in [0, 1], penalizing conflicting numbers."""
    audio_compact, audio_tokens, audio_numbers = normalized_stimulus_name(audio.stem)
    text_compact, text_tokens, text_numbers = normalized_stimulus_name(transcript.stem)
    if not audio_compact or not text_compact:
        return 0.0
    compact_score = SequenceMatcher(None, audio_compact, text_compact).ratio()
    union = audio_tokens | text_tokens
    token_score = len(audio_tokens & text_tokens) / len(union) if union else 0.0
    score = 0.75 * compact_score + 0.25 * token_score
    if audio_numbers and text_numbers and audio_numbers != text_numbers:
        score *= 0.5
    return round(float(score), 6)


def _transcript_extension_priority(path: Path) -> int:
    """Prefer structured annotations when equal-name transcript files coexist."""
    return {
        ".csv": 5,
        ".tsv": 4,
        ".textgrid": 3,
        ".eaf": 2,
        ".txt": 1,
    }.get(path.suffix.casefold(), 0)


def filename_alias(path: Path) -> str:
    """Return the case-insensitive prefix before the first underscore."""
    stem = path.stem.strip().casefold()
    return stem.split("_", maxsplit=1)[0] if "_" in stem else ""


def _best_one_to_one_assignment(
    audio_paths: Sequence[Path], transcript_paths: Sequence[Path]
) -> list[tuple[float, Path, Path]]:
    """Maximize total filename similarity for one equal-cardinality alias group."""
    audio = sorted(audio_paths)
    transcripts = sorted(transcript_paths)
    if len(audio) > 12:
        # Avoid exponential memory for unusually large aliases while retaining
        # deterministic, score-first one-to-one behavior.
        edges = sorted(
            (
                (transcript_filename_score(a, t), a, t)
                for a in audio
                for t in transcripts
            ),
            reverse=True,
            key=lambda item: (item[0], str(item[1]), str(item[2])),
        )
        selected: list[tuple[float, Path, Path]] = []
        used_audio: set[Path] = set()
        used_text: set[Path] = set()
        for edge in edges:
            if edge[1] not in used_audio and edge[2] not in used_text:
                selected.append(edge)
                used_audio.add(edge[1])
                used_text.add(edge[2])
        return selected

    states: dict[int, tuple[float, list[tuple[float, Path, Path]]]] = {0: (0.0, [])}
    for current_audio in audio:
        next_states: dict[int, tuple[float, list[tuple[float, Path, Path]]]] = {}
        for mask, (total, pairs) in states.items():
            for index, transcript in enumerate(transcripts):
                bit = 1 << index
                if mask & bit:
                    continue
                score = transcript_filename_score(current_audio, transcript)
                candidate = (total + score, pairs + [(score, current_audio, transcript)])
                existing = next_states.get(mask | bit)
                if existing is None or candidate[0] > existing[0]:
                    next_states[mask | bit] = candidate
        states = next_states
    return max(states.values(), key=lambda item: item[0])[1] if states else []


def select_transcript_matches(
    audio_paths: Sequence[Path],
    transcript_paths: Sequence[Path],
    *,
    threshold: float = 0.72,
    min_margin: float = 0.05,
    alias_count_heuristic: bool = True,
    alias_min_score: float = 0.35,
) -> tuple[dict[Path, TranscriptMatch], list[dict[str, Any]]]:
    """Select unique transcript matches using similarity, then alias cardinality."""
    candidates: list[tuple[float, Path, Path]] = []
    ambiguities: list[dict[str, Any]] = []
    for audio in audio_paths:
        ranked = sorted(
            (
                (transcript_filename_score(audio, transcript), transcript)
                for transcript in transcript_paths
            ),
            reverse=True,
            key=lambda item: (
                item[0],
                _transcript_extension_priority(item[1]),
                str(item[1]),
            ),
        )
        if not ranked or ranked[0][0] < threshold:
            continue
        second_score = ranked[1][0] if len(ranked) > 1 else 0.0
        same_normalized_name = (
            len(ranked) > 1
            and normalized_stimulus_name(ranked[0][1].stem)[0]
            == normalized_stimulus_name(ranked[1][1].stem)[0]
        )
        if ranked[0][0] - second_score < min_margin and not same_normalized_name:
            ambiguities.append(
                {
                    "audio": audio,
                    "best": ranked[0][1],
                    "best_score": ranked[0][0],
                    "second": ranked[1][1],
                    "second_score": second_score,
                    "min_margin": min_margin,
                }
            )
            continue
        candidates.append((ranked[0][0], audio, ranked[0][1]))

    matches: dict[Path, TranscriptMatch] = {}
    used_transcripts: set[Path] = set()
    for score, audio, transcript in sorted(
        candidates,
        key=lambda item: (item[0], str(item[1]), str(item[2])),
        reverse=True,
    ):
        if transcript in used_transcripts:
            continue
        matches[audio] = TranscriptMatch(audio, transcript, score)
        used_transcripts.add(transcript)

    if alias_count_heuristic:
        original_audio_counts: dict[str, int] = {}
        original_text_counts: dict[str, int] = {}
        for path in audio_paths:
            alias = filename_alias(path)
            if alias:
                original_audio_counts[alias] = original_audio_counts.get(alias, 0) + 1
        for path in transcript_paths:
            alias = filename_alias(path)
            if alias:
                original_text_counts[alias] = original_text_counts.get(alias, 0) + 1

        unmatched_audio = [path for path in audio_paths if path not in matches]
        unused_transcripts = [
            path for path in transcript_paths if path not in used_transcripts
        ]
        audio_by_alias: dict[str, list[Path]] = {}
        text_by_alias: dict[str, list[Path]] = {}
        for path in unmatched_audio:
            alias = filename_alias(path)
            if alias:
                audio_by_alias.setdefault(alias, []).append(path)
        for path in unused_transcripts:
            alias = filename_alias(path)
            if alias:
                text_by_alias.setdefault(alias, []).append(path)

        for alias in sorted(audio_by_alias.keys() & text_by_alias.keys()):
            alias_audio = audio_by_alias[alias]
            alias_text = text_by_alias[alias]
            if (
                original_audio_counts[alias] != original_text_counts[alias]
                or len(alias_audio) != len(alias_text)
            ):
                continue
            assignment = _best_one_to_one_assignment(alias_audio, alias_text)
            for score, audio, transcript in assignment:
                if score < alias_min_score:
                    continue
                matches[audio] = TranscriptMatch(
                    audio, transcript, score, method="equal_alias_count"
                )
                used_transcripts.add(transcript)
    return matches, ambiguities


@dataclass
class PipelineReport:
    """Serializable summary returned even when some batch items fail."""

    run_id: str
    log_path: str
    status: str
    processes: list[ProcessResult]

    @property
    def failed_items(self) -> int:
        return sum(process.failed for process in self.processes)

    @property
    def skipped_items(self) -> int:
        return sum(process.skipped for process in self.processes)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_items"] = self.failed_items
        payload["skipped_items"] = self.skipped_items
        return payload


class BatchPipeline:
    """Run configured processes item-by-item with structured failure isolation."""

    def __init__(
        self,
        config: dict[str, Any],
        config_dir: Path,
        registry: EventRegistry,
        reporter: TerminalReporter | None = None,
    ) -> None:
        """Configure a batch run from validated YAML and a shared registry."""
        self.config = config
        self.config_dir = config_dir
        self.registry = registry
        self.reporter = reporter or TerminalReporter(enabled=False)
        self.default_tr_s = float(config.get("defaults", {}).get("tr_s", 1.0))
        batch = config.get("batch", {})
        self.continue_on_error = bool(batch.get("continue_on_error", True))
        self._ctc_runtime: Any | None = None
        self._openl3_runtime: Any | None = None
        self._transcriber: Any | None = None
        self._sentence_model_name: str | None = None
        self._transcript_matches: dict[Path, TranscriptMatch] = {}

    def run(self, selected: Sequence[str] | None = None) -> PipelineReport:
        """Run enabled processes and always return completed item outcomes."""
        names = selected_processes(self.config, selected)
        run_timer = self.registry.timer()
        self.reporter.pipeline_pending(names)
        self.registry.emit(
            "pipeline_started",
            "started",
            source_path=self.config_dir,
            details={"processes": names, "continue_on_error": self.continue_on_error},
        )
        results: list[ProcessResult] = []
        try:
            self._preload_models(names)
            for name in names:
                result = self._run_process(name, self.config["processes"][name])
                results.append(result)
                if result.failed and not self.continue_on_error:
                    break
        except KeyboardInterrupt:
            self.reporter.pipeline_interrupted()
            self.registry.emit(
                "pipeline_interrupted",
                "failed",
                duration_ms=run_timer.elapsed_ms,
                details={"reason": "keyboard_interrupt"},
            )
            raise
        except Exception as error:
            self.registry.emit(
                "pipeline_failed",
                "failed",
                duration_ms=run_timer.elapsed_ms,
                error=error,
            )
            raise
        finally:
            self._close_models()

        failures = sum(result.failed for result in results)
        status = "success" if failures == 0 else "failed"
        self.registry.emit(
            "pipeline_completed",
            status,
            duration_ms=run_timer.elapsed_ms,
            details={
                "processes_completed": len(results),
                "items_succeeded": sum(result.succeeded for result in results),
                "items_skipped": sum(result.skipped for result in results),
                "items_failed": failures,
            },
        )
        self.reporter.pipeline_finished(
            sum(result.succeeded for result in results),
            failures,
            sum(result.skipped for result in results),
        )
        return PipelineReport(
            run_id=self.registry.run_id,
            log_path=str(self.registry.path),
            status=status,
            processes=results,
        )

    def _preload_models(self, names: Sequence[str]) -> None:
        """Load reusable models required by selected processes before any items run."""
        self._preload_ctc_model(names)
        # OpenL3 is intentionally initialized immediately after the phoneme model.
        self._preload_openl3_model(names)
        self._preload_transcription_models(names)
        self._preload_sentence_model(names)

    def _preload_ctc_model(self, names: Sequence[str]) -> None:
        """Preload the selected CTC model once when phonetics uses that backend."""
        if "acoustic_phonetics" not in names:
            return
        process = self.config["processes"]["acoustic_phonetics"]
        options = process.get("settings", {}).get("extractor_options", {})
        if options.get("posterior_backend", "ctc") != "ctc":
            return
        inputs = discover_files(
            process["input"],
            self.config_dir,
            label="processes.acoustic_phonetics.input",
        )
        invalid = [path for path in inputs if path.suffix.casefold() != ".wav"]
        if invalid:
            raise InputError(
                "Process 'acoustic_phonetics' only accepts WAV inputs; invalid files: "
                + ", ".join(str(path) for path in invalid)
            )

        model = str(
            options.get(
                "ctc_model", "bobboyms/wav2vec2-base-en-phoneme-ctc-41h"
            )
        )
        self.reporter.model_pending("acoustic_phonetics", model)
        timer = self.registry.timer()
        self.registry.emit(
            "model_load_started",
            "started",
            process="acoustic_phonetics",
            details={"model": model, "requested_device": options.get("ctc_device", "auto")},
        )
        try:
            from natural_features.features.speech.phonology import load_ctc_runtime

            self._ctc_runtime = load_ctc_runtime(
                model=model,
                local_files_only=bool(options.get("ctc_local_files_only", True)),
                device=str(options.get("ctc_device", "auto")),
            )
        except Exception as error:
            self.reporter.model_finished(
                "acoustic_phonetics", model, "failed", detail=str(error)
            )
            self.registry.emit(
                "model_load_failed",
                "failed",
                process="acoustic_phonetics",
                duration_ms=timer.elapsed_ms,
                error=error,
                details={"model": model},
            )
            raise ProcessError(
                "acoustic_phonetics",
                None,
                f"unable to preload CTC model '{model}': {error}",
                hint=(
                    "Check ctc_model/ctc_local_files_only, install the audio extras, "
                    "or set ctc_device to an available device."
                ),
            ) from error

        detail = f"ready on {self._ctc_runtime.device}"
        self.reporter.model_finished(
            "acoustic_phonetics", model, "success", detail=detail
        )
        self.registry.emit(
            "model_load_completed",
            "success",
            process="acoustic_phonetics",
            duration_ms=timer.elapsed_ms,
            details={"model": model, "device": self._ctc_runtime.device},
        )

    def _preload_openl3_model(self, names: Sequence[str]) -> None:
        """Preload OpenL3 with a TensorFlow-specific device configuration."""
        if "audio_embeddings" not in names:
            return
        process = self.config["processes"]["audio_embeddings"]
        options = process.get("settings", {}).get("embedding_options", {})
        model = (
            f"openl3:{options.get('input_repr', 'linear')}:"
            f"{options.get('content_type', 'env')}:"
            f"{options.get('embedding_size', 512)}"
        )
        self.reporter.model_pending("audio_embeddings", model)
        timer = self.registry.timer()
        self.registry.emit(
            "model_load_started",
            "started",
            process="audio_embeddings",
            details={
                "model": model,
                "framework": "tensorflow",
                "requested_device": options.get("tensorflow_device", "auto"),
            },
        )
        try:
            from langfeat_analysis.preprocessing.audio import load_openl3_runtime

            self._openl3_runtime = load_openl3_runtime(options)
        except Exception as error:
            self._model_load_failed("audio_embeddings", model, timer.elapsed_ms, error)
            raise ProcessError(
                "audio_embeddings",
                None,
                f"unable to preload OpenL3 model '{model}': {error}",
                hint=(
                    "Check the OpenL3 options and tensorflow_device. PyTorch's "
                    "ctc_device setting does not configure TensorFlow."
                ),
            ) from error
        self._model_load_succeeded(
            "audio_embeddings",
            model,
            timer.elapsed_ms,
            self._openl3_runtime.device,
            framework="tensorflow",
        )

    def _preload_transcription_models(self, names: Sequence[str]) -> None:
        """Preload the Qwen/MLX ASR and forced-aligner models when enabled."""
        if "transcripts" not in names:
            return
        process = self.config["processes"]["transcripts"]
        _, unmatched = self._plan_transcriptions(process)
        if not unmatched:
            self.registry.emit(
                "model_load_skipped",
                "skipped",
                process="transcripts",
                details={
                    "reason": "every audio input has a validated transcript match",
                    "matched_audio": len(self._transcript_matches),
                },
            )
            return
        settings = dict(process.get("settings", {}))
        output = process["output"]
        settings["output_filename_template"] = output.get(
            "file_template", "{stimulus}-annotations.csv"
        )
        model = "transcription+forced-aligner"
        self.reporter.model_pending("transcripts", model)
        timer = self.registry.timer()
        self.registry.emit("model_load_started", "started", process="transcripts", details={"model": model})
        try:
            from langfeat_analysis.preprocessing.text import TranscriptGenerator

            self._transcriber = TranscriptGenerator(
                output_path=resolve_path(output["directory"], self.config_dir),
                registry=self.registry,
                **settings,
            )
            backend = self._transcriber.preload_models()
        except Exception as error:
            self._model_load_failed("transcripts", model, timer.elapsed_ms, error)
            raise ProcessError("transcripts", None, f"unable to preload transcription models: {error}") from error
        self._model_load_succeeded("transcripts", model, timer.elapsed_ms, backend)

    def _plan_transcriptions(
        self, process: dict[str, Any]
    ) -> tuple[list[Path], list[Path]]:
        """Match audio to existing text and return all audio plus unmatched audio."""
        input_config = process["input"]
        audio_spec = input_config.get("audio", input_config)
        audio_paths = discover_files(
            audio_spec,
            self.config_dir,
            label="processes.transcripts.input.audio",
        )
        invalid = [path for path in audio_paths if path.suffix.casefold() != ".wav"]
        if invalid:
            raise InputError(
                "Process 'transcripts' only accepts WAV audio inputs; invalid files: "
                + ", ".join(str(path) for path in invalid)
            )

        transcript_paths: list[Path] = []
        existing_spec = input_config.get("existing_transcripts")
        if isinstance(existing_spec, dict):
            transcript_paths.extend(
                self._discover_optional_files(
                    existing_spec,
                    "processes.transcripts.input.existing_transcripts",
                )
            )
        output_spec = {
            "directory": process["output"]["directory"],
            "patterns": [
                "*.csv", "*.CSV", "*.tsv", "*.TSV", "*.txt", "*.TXT",
                "*.textgrid", "*.TextGrid", "*.eaf", "*.EAF",
            ],
        }
        transcript_paths.extend(
            self._discover_optional_files(
                output_spec, "processes.transcripts.output existing files"
            )
        )
        transcript_paths = sorted(set(transcript_paths))

        matching = input_config.get("matching", {})
        threshold = float(matching.get("threshold", 0.72))
        min_margin = float(matching.get("min_margin", 0.05))
        alias_count_heuristic = bool(matching.get("alias_count_heuristic", True))
        alias_min_score = float(matching.get("alias_min_score", 0.35))
        matches, ambiguities = select_transcript_matches(
            audio_paths,
            transcript_paths,
            threshold=threshold,
            min_margin=min_margin,
            alias_count_heuristic=alias_count_heuristic,
            alias_min_score=alias_min_score,
        )
        for ambiguity in ambiguities:
            self.registry.emit(
                "transcript_match_ambiguous",
                "warning",
                process="transcripts",
                input_path=ambiguity["audio"],
                details={
                    key: str(value) if isinstance(value, Path) else value
                    for key, value in ambiguity.items()
                    if key != "audio"
                },
            )
        self._transcript_matches = matches
        return audio_paths, [path for path in audio_paths if path not in matches]

    def _discover_optional_files(
        self, specification: dict[str, Any], label: str
    ) -> list[Path]:
        """Discover optional transcript files, tolerating an absent directory."""
        if "files" in specification:
            return discover_files(specification, self.config_dir, label=label, allow_empty=True)
        directory = resolve_path(specification.get("directory", ""), self.config_dir)
        if not directory.is_dir():
            return []
        return discover_files(specification, self.config_dir, label=label, allow_empty=True)

    def _preload_sentence_model(self, names: Sequence[str]) -> None:
        """Preload the configured SentenceTransformer when text embeddings run."""
        if "text_embeddings" not in names:
            return
        model = str(
            self.config["processes"]["text_embeddings"]["settings"]["model_name"]
        )
        self.reporter.model_pending("text_embeddings", model)
        timer = self.registry.timer()
        self.registry.emit(
            "model_load_started",
            "started",
            process="text_embeddings",
            details={"model": model},
        )
        try:
            from langfeat_analysis.preprocessing.text import preload_sentence_model

            preload_sentence_model(model)
            self._sentence_model_name = model
        except Exception as error:
            self._model_load_failed("text_embeddings", model, timer.elapsed_ms, error)
            raise ProcessError(
                "text_embeddings",
                None,
                f"unable to preload sentence model '{model}': {error}",
            ) from error
        self._model_load_succeeded(
            "text_embeddings", model, timer.elapsed_ms, "framework auto"
        )

    def _model_load_failed(
        self, process: str, model: str, duration_ms: float, error: Exception
    ) -> None:
        self.reporter.model_finished(process, model, "failed", detail=str(error))
        self.registry.emit(
            "model_load_failed", "failed", process=process,
            duration_ms=duration_ms, error=error, details={"model": model},
        )

    def _model_load_succeeded(
        self, process: str, model: str, duration_ms: float, device: str, **details: Any
    ) -> None:
        self.reporter.model_finished(process, model, "success", detail=f"ready on {device}")
        self.registry.emit(
            "model_load_completed", "success", process=process,
            duration_ms=duration_ms,
            details={"model": model, "device": device, **details},
        )

    def _close_models(self) -> None:
        """Release reusable inference resources after the pipeline finishes."""
        resources = [
            ("acoustic_phonetics", self._ctc_runtime, "close"),
            ("audio_embeddings", self._openl3_runtime, "close"),
            ("transcripts", self._transcriber, "unload_models"),
        ]
        self._ctc_runtime = None
        self._openl3_runtime = None
        self._transcriber = None
        for process, resource, method in resources:
            if resource is None:
                continue
            try:
                getattr(resource, method)()
            except Exception as error:
                logging.warning("Unable to fully release %s models: %s", process, error)
                self.registry.emit(
                    "model_unload_failed", "warning", process=process, error=error
                )
        if self._sentence_model_name is not None:
            model_name = self._sentence_model_name
            self._sentence_model_name = None
            try:
                from langfeat_analysis.preprocessing.text import unload_sentence_model

                unload_sentence_model(model_name)
            except Exception as error:
                logging.warning("Unable to release text embedding model: %s", error)
                self.registry.emit(
                    "model_unload_failed",
                    "warning",
                    process="text_embeddings",
                    error=error,
                )

    def _report_ctc_progress(
        self, input_path: Path, completed: int, total: int
    ) -> None:
        """Send chunk progress to both the terminal and JSONL event registry."""
        self.reporter.chunk_progress(
            "acoustic_phonetics", input_path, completed, total
        )
        self.registry.emit(
            "item_progress",
            "started",
            process="acoustic_phonetics",
            input_path=input_path,
            details={"unit": "chunk", "completed": completed, "total": total},
        )

    def _run_process(self, name: str, process: dict[str, Any]) -> ProcessResult:
        timer = self.registry.timer()
        process_error_message = ""
        self.reporter.process_pending(name)
        self.registry.emit("process_started", "started", process=name)
        try:
            if name in AUDIO_PROCESSES:
                result = self._run_audio_process(name, process)
            elif name == "transcripts":
                result = self._run_transcripts(process)
            elif name == "text_embeddings":
                result = self._run_text_embeddings(process)
            else:  # Configuration validation should make this unreachable.
                raise ProcessError(name, None, "process implementation is unavailable")
        except Exception as error:
            process_error_message = str(error)
            self.registry.emit(
                "process_failed",
                "failed",
                process=name,
                duration_ms=timer.elapsed_ms,
                error=error,
            )
            if not self.continue_on_error:
                raise
            result = ProcessResult(
                process=name,
                status="failed",
                items=[
                    ItemResult(
                        process=name,
                        input_path="",
                        status="failed",
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                ],
            )
        else:
            self.registry.emit(
                "process_completed",
                "success" if result.failed == 0 else "failed",
                process=name,
                duration_ms=timer.elapsed_ms,
                details={
                    "succeeded": result.succeeded,
                    "skipped": result.skipped,
                    "failed": result.failed,
                },
            )
        self.reporter.process_finished(
            name,
            result.succeeded,
            result.failed,
            skipped=result.skipped,
            error_message=process_error_message,
        )
        return result

    def _run_audio_process(
        self, name: str, process: dict[str, Any]
    ) -> ProcessResult:
        inputs = discover_files(
            process["input"], self.config_dir, label=f"processes.{name}.input"
        )
        invalid = [path for path in inputs if path.suffix.casefold() != ".wav"]
        if invalid:
            raise InputError(
                f"Process '{name}' only accepts WAV inputs; invalid files: "
                + ", ".join(str(path) for path in invalid)
            )
        if name == "audio_embeddings":
            return self._run_audio_embeddings(process, inputs)
        return self._run_items(
            name,
            inputs,
            lambda path: self._run_audio_item(name, process, path),
        )

    def _run_audio_embeddings(
        self, process: dict[str, Any], inputs: Sequence[Path]
    ) -> ProcessResult:
        """Run memory-aware multi-stimulus OpenL3 batches with item isolation."""
        from langfeat_analysis.preprocessing.audio import AudioVectorizer

        output = process["output"]
        settings = process.get("settings", {})
        processor = AudioVectorizer(
            input_dir=inputs[0].parent,
            output_dir=resolve_path(output["directory"], self.config_dir),
            tr_s=float(settings.get("tr_s", self.default_tr_s)),
            embedding_options=settings.get("embedding_options"),
            runtime=self._openl3_runtime,
            filename_template=output.get("file_template", "{title}.json"),
            registry=self.registry,
        )
        groups = processor.compatible_groups(inputs)
        results: list[ItemResult] = []
        self.reporter.items_pending("audio_embeddings", len(inputs))

        def execute(group: list[Path]) -> None:
            timer = self.registry.timer()
            try:
                outputs_by_path = processor.process_paths(group)
            except Exception as error:
                if len(group) > 1:
                    self.registry.emit(
                        "stimulus_batch_retry",
                        "warning",
                        process="audio_embeddings",
                        input_path=group[0],
                        error=error,
                        details={
                            "stimulus_count": len(group),
                            "retry": "sequential",
                        },
                    )
                    for path in group:
                        execute([path])
                        if (
                            results
                            and results[-1].status == "failed"
                            and not self.continue_on_error
                        ):
                            break
                    return
                path = group[0]
                explained = self._explain_error("audio_embeddings", path, error)
                results.append(self._failed_item("audio_embeddings", path, explained))
                self.registry.emit(
                    "item_failed", "failed", process="audio_embeddings",
                    input_path=path, duration_ms=timer.elapsed_ms, error=explained,
                )
                self.reporter.item_finished(
                    "audio_embeddings", path, "failed", error_message=str(explained)
                )
                return

            for path in group:
                outputs = outputs_by_path.get(path, [])
                if not outputs or any(not Path(value).is_file() for value in outputs):
                    error = ProcessError(
                        "audio_embeddings", path,
                        "OpenL3 completed without producing every reported output",
                    )
                    results.append(self._failed_item("audio_embeddings", path, error))
                    self.registry.emit(
                        "item_failed", "failed", process="audio_embeddings",
                        input_path=path, duration_ms=timer.elapsed_ms, error=error,
                    )
                    self.reporter.item_finished(
                        "audio_embeddings", path, "failed", error_message=str(error)
                    )
                    continue
                output_strings = [str(value) for value in outputs]
                results.append(
                    ItemResult("audio_embeddings", str(path), "success", output_strings)
                )
                self.registry.emit(
                    "item_completed", "success", process="audio_embeddings",
                    input_path=path, duration_ms=timer.elapsed_ms,
                    details={"output_paths": output_strings, "stimulus_batch_size": len(group)},
                )
                self.reporter.item_finished(
                    "audio_embeddings", path, "success", output_count=len(outputs)
                )

        for group in groups:
            for path in group:
                self.registry.emit(
                    "item_started", "started", process="audio_embeddings", input_path=path
                )
            execute(group)
            if any(item.status == "failed" for item in results) and not self.continue_on_error:
                break
        status = "success" if all(item.status == "success" for item in results) else "failed"
        return ProcessResult("audio_embeddings", status, results)

    def _run_audio_item(
        self, name: str, process: dict[str, Any], path: Path
    ) -> list[Path]:
        try:
            from langfeat_analysis.preprocessing.audio import (
                AudioFeaturePreprocessor,
                AudioPhonemePreprocessor,
                AudioVectorizer,
            )
        except ImportError as error:
            raise ProcessError(
                name,
                path,
                f"required audio dependency could not be imported: {error}",
                hint="Install the project with `pip install -e '.[audio]'`.",
            ) from error

        output = process["output"]
        output_dir = resolve_path(output["directory"], self.config_dir)
        template = output.get("file_template", "{title}.json")
        settings = process.get("settings", {})
        tr_s = float(settings.get("tr_s", self.default_tr_s))
        common = {
            "input_dir": path.parent,
            "output_dir": output_dir,
            "tr_s": tr_s,
            "pattern": path.name,
            "filename_template": template,
            "registry": self.registry,
        }
        try:
            if name == "audio_features":
                processor = AudioFeaturePreprocessor(
                    features=settings["features"], **common
                )
            elif name == "acoustic_phonetics":
                processor = AudioPhonemePreprocessor(
                    extractor_options=settings.get("extractor_options"),
                    ctc_runtime=self._ctc_runtime,
                    progress_callback=lambda completed, total: self._report_ctc_progress(
                        path, completed, total
                    ),
                    **common,
                )
            else:
                processor = AudioVectorizer(
                    embedding_options=settings.get("embedding_options"),
                    runtime=self._openl3_runtime,
                    **common,
                )
            return processor.process()
        except ProcessError:
            raise
        except Exception as error:
            raise self._explain_error(name, path, error) from error

    def _run_transcripts(self, process: dict[str, Any]) -> ProcessResult:
        inputs, pending_inputs = self._plan_transcriptions(process)
        try:
            from langfeat_analysis.preprocessing.text import TranscriptGenerator
        except ImportError as error:
            raise ProcessError(
                "transcripts",
                None,
                f"required transcription dependency could not be imported: {error}",
                hint="Install the project with `pip install -e '.[text]'`.",
            ) from error

        output = process["output"]
        options = dict(process.get("settings", {}))
        options["output_filename_template"] = output.get(
            "file_template", "{stimulus}-annotations.csv"
        )
        owns_transcriber = self._transcriber is None
        transcriber = self._transcriber or TranscriptGenerator(
            output_path=resolve_path(output["directory"], self.config_dir),
            registry=self.registry,
            **options,
        )
        try:
            result = self._run_items(
                "transcripts",
                pending_inputs,
                lambda path: [Path(transcriber.transcribe_audio(path))],
                total_items=len(inputs),
            )
            for path in inputs:
                match = self._transcript_matches.get(path)
                if match is None:
                    continue
                item = ItemResult(
                    "transcripts",
                    str(path),
                    "skipped",
                    output_paths=[str(match.transcript)],
                )
                result.items.append(item)
                self.registry.emit(
                    "item_skipped",
                    "skipped",
                    process="transcripts",
                    input_path=path,
                    output_path=match.transcript,
                    details={
                        "reason": "validated existing transcript filename match",
                        "match_score": match.score,
                        "match_method": match.method,
                        "alias": filename_alias(path),
                    },
                )
                self.reporter.item_finished(
                    "transcripts",
                    path,
                    "skipped",
                    skip_message=(
                        f"matched {match.transcript.name} at score {match.score:.3f} "
                        f"via {match.method}"
                    ),
                )
            result.items.sort(key=lambda item: item.input_path)
            result.status = "success" if result.failed == 0 else "failed"
            return result
        finally:
            if owns_transcriber:
                transcriber.unload_models()

    def _run_text_embeddings(self, process: dict[str, Any]) -> ProcessResult:
        pairs, pairing_errors = self._match_text_inputs(process)
        if pairing_errors and not self.continue_on_error:
            return ProcessResult(
                process="text_embeddings", status="failed", items=pairing_errors
            )
        try:
            from langfeat_analysis.preprocessing.text import TextEmbedderGrid
        except ImportError as error:
            raise ProcessError(
                "text_embeddings",
                None,
                f"required text dependency could not be imported: {error}",
                hint="Install the project with `pip install -e '.[text]'`.",
            ) from error

        settings = process.get("settings", {})
        output = process["output"]
        output_dir = resolve_path(output["directory"], self.config_dir)

        def embed(pair: tuple[Path, Path]) -> list[Path]:
            annotation, stimulus = pair
            try:
                return [
                    TextEmbedderGrid(
                        annotation_path=annotation,
                        stimulus_audio=stimulus,
                        output_dir=output_dir,
                        tr_s=float(settings.get("tr_s", self.default_tr_s)),
                        model_name=settings["model_name"],
                        context_window=int(settings.get("context_window", 3)),
                        output_filename_template=output.get(
                            "file_template", "{title}.json"
                        ),
                        strict_annotations=bool(
                            settings.get("strict_annotations", False)
                        ),
                        registry=self.registry,
                    ).process()
                ]
            except Exception as error:
                raise self._explain_error(
                    "text_embeddings", annotation, error
                ) from error

        result = self._run_items(
            "text_embeddings",
            pairs,
            embed,
            input_name=lambda pair: str(pair[0]),
            total_items=len(pairs) + len(pairing_errors),
        )
        for item in pairing_errors:
            self.reporter.item_finished(
                "text_embeddings",
                item.input_path,
                "failed",
                error_message=item.error_message,
            )
        result.items = pairing_errors + result.items
        result.status = "success" if result.failed == 0 else "failed"
        return result

    def _match_text_inputs(
        self, process: dict[str, Any]
    ) -> tuple[list[tuple[Path, Path]], list[ItemResult]]:
        input_config = process["input"]
        annotations = discover_files(
            input_config["annotations"],
            self.config_dir,
            label="processes.text_embeddings.input.annotations",
        )
        stimuli = discover_files(
            input_config["stimuli"],
            self.config_dir,
            label="processes.text_embeddings.input.stimuli",
        )
        matching = input_config.get("matching", {})
        suffix = str(matching.get("annotation_suffix", "-annotations"))
        case_sensitive = bool(matching.get("case_sensitive", False))
        strategy = str(matching.get("strategy", "stem"))
        normalize = (lambda value: value) if case_sensitive else str.casefold

        stimulus_by_stem: dict[str, Path] = {}
        duplicates: set[str] = set()
        for stimulus in stimuli:
            key = normalize(stimulus.stem)
            if key in stimulus_by_stem:
                duplicates.add(key)
            stimulus_by_stem[key] = stimulus
        if duplicates:
            raise InputError(
                "Stimulus matching is ambiguous because multiple audio files share "
                f"these stems: {', '.join(sorted(duplicates))}."
            )

        collector_stimuli: dict[str, set[Path]] = {}
        if strategy in {"collector", "auto"}:
            for stimulus in stimuli:
                for key in collector_match_keys(
                    stimulus.stem, "audio", case_sensitive=case_sensitive
                ):
                    collector_stimuli.setdefault(key, set()).add(stimulus)

        pairs: list[tuple[Path, Path]] = []
        errors: list[ItemResult] = []
        for annotation in annotations:
            stem = annotation.stem
            if suffix and normalize(stem).endswith(normalize(suffix)):
                stem = stem[: -len(suffix)]
            stimulus = (
                stimulus_by_stem.get(normalize(stem))
                if strategy in {"stem", "auto"}
                else None
            )
            collector_candidates: set[Path] = set()
            if stimulus is None and strategy in {"collector", "auto"}:
                for key in collector_match_keys(
                    annotation.stem, "text", case_sensitive=case_sensitive
                ):
                    collector_candidates.update(collector_stimuli.get(key, set()))
                if len(collector_candidates) == 1:
                    stimulus = next(iter(collector_candidates))
                elif len(collector_candidates) > 1:
                    error = ProcessError(
                        "text_embeddings",
                        annotation,
                        "collector filename matches multiple audio stimuli: "
                        + ", ".join(sorted(path.name for path in collector_candidates)),
                        hint="Use --audio-input-dir with a narrower directory or rename ambiguous files.",
                    )
                    errors.append(
                        self._failed_item("text_embeddings", annotation, error)
                    )
                    self.registry.emit(
                        "item_failed",
                        "failed",
                        process="text_embeddings",
                        input_path=annotation,
                        error=error,
                    )
                    continue
            if stimulus is None:
                error = ProcessError(
                    "text_embeddings",
                    annotation,
                    f"no stimulus audio matched annotation stem '{stem}' "
                    f"using strategy '{strategy}'",
                    hint=(
                        "Check matching.strategy/annotation_suffix, or use the "
                        "collector-style alias_audio_NAME_CHECKSUM and "
                        "alias_text_NAME_CHECKSUM filenames."
                    ),
                )
                errors.append(self._failed_item("text_embeddings", annotation, error))
                self.registry.emit(
                    "item_failed",
                    "failed",
                    process="text_embeddings",
                    input_path=annotation,
                    error=error,
                )
            else:
                pairs.append((annotation, stimulus))
        return pairs, errors

    def _run_items(
        self,
        process: str,
        items: Sequence[Any],
        operation: Callable[[Any], list[Path]],
        *,
        input_name: Callable[[Any], str] = str,
        total_items: int | None = None,
    ) -> ProcessResult:
        results: list[ItemResult] = []
        self.reporter.items_pending(process, total_items if total_items is not None else len(items))
        for item in items:
            name = input_name(item)
            timer = self.registry.timer()
            self.registry.emit(
                "item_started", "started", process=process, input_path=name
            )
            try:
                outputs = operation(item)
                if not outputs:
                    raise ProcessError(
                        process, name, "processor completed without producing output files"
                    )
                missing = [path for path in outputs if not Path(path).is_file()]
                if missing:
                    raise ProcessError(
                        process,
                        name,
                        "processor reported outputs that do not exist: "
                        + ", ".join(str(path) for path in missing),
                    )
            except Exception as error:
                explained = (
                    error
                    if isinstance(error, ProcessError)
                    else self._explain_error(process, Path(name), error)
                )
                results.append(self._failed_item(process, name, explained))
                self.registry.emit(
                    "item_failed",
                    "failed",
                    process=process,
                    input_path=name,
                    duration_ms=timer.elapsed_ms,
                    error=explained,
                )
                logging.error("%s", explained)
                self.reporter.item_finished(
                    process,
                    name,
                    "failed",
                    error_message=str(explained),
                )
                if not self.continue_on_error:
                    break
            else:
                output_strings = [str(path) for path in outputs]
                results.append(
                    ItemResult(process, name, "success", output_strings)
                )
                self.registry.emit(
                    "item_completed",
                    "success",
                    process=process,
                    input_path=name,
                    duration_ms=timer.elapsed_ms,
                    details={"output_paths": output_strings},
                )
                self.reporter.item_finished(
                    process,
                    name,
                    "success",
                    output_count=len(output_strings),
                )
        status = "success" if all(item.status == "success" for item in results) else "failed"
        return ProcessResult(process=process, status=status, items=results)

    @staticmethod
    def _failed_item(
        process: str, input_path: str | Path, error: Exception
    ) -> ItemResult:
        return ItemResult(
            process=process,
            input_path=str(input_path),
            status="failed",
            error_type=type(error).__name__,
            error_message=str(error),
        )

    @staticmethod
    def _explain_error(
        process: str, input_path: str | Path, error: Exception
    ) -> ProcessError:
        """Translate common low-level exceptions into actionable failures."""
        if isinstance(error, FileNotFoundError):
            reason = f"a required file was not found: {error}"
            hint = "Verify configured paths and generated upstream outputs."
        elif isinstance(error, PermissionError):
            reason = f"permission was denied: {error}"
            hint = "Check read access on inputs and write access on the output directory."
        elif isinstance(error, ImportError):
            reason = f"a required Python dependency is unavailable: {error}"
            hint = "Install the appropriate project extra: .[audio], .[text], or .[all]."
        elif isinstance(error, MemoryError):
            reason = "the process exhausted available memory"
            hint = "Reduce model size or process shorter stimuli."
        elif isinstance(error, KeyError):
            reason = f"required data or configuration key is missing: {error}"
            hint = "Check the process settings and annotation column names."
        elif isinstance(error, ValueError):
            reason = f"input data or setting is invalid: {error}"
            hint = "Inspect the named input and the corresponding YAML settings."
        else:
            reason = f"{type(error).__name__}: {error}"
            hint = "See this item's traceback in the JSONL registry for the failing call."
        return ProcessError(process, input_path, reason, hint=hint)
