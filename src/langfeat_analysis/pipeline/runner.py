"""Batch-safe orchestration for all preprocessing processes."""

from __future__ import annotations

import logging
import re
from dataclasses import asdict, dataclass, field
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

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["failed_items"] = self.failed_items
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

        failures = sum(result.failed for result in results)
        status = "success" if failures == 0 else "failed"
        self.registry.emit(
            "pipeline_completed",
            status,
            duration_ms=run_timer.elapsed_ms,
            details={
                "processes_completed": len(results),
                "items_succeeded": sum(result.succeeded for result in results),
                "items_failed": failures,
            },
        )
        self.reporter.pipeline_finished(
            sum(result.succeeded for result in results), failures
        )
        return PipelineReport(
            run_id=self.registry.run_id,
            log_path=str(self.registry.path),
            status=status,
            processes=results,
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
                details={"succeeded": result.succeeded, "failed": result.failed},
            )
        self.reporter.process_finished(
            name,
            result.succeeded,
            result.failed,
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
        return self._run_items(
            name,
            inputs,
            lambda path: self._run_audio_item(name, process, path),
        )

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
                    extractor_options=settings.get("extractor_options"), **common
                )
            else:
                processor = AudioVectorizer(
                    embedding_options=settings.get("embedding_options"), **common
                )
            return processor.process()
        except ProcessError:
            raise
        except Exception as error:
            raise self._explain_error(name, path, error) from error

    def _run_transcripts(self, process: dict[str, Any]) -> ProcessResult:
        inputs = discover_files(
            process["input"], self.config_dir, label="processes.transcripts.input"
        )
        invalid = [path for path in inputs if path.suffix.casefold() != ".wav"]
        if invalid:
            raise InputError(
                "Process 'transcripts' only accepts WAV inputs; invalid files: "
                + ", ".join(str(path) for path in invalid)
            )
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
        transcriber = TranscriptGenerator(
            output_path=resolve_path(output["directory"], self.config_dir),
            registry=self.registry,
            **options,
        )
        try:
            return self._run_items(
                "transcripts",
                inputs,
                lambda path: [Path(transcriber.transcribe_audio(path))],
            )
        finally:
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
