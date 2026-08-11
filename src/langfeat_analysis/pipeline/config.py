"""YAML loading, validation, and portable path discovery."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

import yaml

from .errors import ConfigurationError, InputError


PROCESS_ORDER = (
    "audio_features",
    "acoustic_phonetics",
    "audio_embeddings",
    "transcripts",
    "text_embeddings",
)


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load YAML and report syntax errors with their source location."""
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise ConfigurationError(
            f"Configuration file does not exist: {path}. "
            "Pass --config with a readable YAML file."
        )
    try:
        with path.open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
    except yaml.YAMLError as error:
        mark = getattr(error, "problem_mark", None)
        location = f" at line {mark.line + 1}, column {mark.column + 1}" if mark else ""
        raise ConfigurationError(f"Invalid YAML in {path}{location}: {error}") from error
    except OSError as error:
        raise ConfigurationError(f"Unable to read configuration {path}: {error}") from error

    validate_config(config, path)
    return config, path.parent


def validate_config(config: Any, path: Path | None = None) -> None:
    """Validate required process structure before any models or files are loaded."""
    label = str(path or "configuration")
    if not isinstance(config, dict):
        raise ConfigurationError(f"{label} must contain a YAML mapping at its root.")
    if config.get("version") != 1:
        raise ConfigurationError(
            f"{label} must declare supported schema 'version: 1', "
            f"got {config.get('version')!r}."
        )
    models = config.get("models", {})
    if not isinstance(models, dict):
        raise ConfigurationError("'models' must be a mapping.")
    if "offline" in models and not isinstance(models["offline"], bool):
        raise ConfigurationError("models.offline must be true or false.")
    cache_directory = models.get("cache_directory")
    if cache_directory is not None and not isinstance(cache_directory, (str, Path)):
        raise ConfigurationError("models.cache_directory must be a filesystem path.")
    processes = config.get("processes")
    if not isinstance(processes, dict) or not processes:
        raise ConfigurationError(f"{label} requires a non-empty 'processes' mapping.")
    unknown = set(processes) - set(PROCESS_ORDER)
    if unknown:
        raise ConfigurationError(
            f"Unknown process name(s): {', '.join(sorted(unknown))}. "
            f"Supported processes: {', '.join(PROCESS_ORDER)}."
        )

    default_tr = config.get("defaults", {}).get("tr_s", 1.0)
    _positive_number(default_tr, "defaults.tr_s")
    batch = config.get("batch", {})
    if not isinstance(batch, dict):
        raise ConfigurationError("'batch' must be a mapping.")
    if "workers" in batch and batch["workers"] != 1:
        raise ConfigurationError(
            "batch.workers currently must be 1 because ASR and embedding models "
            "are not safe to share across worker processes. Batch inputs are still "
            "processed independently and continue_on_error is supported."
        )

    for name, process in processes.items():
        if not isinstance(process, dict):
            raise ConfigurationError(f"processes.{name} must be a mapping.")
        if not isinstance(process.get("enabled", False), bool):
            raise ConfigurationError(f"processes.{name}.enabled must be true or false.")
        if not isinstance(process.get("input"), dict):
            raise ConfigurationError(f"processes.{name}.input must be a mapping.")
        output = process.get("output")
        if not isinstance(output, dict) or not output.get("directory"):
            raise ConfigurationError(
                f"processes.{name}.output.directory is required."
            )
        settings = process.get("settings", {})
        if not isinstance(settings, dict):
            raise ConfigurationError(f"processes.{name}.settings must be a mapping.")
        if "tr_s" in settings:
            _positive_number(settings["tr_s"], f"processes.{name}.settings.tr_s")
        if name == "audio_features":
            features = settings.get("features")
            if not isinstance(features, list) or not features:
                raise ConfigurationError(
                    "processes.audio_features.settings.features must be a non-empty list."
                )
        if name == "acoustic_phonetics":
            options = settings.get("extractor_options", {})
            if not isinstance(options, dict):
                raise ConfigurationError(
                    "processes.acoustic_phonetics.settings.extractor_options "
                    "must be a mapping."
                )
            device = options.get("ctc_device", "auto")
            if device not in {"auto", "cuda", "mps", "cpu"}:
                raise ConfigurationError(
                    "processes.acoustic_phonetics.settings.extractor_options."
                    "ctc_device must be auto, cuda, mps, or cpu."
                )
            if "ctc_chunk_seconds" in options:
                _positive_number(
                    options["ctc_chunk_seconds"],
                    "processes.acoustic_phonetics.settings.extractor_options."
                    "ctc_chunk_seconds",
                )
            batch_size = options.get("ctc_batch_size")
            if batch_size is not None and (
                isinstance(batch_size, bool)
                or not isinstance(batch_size, int)
                or batch_size <= 0
            ):
                raise ConfigurationError(
                    "processes.acoustic_phonetics.settings.extractor_options."
                    "ctc_batch_size must be a positive integer or null."
                )
        if name == "audio_embeddings":
            options = settings.get("embedding_options", {})
            if not isinstance(options, dict):
                raise ConfigurationError(
                    "processes.audio_embeddings.settings.embedding_options "
                    "must be a mapping."
                )
            if options.get("tensorflow_device", "auto") not in {"auto", "gpu", "cpu"}:
                raise ConfigurationError(
                    "processes.audio_embeddings.settings.embedding_options."
                    "tensorflow_device must be auto, gpu, or cpu."
                )
            for key in ("inference_batch_size", "stimulus_batch_size"):
                value = options.get(key)
                if value is not None and (
                    isinstance(value, bool) or not isinstance(value, int) or value <= 0
                ):
                    raise ConfigurationError(
                        "processes.audio_embeddings.settings.embedding_options."
                        f"{key} must be a positive integer."
                    )
            if "max_stimulus_batch_seconds" in options:
                _positive_number(
                    options["max_stimulus_batch_seconds"],
                    "processes.audio_embeddings.settings.embedding_options."
                    "max_stimulus_batch_seconds",
                )
            if "memory_fallback" in options and not isinstance(
                options["memory_fallback"], bool
            ):
                raise ConfigurationError(
                    "processes.audio_embeddings.settings.embedding_options."
                    "memory_fallback must be true or false."
                )
        if name == "transcripts":
            input_config = process["input"]
            if "audio" in input_config and not isinstance(input_config["audio"], dict):
                raise ConfigurationError(
                    "processes.transcripts.input.audio must be a mapping."
                )
            existing = input_config.get("existing_transcripts")
            if existing is not None and not isinstance(existing, dict):
                raise ConfigurationError(
                    "processes.transcripts.input.existing_transcripts must be a mapping."
                )
            matching = input_config.get("matching", {})
            if not isinstance(matching, dict):
                raise ConfigurationError(
                    "processes.transcripts.input.matching must be a mapping."
                )
            for key, default in (
                ("threshold", 0.72),
                ("min_margin", 0.05),
                ("alias_min_score", 0.35),
            ):
                try:
                    value = float(matching.get(key, default))
                except (TypeError, ValueError) as error:
                    raise ConfigurationError(
                        f"processes.transcripts.input.matching.{key} must be a number."
                    ) from error
                if not 0.0 <= value <= 1.0:
                    raise ConfigurationError(
                        f"processes.transcripts.input.matching.{key} must be between 0 and 1."
                    )
            if "alias_count_heuristic" in matching and not isinstance(
                matching["alias_count_heuristic"], bool
            ):
                raise ConfigurationError(
                    "processes.transcripts.input.matching.alias_count_heuristic "
                    "must be true or false."
                )
        if name == "text_embeddings":
            input_config = process["input"]
            for key in ("annotations", "stimuli"):
                if not isinstance(input_config.get(key), dict):
                    raise ConfigurationError(
                        f"processes.text_embeddings.input.{key} must be a mapping."
                    )
            if not settings.get("model_name"):
                raise ConfigurationError(
                    "processes.text_embeddings.settings.model_name is required."
                )
            strategy = input_config.get("matching", {}).get("strategy", "stem")
            if strategy not in {"stem", "collector", "auto"}:
                raise ConfigurationError(
                    "processes.text_embeddings.input.matching.strategy must be "
                    "'stem', 'collector', or 'auto'."
                )


def _positive_number(value: Any, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ConfigurationError(f"{field} must be a positive number, got {value!r}.") from error
    if number <= 0:
        raise ConfigurationError(f"{field} must be greater than zero, got {number}.")
    return number


def resolve_path(value: str | Path, config_dir: Path) -> Path:
    """Resolve a configured path relative to its YAML file."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def discover_files(
    specification: dict[str, Any],
    config_dir: Path,
    *,
    label: str,
    allow_empty: bool = False,
) -> list[Path]:
    """Resolve explicit files or a directory glob without hiding missing inputs."""
    if "files" in specification:
        values = specification["files"]
        if not isinstance(values, list) or not values:
            raise ConfigurationError(f"{label}.files must be a non-empty list.")
        paths = [resolve_path(value, config_dir) for value in values]
        missing = [path for path in paths if not path.is_file()]
        if missing:
            raise InputError(
                f"{label} lists missing or non-file inputs: "
                + ", ".join(str(path) for path in missing)
            )
    else:
        directory_value = specification.get("directory")
        if not directory_value:
            raise ConfigurationError(
                f"{label} needs either a 'files' list or a 'directory'."
            )
        directory = resolve_path(directory_value, config_dir)
        if not directory.is_dir():
            raise InputError(f"{label} directory does not exist: {directory}")
        configured_patterns = specification.get("patterns")
        if configured_patterns is None:
            configured_patterns = [specification.get("pattern", "*")]
        if (
            not isinstance(configured_patterns, list)
            or not configured_patterns
            or not all(isinstance(pattern, str) and pattern for pattern in configured_patterns)
        ):
            raise ConfigurationError(
                f"{label}.patterns must be a non-empty list of glob strings."
            )
        paths = sorted(
            {
                path
                for pattern in configured_patterns
                for path in directory.glob(pattern)
                if path.is_file()
            }
        )

    if not paths and not allow_empty:
        selection = specification.get(
            "patterns", specification.get("pattern", "explicit files")
        )
        raise InputError(f"{label} resolved no files (selection: {selection!r}).")
    return paths


def selected_processes(
    config: dict[str, Any], selected: Iterable[str] | None
) -> list[str]:
    """Return enabled or explicitly selected processes in dependency-safe order."""
    selection = set(selected) if selected is not None else None
    unknown = (selection or set()) - set(PROCESS_ORDER)
    if unknown:
        raise ConfigurationError(f"Unknown process name(s): {', '.join(sorted(unknown))}")
    names = [
        name
        for name in PROCESS_ORDER
        if name in config["processes"]
        and (
            name in selection
            if selection is not None
            else bool(config["processes"][name].get("enabled", False))
        )
    ]
    if not names:
        raise ConfigurationError(
            "No preprocessing processes are enabled or selected. Enable at least "
            "one process in YAML or pass --only PROCESS."
        )
    return names
