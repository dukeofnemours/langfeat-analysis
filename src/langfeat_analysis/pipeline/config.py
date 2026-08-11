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
