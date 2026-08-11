#!/usr/bin/env python3
"""Run audio and text preprocessing processes declared in a YAML file.

Paths in the YAML are resolved relative to the YAML file, which keeps the
configuration portable across machines.  Use ``--dry-run`` to inspect the
resolved inputs and outputs without loading models or writing files.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_CONFIG = Path(__file__).with_suffix(".yaml")
PROCESS_ORDER = (
    "audio_features",
    "acoustic_phonetics",
    "audio_embeddings",
    "transcripts",
    "text_embeddings",
)


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load and minimally validate a pipeline YAML file.

    Args:
        config_path: YAML path. Relative paths are interpreted from the current
            working directory; paths inside the YAML use the YAML's directory.
    """
    path = Path(config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Pipeline configuration not found: {path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or not isinstance(config.get("processes"), dict):
        raise ValueError("Configuration must contain a 'processes' mapping.")
    return config, path.parent


def resolve_path(value: str | Path, config_dir: Path) -> Path:
    """Expand a configured path and resolve it relative to ``config_dir``."""
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (config_dir / path).resolve()


def discover_files(specification: dict[str, Any], config_dir: Path) -> list[Path]:
    """Resolve an input specification containing ``directory``/``pattern`` or files.

    Args:
        specification: Mapping with either a ``files`` list or a ``directory``
            and optional glob ``pattern``.
        config_dir: Base directory for relative configured paths.
    """
    if "files" in specification:
        paths = [resolve_path(value, config_dir) for value in specification["files"]]
    else:
        directory = resolve_path(specification["directory"], config_dir)
        paths = sorted(directory.glob(specification.get("pattern", "*")))
    return [path for path in paths if path.is_file()]


def require_inputs(paths: Iterable[Path], process_name: str) -> list[Path]:
    """Materialize ``paths`` and fail clearly when a process has no inputs."""
    materialized = list(paths)
    if not materialized:
        raise FileNotFoundError(f"Process {process_name!r} resolved no input files.")
    return materialized


def process_plan(
    name: str, process: dict[str, Any], config_dir: Path, default_tr_s: float
) -> dict[str, Any]:
    """Return a JSON-serializable description of a configured process."""
    output_dir = resolve_path(process["output"]["directory"], config_dir)
    description: dict[str, Any] = {
        "process": name,
        "enabled": bool(process.get("enabled", False)),
        "output_directory": str(output_dir),
        "tr_s": float(process.get("settings", {}).get("tr_s", default_tr_s)),
    }
    input_config = process.get("input", {})
    if name == "text_embeddings":
        annotations = discover_files(input_config["annotations"], config_dir)
        stimuli = discover_files(input_config["stimuli"], config_dir)
        description["annotation_inputs"] = [str(path) for path in annotations]
        description["stimulus_inputs"] = [str(path) for path in stimuli]
    else:
        description["inputs"] = [
            str(path) for path in discover_files(input_config, config_dir)
        ]
    return description


class AudioPipeline:
    """Execute the three supported audio feature processes."""

    def __init__(self, config_dir: Path, default_tr_s: float) -> None:
        """Store the YAML base directory and pipeline-level default TR."""
        self.config_dir = config_dir
        self.default_tr_s = default_tr_s

    def run(self, name: str, process: dict[str, Any]) -> list[Path]:
        """Run one audio process using its input, output, and settings mappings."""
        from scripts.audio.preproc import (
            AudioFeaturePreprocessor,
            AudioPhonemePreprocessor,
            AudioVectorizer,
        )

        input_spec = process["input"]
        input_dir = resolve_path(input_spec["directory"], self.config_dir)
        output_dir = resolve_path(process["output"]["directory"], self.config_dir)
        filename_template = process["output"].get("file_template", "{title}.json")
        pattern = input_spec.get("pattern", "*.wav")
        settings = process.get("settings", {})
        tr_s = float(settings.get("tr_s", self.default_tr_s))

        if name == "audio_features":
            processor = AudioFeaturePreprocessor(
                input_dir,
                output_dir,
                features=settings["features"],
                tr_s=tr_s,
                pattern=pattern,
                filename_template=filename_template,
            )
        elif name == "acoustic_phonetics":
            processor = AudioPhonemePreprocessor(
                input_dir,
                output_dir,
                tr_s=tr_s,
                pattern=pattern,
                extractor_options=settings.get("extractor_options"),
                filename_template=filename_template,
            )
        elif name == "audio_embeddings":
            processor = AudioVectorizer(
                input_dir,
                output_dir,
                tr_s=tr_s,
                pattern=pattern,
                embedding_options=settings.get("embedding_options"),
                filename_template=filename_template,
            )
        else:
            raise ValueError(f"Unsupported audio process: {name}")
        return processor.process()


class TextPipeline:
    """Execute transcription and stimulus-aligned text embedding processes."""

    def __init__(self, config_dir: Path, default_tr_s: float) -> None:
        """Store the YAML base directory and pipeline-level default TR."""
        self.config_dir = config_dir
        self.default_tr_s = default_tr_s

    def run_transcripts(self, process: dict[str, Any]) -> list[Path]:
        """Transcribe every configured audio input into a word-level CSV."""
        from scripts.text.preproc import TranscriptGenerator

        inputs = require_inputs(
            discover_files(process["input"], self.config_dir), "transcripts"
        )
        output_dir = resolve_path(process["output"]["directory"], self.config_dir)
        options = dict(process.get("settings", {}))
        options["output_filename_template"] = process["output"].get(
            "file_template", "{stimulus}-annotations.csv"
        )
        transcriber = TranscriptGenerator(output_path=output_dir, **options)
        try:
            return [Path(transcriber.transcribe_audio(path)) for path in inputs]
        finally:
            transcriber.unload_models()

    def _match_annotations_to_stimuli(
        self, process: dict[str, Any]
    ) -> list[tuple[Path, Path]]:
        """Pair annotations and audio by filename stems using configured suffixes."""
        input_config = process["input"]
        annotations = require_inputs(
            discover_files(input_config["annotations"], self.config_dir),
            "text_embeddings annotations",
        )
        stimuli = require_inputs(
            discover_files(input_config["stimuli"], self.config_dir),
            "text_embeddings stimuli",
        )
        matching = input_config.get("matching", {})
        annotation_suffix = str(matching.get("annotation_suffix", "-annotations"))
        case_sensitive = bool(matching.get("case_sensitive", False))

        def key(value: str) -> str:
            return value if case_sensitive else value.casefold()

        stimulus_by_stem = {key(path.stem): path for path in stimuli}
        pairs: list[tuple[Path, Path]] = []
        unmatched: list[Path] = []
        for annotation in annotations:
            stem = annotation.stem
            if annotation_suffix and key(stem).endswith(key(annotation_suffix)):
                stem = stem[: -len(annotation_suffix)]
            stimulus = stimulus_by_stem.get(key(stem))
            if stimulus is None:
                unmatched.append(annotation)
            else:
                pairs.append((annotation, stimulus))
        if unmatched:
            names = ", ".join(path.name for path in unmatched)
            raise FileNotFoundError(f"No matching stimulus audio for: {names}")
        return pairs

    def run_embeddings(self, process: dict[str, Any]) -> list[Path]:
        """Embed matched annotations using frame counts inferred from their audio."""
        from scripts.text.preproc import TextEmbedderGrid

        settings = process.get("settings", {})
        tr_s = float(settings.get("tr_s", self.default_tr_s))
        output_dir = resolve_path(process["output"]["directory"], self.config_dir)
        filename_template = process["output"].get("file_template", "{title}.json")
        return [
            TextEmbedderGrid(
                annotation_path=annotation,
                stimulus_audio=stimulus,
                output_dir=output_dir,
                tr_s=tr_s,
                model_name=settings["model_name"],
                context_window=int(settings.get("context_window", 3)),
                output_filename_template=filename_template,
                strict_annotations=bool(settings.get("strict_annotations", False)),
            ).process()
            for annotation, stimulus in self._match_annotations_to_stimuli(process)
        ]


class DefaultPipeline:
    """Coordinate enabled preprocessing processes in dependency-safe order."""

    def __init__(self, config: dict[str, Any], config_dir: Path) -> None:
        """Create modality pipelines from a validated configuration mapping."""
        self.config = config
        self.config_dir = config_dir
        self.default_tr_s = float(config.get("defaults", {}).get("tr_s", 1.0))
        if self.default_tr_s <= 0:
            raise ValueError("defaults.tr_s must be greater than zero.")
        self.audio = AudioPipeline(config_dir, self.default_tr_s)
        self.text = TextPipeline(config_dir, self.default_tr_s)

    def run(self, selected: set[str] | None = None) -> dict[str, list[str]]:
        """Run enabled (and optionally selected) processes and return their outputs.

        Args:
            selected: Optional process-name allowlist supplied by ``--only``.
                Explicitly selected processes run even when YAML ``enabled`` is false.
        """
        unknown = (selected or set()) - set(PROCESS_ORDER)
        if unknown:
            raise ValueError(f"Unknown process name(s): {', '.join(sorted(unknown))}")

        created: dict[str, list[str]] = {}
        for name in PROCESS_ORDER:
            process = self.config["processes"].get(name)
            if process is None:
                continue
            should_run = name in selected if selected is not None else process.get("enabled", False)
            if not should_run:
                continue
            logging.info("Running process: %s", name)
            if name in {"audio_features", "acoustic_phonetics", "audio_embeddings"}:
                outputs = self.audio.run(name, process)
            elif name == "transcripts":
                outputs = self.text.run_transcripts(process)
            else:
                outputs = self.text.run_embeddings(process)
            created[name] = [str(path) for path in outputs]
        return created


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for configuration, selection, and dry-run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline YAML file."
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=PROCESS_ORDER,
        help="Run only this process; repeat to select multiple processes.",
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Resolve and print I/O without running."
    )
    return parser.parse_args()


def main() -> int:
    """Load the YAML and execute or display the selected preprocessing plan."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config, config_dir = load_config(args.config)
    selected = set(args.only) if args.only else None

    if args.dry_run:
        plans = []
        for name in PROCESS_ORDER:
            process = config["processes"].get(name)
            if process is None:
                continue
            if selected is not None and name not in selected:
                continue
            plans.append(
                process_plan(
                    name,
                    process,
                    config_dir,
                    float(config.get("defaults", {}).get("tr_s", 1.0)),
                )
            )
        print(json.dumps(plans, indent=2))
        return 0

    outputs = DefaultPipeline(config, config_dir).run(selected)
    print(json.dumps(outputs, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
