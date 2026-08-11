"""Command-line interface for the LangFeat batch preprocessing pipeline."""

from __future__ import annotations

import argparse
import copy
import json
import logging
import sys
from pathlib import Path
from typing import Any

from langfeat_analysis.pipeline.config import (
    PROCESS_ORDER,
    discover_files,
    load_config,
    resolve_path,
    selected_processes,
)
from langfeat_analysis.pipeline.errors import ConfigurationError, PipelineError
from langfeat_analysis.pipeline.runner import BatchPipeline
from langfeat_analysis.registry import EventRegistry
from langfeat_analysis.io import atomic_json_dump


DEFAULT_CONFIG = Path.cwd() / "configs" / "preproc.yaml"
OUTPUT_SUBDIRECTORIES = {
    "audio_features": Path("audio/features"),
    "acoustic_phonetics": Path("audio/phonetics"),
    "audio_embeddings": Path("audio/embeddings"),
    "transcripts": Path("transcripts"),
    "text_embeddings": Path("text/embeddings"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse batch, reporting, and process-selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline YAML file."
    )
    parser.add_argument(
        "-i",
        "--input-dir",
        type=Path,
        help=(
            "Input storage root. Uses INPUT/audio and INPUT/text when those "
            "directories exist; otherwise uses INPUT as a flat directory."
        ),
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        help="Output root for all process-specific results and logs.",
    )
    parser.add_argument(
        "--audio-input-dir",
        type=Path,
        help="Override the audio directory inferred from --input-dir.",
    )
    parser.add_argument(
        "--annotation-input-dir",
        type=Path,
        help="Override the annotation directory inferred from --input-dir.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        help="Override the registry directory (defaults to OUTPUT/logs with -o).",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=PROCESS_ORDER,
        help="Run only this process; repeat to select several processes.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and display resolved I/O without model loading.",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="Stop the batch after the first failed item.",
    )
    parser.add_argument(
        "--allow-partial-success",
        action="store_true",
        help="Return exit code 0 when some batch items fail; failures remain logged.",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Also write the final machine-readable report to this JSON file.",
    )
    return parser.parse_args(argv)


def _modality_directory(root: Path, modality: str) -> Path:
    """Use ``root/modality`` when present, otherwise retain a flat root."""
    candidate = root / modality
    return candidate if candidate.is_dir() else root


def apply_directory_overrides(
    config: dict[str, Any],
    config_dir: Path,
    *,
    input_dir: Path | None = None,
    output_dir: Path | None = None,
    audio_input_dir: Path | None = None,
    annotation_input_dir: Path | None = None,
    log_dir: Path | None = None,
) -> dict[str, Any]:
    """Apply CLI path overrides without modifying the loaded YAML mapping.

    ``input_dir`` follows the layout produced by ``collect-stimuli.sh``:
    ``INPUT/audio`` contains stimuli and ``INPUT/text`` contains annotations.
    Explicit modality arguments take precedence over this inference.
    """
    updated = copy.deepcopy(config)
    processes = updated["processes"]

    input_root = input_dir.expanduser().resolve() if input_dir else None
    audio_dir = (
        audio_input_dir.expanduser().resolve()
        if audio_input_dir
        else _modality_directory(input_root, "audio")
        if input_root
        else None
    )
    annotation_dir = (
        annotation_input_dir.expanduser().resolve()
        if annotation_input_dir
        else _modality_directory(input_root, "text")
        if input_root
        else None
    )

    if audio_dir is not None:
        for name in ("audio_features", "acoustic_phonetics", "audio_embeddings", "transcripts"):
            if name in processes:
                processes[name]["input"] = {
                    "directory": str(audio_dir),
                    "patterns": ["*.wav", "*.WAV"],
                }
        if "text_embeddings" in processes:
            processes["text_embeddings"]["input"]["stimuli"] = {
                "directory": str(audio_dir),
                "patterns": ["*.wav", "*.WAV"],
            }

    if annotation_dir is not None and "text_embeddings" in processes:
        processes["text_embeddings"]["input"]["annotations"] = {
            "directory": str(annotation_dir),
            "patterns": ["*.csv", "*.CSV", "*.tsv", "*.TSV", "*.textgrid", "*.TextGrid"],
        }
        processes["text_embeddings"]["input"].setdefault("matching", {})[
            "strategy"
        ] = "auto"

    resolved_output = output_dir.expanduser().resolve() if output_dir else None
    if resolved_output is not None:
        old_transcript_output = None
        old_annotation_input = None
        if "transcripts" in processes:
            old_transcript_output = resolve_path(
                processes["transcripts"]["output"]["directory"], config_dir
            )
        if "text_embeddings" in processes and input_root is None and annotation_input_dir is None:
            old_annotation_input = resolve_path(
                processes["text_embeddings"]["input"]["annotations"]["directory"],
                config_dir,
            )

        for name, relative in OUTPUT_SUBDIRECTORIES.items():
            if name in processes:
                processes[name]["output"]["directory"] = str(
                    resolved_output / relative
                )

        # Preserve an existing transcript -> text-embedding dependency when
        # only the output root changes.
        if (
            old_transcript_output is not None
            and old_annotation_input == old_transcript_output
            and "text_embeddings" in processes
        ):
            processes["text_embeddings"]["input"]["annotations"]["directory"] = str(
                resolved_output / OUTPUT_SUBDIRECTORIES["transcripts"]
            )
        updated.setdefault("logging", {})["directory"] = str(resolved_output / "logs")

    if log_dir is not None:
        updated.setdefault("logging", {})["directory"] = str(
            log_dir.expanduser().resolve()
        )
    return updated


def build_plan(
    config: dict[str, Any], config_dir: Path, selected: list[str] | None
) -> list[dict[str, Any]]:
    """Resolve configured inputs for a no-write dry run."""
    plan: list[dict[str, Any]] = []
    for name in selected_processes(config, selected):
        process = config["processes"][name]
        output_dir = resolve_path(process["output"]["directory"], config_dir)
        item: dict[str, Any] = {
            "process": name,
            "output_directory": str(output_dir),
            "file_template": process["output"].get("file_template", ""),
        }
        if name == "text_embeddings":
            input_config = process["input"]
            item["annotations"] = [
                str(path)
                for path in discover_files(
                    input_config["annotations"],
                    config_dir,
                    label="processes.text_embeddings.input.annotations",
                    allow_empty=True,
                )
            ]
            item["stimuli"] = [
                str(path)
                for path in discover_files(
                    input_config["stimuli"],
                    config_dir,
                    label="processes.text_embeddings.input.stimuli",
                )
            ]
        else:
            item["inputs"] = [
                str(path)
                for path in discover_files(
                    process["input"],
                    config_dir,
                    label=f"processes.{name}.input",
                )
            ]
        plan.append(item)
    return plan


def create_registry(config: dict[str, Any], config_dir: Path) -> EventRegistry:
    """Build the run registry from the YAML logging section."""
    settings = config.get("logging", {})
    if not isinstance(settings, dict):
        raise ConfigurationError("'logging' must be a mapping.")
    log_dir = resolve_path(settings.get("directory", "../logs"), config_dir)
    return EventRegistry(
        log_dir,
        filename=settings.get("filename"),
        include_traceback=bool(settings.get("include_traceback", True)),
        fsync_on_write=bool(settings.get("fsync_on_write", False)),
    )


def _write_report(path: Path, report: dict[str, Any]) -> None:
    """Write a final JSON report, creating its parent directory safely."""
    destination = path.expanduser().resolve()
    atomic_json_dump(destination, report, indent=2)


def main(argv: list[str] | None = None) -> int:
    """Validate configuration, run the batch, and return automation-safe status."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    try:
        config, config_dir = load_config(args.config)
        config = apply_directory_overrides(
            config,
            config_dir,
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            audio_input_dir=args.audio_input_dir,
            annotation_input_dir=args.annotation_input_dir,
            log_dir=args.log_dir,
        )
        if args.dry_run:
            print(json.dumps(build_plan(config, config_dir, args.only), indent=2))
            return 0

        if args.fail_fast:
            config.setdefault("batch", {})["continue_on_error"] = False
        registry = create_registry(config, config_dir)
        report = BatchPipeline(config, config_dir, registry).run(args.only)
        payload = report.to_dict()
        if args.report:
            _write_report(args.report, payload)
        print(json.dumps(payload, indent=2))
        return 0 if report.failed_items == 0 or args.allow_partial_success else 1
    except PipelineError as error:
        logging.error("%s", error)
        return 2
    except (OSError, ValueError) as error:
        logging.error("Unable to initialize pipeline: %s", error)
        return 2
    except KeyboardInterrupt:
        logging.error("Pipeline interrupted by user.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
