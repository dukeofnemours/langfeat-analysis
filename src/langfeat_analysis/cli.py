"""Command-line interface for the LangFeat batch preprocessing pipeline."""

from __future__ import annotations

import argparse
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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse batch, reporting, and process-selection arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG, help="Pipeline YAML file."
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
