"""Download or verify model assets before pipeline initialization."""

from __future__ import annotations

import importlib.util
import os
import platform
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from langfeat_analysis.pipeline.config import discover_files, resolve_path, selected_processes
from langfeat_analysis.pipeline.errors import ProcessError
from langfeat_analysis.pipeline.progress import TerminalReporter
from langfeat_analysis.registry import EventRegistry


@dataclass(frozen=True)
class ModelAsset:
    """One model repository or bundled weight file required by a process."""

    process: str
    model: str
    provider: str = "huggingface"


@dataclass(frozen=True)
class CachedModel:
    """Serializable result for one downloaded or verified model asset."""

    process: str
    model: str
    provider: str
    status: str
    path: str


def configure_model_cache_environment(
    config: dict[str, Any], config_dir: Path, *, offline: bool | None = None
) -> Path | None:
    """Apply configured cache location and offline flags before model imports."""
    settings = config.get("models", {})
    directory = settings.get("cache_directory")
    cache_root: Path | None = None
    if directory:
        candidate = Path(directory).expanduser()
        cache_root = (
            candidate.resolve()
            if candidate.is_absolute()
            else (config_dir / candidate).resolve()
        )
        cache_root.mkdir(parents=True, exist_ok=True)
        os.environ["HF_HOME"] = str(cache_root)
    offline_enabled = bool(settings.get("offline", False)) if offline is None else offline
    if offline_enabled:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    return cache_root


def required_model_assets(
    config: dict[str, Any],
    selected: Sequence[str] | None = None,
    config_dir: Path | None = None,
) -> list[ModelAsset]:
    """Return de-duplicated model assets for the selected enabled processes."""
    names = selected_processes(config, selected)
    assets: list[ModelAsset] = []
    processes = config["processes"]

    if "acoustic_phonetics" in names:
        options = processes["acoustic_phonetics"].get("settings", {}).get(
            "extractor_options", {}
        )
        if options.get("posterior_backend", "ctc") == "ctc":
            assets.append(
                ModelAsset(
                    "acoustic_phonetics",
                    str(
                        options.get(
                            "ctc_model",
                            "bobboyms/wav2vec2-base-en-phoneme-ctc-41h",
                        )
                    ),
                )
            )

    if "audio_embeddings" in names:
        options = processes["audio_embeddings"].get("settings", {}).get(
            "embedding_options", {}
        )
        identifier = (
            f"{options.get('input_repr', 'linear')}:"
            f"{options.get('content_type', 'env')}"
        )
        assets.append(ModelAsset("audio_embeddings", identifier, "openl3-bundled"))

    if "transcripts" in names and _transcription_models_needed(
        processes["transcripts"], config_dir
    ):
        settings = processes["transcripts"].get("settings", {})
        if _uses_mlx_backend():
            assets.extend(
                [
                    ModelAsset(
                        "transcripts",
                        str(
                            settings.get(
                                "mlx_asr_model",
                                "mlx-community/Qwen3-ASR-0.6B-8bit",
                            )
                        ),
                    ),
                    ModelAsset(
                        "transcripts",
                        str(
                            settings.get(
                                "mlx_aligner_model",
                                "mlx-community/Qwen3-ForcedAligner-0.6B-8bit",
                            )
                        ),
                    ),
                ]
            )
        else:
            assets.extend(
                [
                    ModelAsset(
                        "transcripts",
                        str(
                            settings.get("pytorch_asr_model", "Qwen/Qwen3-ASR-0.6B")
                        ),
                    ),
                    ModelAsset(
                        "transcripts",
                        str(
                            settings.get(
                                "pytorch_aligner_model",
                                "Qwen/Qwen3-ForcedAligner-0.6B",
                            )
                        ),
                    ),
                ]
            )

    if "text_embeddings" in names:
        assets.append(
            ModelAsset(
                "text_embeddings",
                str(processes["text_embeddings"]["settings"]["model_name"]),
            )
        )

    return list(dict.fromkeys(assets))


def cache_models(
    config: dict[str, Any],
    config_dir: Path,
    registry: EventRegistry,
    reporter: TerminalReporter,
    *,
    selected: Sequence[str] | None = None,
    download: bool = True,
) -> list[CachedModel]:
    """Download or locally verify every required model without running inputs."""
    cache_root = configure_model_cache_environment(
        config, config_dir, offline=not download
    )
    results: list[CachedModel] = []
    for asset in required_model_assets(config, selected, config_dir):
        reporter.model_pending(asset.process, asset.model)
        timer = registry.timer()
        registry.emit(
            "model_cache_started",
            "started",
            process=asset.process,
            details={"model": asset.model, "provider": asset.provider, "download": download},
        )
        try:
            path = _cache_asset(asset, config_dir, cache_root, download=download)
        except Exception as error:
            action = "cache" if download else "find in the local cache"
            reporter.model_finished(asset.process, asset.model, "failed", detail=str(error))
            registry.emit(
                "model_cache_failed",
                "failed",
                process=asset.process,
                duration_ms=timer.elapsed_ms,
                error=error,
                details={"model": asset.model, "provider": asset.provider},
            )
            raise ProcessError(
                asset.process,
                None,
                f"unable to {action} model '{asset.model}': {error}",
                hint=(
                    "Run `lafa --cache-models --config CONFIG` while online, then "
                    "reuse the same models.cache_directory in the offline environment."
                ),
            ) from error
        result = CachedModel(
            asset.process, asset.model, asset.provider, "cached", str(path)
        )
        results.append(result)
        reporter.model_finished(
            asset.process, asset.model, "success", detail=f"cached at {path}"
        )
        registry.emit(
            "model_cache_completed",
            "success",
            process=asset.process,
            duration_ms=timer.elapsed_ms,
            details=asdict(result),
        )
    return results


def _cache_asset(
    asset: ModelAsset,
    config_dir: Path,
    cache_root: Path | None,
    *,
    download: bool,
) -> Path:
    """Cache one Hugging Face snapshot or verify one bundled OpenL3 weight."""
    if asset.provider == "openl3-bundled":
        package = importlib.util.find_spec("openl3")
        if package is None or not package.submodule_search_locations:
            raise RuntimeError("OpenL3 is not installed; install the project audio extra.")
        input_repr, content_type = asset.model.split(":", maxsplit=1)
        path = (
            Path(next(iter(package.submodule_search_locations)))
            / f"openl3_audio_{input_repr}_{content_type}.h5"
        )
        if not path.is_file():
            raise FileNotFoundError(f"bundled OpenL3 weights are missing: {path}")
        return path.resolve()

    local = Path(asset.model).expanduser()
    candidates = [local, config_dir / local]
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as error:
        raise RuntimeError(
            "huggingface-hub is required to cache model snapshots."
        ) from error
    hub_cache = cache_root / "hub" if cache_root is not None else None
    snapshot = snapshot_download(
        repo_id=asset.model,
        cache_dir=str(hub_cache) if hub_cache is not None else None,
        local_files_only=not download,
    )
    return Path(snapshot).resolve()


def _uses_mlx_backend() -> bool:
    """Match TranscriptGenerator's Apple-Silicon backend selection."""
    if platform.system() != "Darwin" or platform.machine() not in {"arm64", "aarch64"}:
        return False
    try:
        import torch

        return bool(torch.backends.mps.is_available())
    except ImportError:
        return False


def _transcription_models_needed(
    process: dict[str, Any], config_dir: Path | None
) -> bool:
    """Return false only when every discoverable audio has a validated text match."""
    if config_dir is None:
        return True
    input_config = process["input"]
    audio_spec = input_config.get("audio", input_config)
    try:
        audio_paths = discover_files(
            audio_spec,
            config_dir,
            label="processes.transcripts.input.audio",
        )
    except Exception:
        return True

    transcript_paths: list[Path] = []
    specifications = []
    if isinstance(input_config.get("existing_transcripts"), dict):
        specifications.append(input_config["existing_transcripts"])
    specifications.append(
        {
            "directory": process["output"]["directory"],
            "patterns": [
                "*.csv", "*.CSV", "*.tsv", "*.TSV", "*.txt", "*.TXT",
                "*.textgrid", "*.TextGrid", "*.eaf", "*.EAF",
            ],
        }
    )
    for specification in specifications:
        if "files" not in specification:
            directory = resolve_path(specification.get("directory", ""), config_dir)
            if not directory.is_dir():
                continue
        try:
            transcript_paths.extend(
                discover_files(
                    specification,
                    config_dir,
                    label="transcript cache preflight",
                    allow_empty=True,
                )
            )
        except Exception:
            continue
    from langfeat_analysis.pipeline.runner import select_transcript_matches

    matching = input_config.get("matching", {})
    matches, _ = select_transcript_matches(
        audio_paths,
        sorted(set(transcript_paths)),
        threshold=float(matching.get("threshold", 0.72)),
        min_margin=float(matching.get("min_margin", 0.05)),
        alias_count_heuristic=bool(matching.get("alias_count_heuristic", True)),
        alias_min_score=float(matching.get("alias_min_score", 0.35)),
    )
    return len(matches) != len(audio_paths)
