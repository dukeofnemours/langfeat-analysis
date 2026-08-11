"""Concurrency-safe JSON Lines event registry for pipeline runs.

Every call to :meth:`EventRegistry.write` appends exactly one JSON object and
one newline.  A registry file therefore remains streamable even when a run
fails partway through a batch.
"""

from __future__ import annotations

import json
import os
import threading
import traceback as traceback_module
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import Any, Literal, Mapping


EventStatus = Literal["started", "success", "failed", "skipped", "warning"]


def utc_now() -> str:
    """Return an ISO-8601 UTC timestamp suitable for machine parsing."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class Action:
    """Backward-compatible action passed to :func:`register_event`."""

    name: str
    status: str


@dataclass
class Event:
    """One self-contained JSONL event."""

    event: str
    status: EventStatus
    run_id: str
    timestamp: str = field(default_factory=utc_now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    process: str = ""
    input_path: str = ""
    output_path: str = ""
    source_path: str = ""
    function: str = ""
    duration_ms: float | None = None
    error_type: str = ""
    error_message: str = ""
    traceback: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe mapping, omitting empty optional fields."""
        return {
            key: value
            for key, value in asdict(self).items()
            if value not in ("", None, {}, [])
        }


class EventRegistry:
    """Append pipeline events to a daily or explicitly named JSONL file."""

    _thread_lock = threading.Lock()

    def __init__(
        self,
        log_dir: str | Path,
        *,
        run_id: str | None = None,
        filename: str | None = None,
        include_traceback: bool = True,
        fsync_on_write: bool = False,
    ) -> None:
        """Configure the registry.

        Args:
            log_dir: Directory in which the JSONL file is created.
            run_id: Stable identifier shared by all events in one run.
            filename: Optional log filename. Defaults to ``YYYY-MM-DD.jsonl``.
            include_traceback: Include formatted exception tracebacks in failed
                events. Disable when logs may be shared externally.
            fsync_on_write: Force each event through the filesystem cache. This
                is safer during a power loss but can be slow on synced/network
                folders; normal writes are always flushed.
        """
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.run_id = run_id or str(uuid.uuid4())
        self.include_traceback = include_traceback
        self.fsync_on_write = fsync_on_write
        log_name = filename or f"{datetime.now(timezone.utc):%Y-%m-%d}.jsonl"
        if Path(log_name).name != log_name or not log_name.endswith(".jsonl"):
            raise ValueError("Registry filename must be a .jsonl basename.")
        self.path = self.log_dir / log_name

    def write(self, event: Event) -> Path:
        """Append one event as one UTF-8 JSON line and flush it to disk."""
        if event.run_id != self.run_id:
            raise ValueError(
                f"Event run_id {event.run_id!r} does not match registry "
                f"run_id {self.run_id!r}."
            )
        self.log_dir.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(event.to_dict(), ensure_ascii=False, default=str)
        with self._thread_lock:
            with self.path.open("a", encoding="utf-8") as file:
                self._lock_file(file)
                try:
                    file.write(payload)
                    file.write("\n")
                    file.flush()
                    if self.fsync_on_write:
                        os.fsync(file.fileno())
                finally:
                    self._unlock_file(file)
        return self.path

    @staticmethod
    def _lock_file(file: Any) -> None:
        """Use an advisory process lock where the platform provides ``fcntl``."""
        try:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - Windows fallback uses thread lock.
            return

    @staticmethod
    def _unlock_file(file: Any) -> None:
        try:
            import fcntl

            fcntl.flock(file.fileno(), fcntl.LOCK_UN)
        except ImportError:  # pragma: no cover - Windows fallback uses thread lock.
            return

    def emit(
        self,
        event: str,
        status: EventStatus,
        *,
        process: str = "",
        input_path: str | Path | None = None,
        output_path: str | Path | None = None,
        source_path: str | Path | None = None,
        function: str = "",
        duration_ms: float | None = None,
        error: BaseException | None = None,
        details: Mapping[str, Any] | None = None,
    ) -> Event:
        """Construct, append, and return an event.

        Args describe the process/item involved. When ``error`` is supplied its
        concrete type, message, and optional traceback are recorded.
        """
        event_record = Event(
            event=event,
            status=status,
            run_id=self.run_id,
            process=process,
            input_path=str(input_path or ""),
            output_path=str(output_path or ""),
            source_path=str(source_path or ""),
            function=function,
            duration_ms=round(duration_ms, 3) if duration_ms is not None else None,
            error_type=type(error).__name__ if error else "",
            error_message=str(error) if error else "",
            traceback=(
                "".join(
                    traceback_module.format_exception(
                        type(error), error, error.__traceback__
                    )
                )
                if error and self.include_traceback
                else ""
            ),
            details=dict(details or {}),
        )
        self.write(event_record)
        return event_record

    def timer(self) -> "EventTimer":
        """Return a small elapsed-time helper for duration fields."""
        return EventTimer()


class EventTimer:
    """Measure elapsed milliseconds using a monotonic clock."""

    def __init__(self) -> None:
        self.started = perf_counter()

    @property
    def elapsed_ms(self) -> float:
        return (perf_counter() - self.started) * 1000.0


def _default_log_dir() -> Path:
    configured = os.environ.get("LANGFEAT_LOG_DIR")
    if configured:
        return Path(configured)
    return Path(__file__).resolve().parents[2] / "logs"


def register_event(
    path: str,
    action: Action,
    func_name: str,
    out_path: str = "",
    *,
    input_path: str = "",
    process: str = "",
    registry: EventRegistry | None = None,
) -> Event:
    """Compatibility wrapper used by preprocessing components.

    Unlike the old implementation, ``action.status`` is preserved and every
    event receives a run id. New pipeline code should pass its shared registry.
    """
    active = registry or EventRegistry(_default_log_dir())
    status_aliases: dict[str, EventStatus] = {
        "started": "started",
        "success": "success",
        "succeeded": "success",
        "failed": "failed",
        "failure": "failed",
        "error": "failed",
        "skipped": "skipped",
        "warning": "warning",
        "warn": "warning",
    }
    normalized_status = status_aliases.get(action.status.casefold(), "success")
    return active.emit(
        action.name,
        normalized_status,
        process=process,
        input_path=input_path,
        output_path=out_path,
        source_path=path,
        function=func_name,
    )
