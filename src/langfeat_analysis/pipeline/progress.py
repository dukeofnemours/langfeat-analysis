"""Human-readable terminal feedback for batch pipeline runs."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TextIO


class TerminalReporter:
    """Write concise pending, progress, success, and failure updates to stderr."""

    def __init__(self, stream: TextIO | None = None, *, enabled: bool = True) -> None:
        """Configure the output stream.

        Args:
            stream: Terminal stream for user-facing updates. Defaults to stderr
                so JSON reports on stdout stay machine-readable.
            enabled: Suppress all terminal updates when false.
        """
        self.stream = stream or sys.stderr
        self.enabled = enabled
        self._completed = 0
        self._total = 0
        self._process = ""

    def pipeline_pending(self, processes: list[str]) -> None:
        """Report the selected process list before work begins."""
        self._write(f"[PENDING] Pipeline: {', '.join(processes)}")

    def process_pending(self, process: str) -> None:
        """Report that a process is resolving inputs or initializing models."""
        self._write(f"[PENDING] {process}: resolving inputs and preparing resources")

    def items_pending(self, process: str, total: int) -> None:
        """Initialize a progress bar for a process's independently run items."""
        self._process = process
        self._completed = 0
        self._total = total
        self._write(f"[PENDING] {process}: {total} item(s) queued")
        if total:
            self._write(self._progress_line())

    def item_finished(
        self,
        process: str,
        input_path: str | Path,
        status: str,
        *,
        output_count: int = 0,
        error_message: str = "",
    ) -> None:
        """Advance the bar and report one item's terminal state."""
        if process != self._process:
            self.items_pending(process, 0)
        self._completed += 1
        filename = Path(input_path).name or str(input_path) or "process setup"
        if status == "success":
            suffix = f" ({output_count} output{'s' if output_count != 1 else ''})"
            self._write(f"[SUCCESS] {process}: {filename}{suffix}")
        else:
            detail = self._compact_error(error_message)
            self._write(f"[FAILED] {process}: {filename} — {detail}")
        if self._total:
            self._write(self._progress_line())

    def process_finished(
        self,
        process: str,
        succeeded: int,
        failed: int,
        *,
        error_message: str = "",
    ) -> None:
        """Report the aggregate outcome of one process."""
        status = "SUCCESS" if failed == 0 else "FAILED"
        message = f"[{status}] {process}: {succeeded} succeeded, {failed} failed"
        if error_message:
            message += f" — {self._compact_error(error_message)}"
        self._write(message)

    def pipeline_finished(self, succeeded: int, failed: int) -> None:
        """Report the final batch outcome."""
        status = "SUCCESS" if failed == 0 else "FAILED"
        self._write(f"[{status}] Pipeline: {succeeded} succeeded, {failed} failed")

    def pipeline_interrupted(self) -> None:
        """Report an interrupt without pretending that remaining work completed."""
        self._write("[FAILED] Pipeline interrupted by user")

    def _progress_line(self) -> str:
        width = 24
        fraction = self._completed / self._total if self._total else 0.0
        filled = round(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        return (
            f"[PROGRESS] {self._process}: [{bar}] "
            f"{self._completed}/{self._total} ({fraction:.0%})"
        )

    @staticmethod
    def _compact_error(message: str) -> str:
        first_line = message.splitlines()[0].strip() if message else "unknown error"
        return first_line if len(first_line) <= 180 else f"{first_line[:177]}..."

    def _write(self, message: str) -> None:
        if not self.enabled:
            return
        self.stream.write(f"{message}\n")
        self.stream.flush()
