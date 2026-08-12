"""Exceptions with actionable messages for pipeline users."""

from __future__ import annotations

from pathlib import Path


class PipelineError(RuntimeError):
    """Base error for an expected, user-actionable pipeline failure."""


class ConfigurationError(PipelineError):
    """Raised when YAML structure or a configured value is invalid."""


class InputError(PipelineError):
    """Raised when configured input files cannot be resolved or validated."""


class ProcessError(PipelineError):
    """Describe which process and input failed while preserving the cause."""

    def __init__(
        self,
        process: str,
        input_path: str | Path | None,
        reason: str,
        *,
        hint: str | None = None,
    ) -> None:
        self.process = process
        self.input_path = Path(input_path) if input_path is not None else None
        self.reason = reason
        self.hint = hint
        target = f" for input '{self.input_path}'" if self.input_path else ""
        message = f"Process '{process}' failed{target}: {reason}"
        if hint:
            message += f" Hint: {hint}"
        super().__init__(message)
