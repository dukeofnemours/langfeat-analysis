"""Safe output helpers that avoid leaving truncated final files."""

from __future__ import annotations

import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator, TextIO


def safe_output_path(output_dir: str | Path, filename: str) -> Path:
    """Return an output path only when ``filename`` is a plain basename."""
    if not filename or Path(filename).name != filename:
        raise ValueError(
            f"Output filename template rendered an unsafe path: {filename!r}. "
            "Templates may create filenames, not subdirectories."
        )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    return destination / filename


@contextmanager
def atomic_text_writer(
    destination: str | Path, *, newline: str | None = None
) -> Iterator[TextIO]:
    """Write beside ``destination`` and atomically replace it on success."""
    path = Path(destination)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline=newline,
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as file:
            temporary_path = Path(file.name)
            yield file
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def atomic_json_dump(destination: str | Path, value: Any, *, indent: int | None = None) -> None:
    """Serialize JSON through an atomic temporary-file replacement."""
    with atomic_text_writer(destination) as file:
        json.dump(value, file, ensure_ascii=False, indent=indent)
        file.write("\n")
