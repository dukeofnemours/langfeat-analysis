#!/usr/bin/env python3
"""Compatibility entry point; prefer the installed ``lafa`` CLI."""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from langfeat_analysis.cli import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
