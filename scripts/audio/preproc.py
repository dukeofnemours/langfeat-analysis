"""Compatibility imports for relocated audio preprocessing components."""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[2] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from langfeat_analysis.preprocessing.audio import *  # noqa: F401,F403,E402
