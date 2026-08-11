"""Compatibility imports for the pipeline event registry.

New code should import from :mod:`langfeat_analysis.registry`. This module is
kept so older analysis scripts continue to work after the project restructure.
"""

from pathlib import Path
import sys


SOURCE_ROOT = Path(__file__).resolve().parents[1] / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from langfeat_analysis.registry import (
    Action,
    Event,
    EventRegistry,
    EventTimer,
    register_event,
    utc_now,
)

__all__ = [
    "Action",
    "Event",
    "EventRegistry",
    "EventTimer",
    "register_event",
    "utc_now",
]
