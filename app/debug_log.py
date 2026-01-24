"""Debug-mode log path. Uses project root relative to this file."""
from __future__ import annotations

import os

_PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
DEBUG_LOG_PATH = os.path.join(_PROJECT_ROOT, ".cursor", "debug.log")


def ensure_log_dir() -> None:
    os.makedirs(os.path.dirname(DEBUG_LOG_PATH), exist_ok=True)
