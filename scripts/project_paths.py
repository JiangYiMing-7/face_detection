"""Shared path helpers for one-off scripts."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def project_path(path: str | Path) -> Path:
    """Resolve relative paths from the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate
