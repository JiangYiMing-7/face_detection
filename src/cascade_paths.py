from __future__ import annotations

from pathlib import Path
import sys

import cv2


def haarcascade_dir() -> Path:
    """Find OpenCV's Haar cascade directory across pip and conda builds."""
    data = getattr(cv2, "data", None)
    if data is not None and getattr(data, "haarcascades", None):
        candidate = Path(data.haarcascades)
        if candidate.exists():
            return candidate

    candidates = []
    prefixes = {Path(sys.prefix)}
    for prefix in prefixes:
        candidates.extend(
            [
                prefix / "share" / "opencv4" / "haarcascades",
                prefix / "share" / "opencv" / "haarcascades",
                prefix / "Library" / "etc" / "haarcascades",
            ]
        )
    cv2_file = getattr(cv2, "__file__", None)
    if cv2_file:
        cv2_path = Path(cv2_file).resolve()
        candidates.extend(
            [
                cv2_path.parent / "data",
                cv2_path.parent / "data" / "haarcascades",
            ]
        )

    for candidate in candidates:
        if (candidate / "haarcascade_frontalface_default.xml").exists():
            return candidate

    raise RuntimeError("Could not find OpenCV haarcascades directory")


def haarcascade_path(filename: str) -> str:
    path = haarcascade_dir() / filename
    if not path.exists():
        raise RuntimeError(f"OpenCV Haar cascade file not found: {path}")
    return str(path)

