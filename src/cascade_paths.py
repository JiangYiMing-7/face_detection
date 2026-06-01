"""跨环境定位 OpenCV Haar cascade XML 文件的辅助函数。"""

from __future__ import annotations

from pathlib import Path
import sys

import cv2


def haarcascade_dir() -> Path:
    """在 pip、conda 等不同安装方式下查找 OpenCV Haar cascade 目录。"""
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
    """返回指定 OpenCV Haar cascade XML 文件的绝对路径。"""
    path = haarcascade_dir() / filename
    if not path.exists():
        raise RuntimeError(f"OpenCV Haar cascade file not found: {path}")
    return str(path)
