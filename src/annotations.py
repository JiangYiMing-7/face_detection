"""Annotation loading utilities for evaluation datasets."""

from __future__ import annotations

import json
from pathlib import Path


def load_annotations(path: str | Path) -> dict[str, list[tuple[int, int, int, int]]]:
    """Load simple face-box annotations from JSON.

    Supported formats:
    1. {"image.jpg": [[x, y, w, h], ...]}
    2. [{"image": "image.jpg", "boxes": [[x, y, w, h], ...]}, ...]
    """
    annotation_path = Path(path)
    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        items = data.items()
    elif isinstance(data, list):
        items = ((item["image"], item.get("boxes", [])) for item in data)
    else:
        raise ValueError("Unsupported annotation JSON format")

    annotations: dict[str, list[tuple[int, int, int, int]]] = {}
    for image_name, boxes in items:
        annotations[str(image_name)] = [tuple(int(value) for value in box[:4]) for box in boxes]
    return annotations
