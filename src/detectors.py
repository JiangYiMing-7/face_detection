from __future__ import annotations

from pathlib import Path

import cv2

from .cascade import CascadeClassifier
from .cascade_paths import haarcascade_path
from .nms import non_max_suppression
from .sliding_window import detect_multiscale


class OpenCVHaarDetector:
    """OpenCV 预训练 Haar Cascade 基线检测器。"""

    name = "opencv"

    def __init__(self) -> None:
        paths = [
            haarcascade_path("haarcascade_frontalface_default.xml"),
            haarcascade_path("haarcascade_frontalface_alt2.xml"),
        ]
        self.classifiers = []
        for path in paths:
            classifier = cv2.CascadeClassifier(path)
            if classifier.empty():
                raise RuntimeError(f"Failed to load Haar cascade: {path}")
            self.classifiers.append(classifier)

    def detect(self, frame, options: dict) -> tuple:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if options.get("equalize"):
            gray = cv2.equalizeHist(gray)

        rectangles = []
        for classifier in self.classifiers:
            faces = classifier.detectMultiScale(
                gray,
                scaleFactor=float(options.get("scale_factor", 1.1)),
                minNeighbors=int(options.get("min_neighbors", 4)),
                minSize=(int(options.get("min_size", 30)), int(options.get("min_size", 30))),
                flags=cv2.CASCADE_SCALE_IMAGE,
            )
            rectangles.extend(tuple(int(value) for value in face) for face in faces)
        return gray, [tuple(map(int, rect[:4])) for rect in non_max_suppression(rectangles, threshold=0.25)]


class CustomCascadeDetector:
    """加载本项目训练出的手写 Viola-Jones Cascade 模型。"""

    name = "custom"

    def __init__(self, model_path: str | Path) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Custom cascade model not found: {self.model_path}. "
                "Run train.py first or use --detector opencv."
            )
        self.cascade = CascadeClassifier.load(self.model_path)

    def detect(self, frame, options: dict) -> tuple:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if options.get("equalize"):
            gray = cv2.equalizeHist(gray)
        faces = detect_multiscale(
            gray,
            self.cascade,
            scale_factor=float(options.get("scale_factor", 1.2)),
            step=int(options.get("window_step", 4)),
            min_size=int(options.get("min_size", 30)),
            nms_threshold=float(options.get("nms_threshold", 0.3)),
        )
        return gray, faces


def create_detector(name: str, model_path: str | Path = "models/custom_cascade.json"):
    if name == "opencv":
        return OpenCVHaarDetector()
    if name == "custom":
        return CustomCascadeDetector(model_path)
    raise ValueError(f"Unknown detector: {name}")
