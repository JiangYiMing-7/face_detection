"""演示和评估入口复用的检测器适配层。"""

from __future__ import annotations

from pathlib import Path

import cv2

from .cascade import CascadeClassifier
from .cascade_paths import haarcascade_path
from .nms import non_max_suppression
from .sliding_window import detect_multiscale


def _expand_boxes(
    boxes: list[tuple[int, int, int, int]],
    frame_shape: tuple,
    pad_x_ratio: float = 0.06,
    pad_top_ratio: float = 0.05,
    pad_bottom_ratio: float = 0.18,
) -> list[tuple[int, int, int, int]]:
    """把自实现检测器偏紧的检测框扩展为更完整的人脸区域。

    pad_* 比例默认与评估保持一致；演示界面可传入更大的下扩比例框住下巴。
    """
    height, width = frame_shape[:2]
    expanded = []
    for x, y, w, h in boxes:
        pad_x = int(round(w * pad_x_ratio))
        pad_top = int(round(h * pad_top_ratio))
        pad_bottom = int(round(h * pad_bottom_ratio))
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_top)
        x2 = min(width, x + w + pad_x)
        y2 = min(height, y + h + pad_bottom)
        expanded.append((x1, y1, x2 - x1, y2 - y1))
    return expanded


class OpenCVHaarDetector:
    """OpenCV 预训练 Haar Cascade 基线检测器。"""

    name = "opencv"

    def __init__(self) -> None:
        """加载 OpenCV 自带的正脸 Haar cascade XML 文件。"""
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
        """运行 OpenCV 基线检测器，返回灰度图和去重后的人脸框。"""
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
        """从磁盘加载本项目训练出的 JSON cascade 模型。"""
        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Custom cascade model not found: {self.model_path}. "
                "Run train.py first or use --detector opencv."
            )
        self.cascade = CascadeClassifier.load(self.model_path)

    def detect(self, frame, options: dict) -> tuple:
        """运行自训练 cascade，并按演示界面参数做预处理和后处理。"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if options.get("clahe"):
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
        elif options.get("equalize"):
            gray = cv2.equalizeHist(gray)
        faces = detect_multiscale(
            gray,
            self.cascade,
            scale_factor=float(options.get("scale_factor", 1.2)),
            step=int(options.get("window_step", 4)),
            min_size=int(options.get("min_size", 30)),
            nms_threshold=float(options.get("nms_threshold", 0.3)),
            min_neighbors=int(options.get("min_neighbors", 0)),
            score_threshold=float(options.get("score_threshold", 0.0)),
            variance_normalize=bool(options.get("variance_normalize", True)),
        )
        faces = _expand_boxes(
            faces,
            frame.shape,
            pad_x_ratio=float(options.get("pad_x_ratio", 0.06)),
            pad_top_ratio=float(options.get("pad_top_ratio", 0.05)),
            pad_bottom_ratio=float(options.get("pad_bottom_ratio", 0.18)),
        )
        return gray, faces


def create_detector(name: str, model_path: str | Path = "models/custom_cascade.json"):
    """按名称创建 OpenCV 基线检测器或自训练 cascade 检测器。"""
    if name == "opencv":
        return OpenCVHaarDetector()
    if name == "custom":
        return CustomCascadeDetector(model_path)
    raise ValueError(f"Unknown detector: {name}")
