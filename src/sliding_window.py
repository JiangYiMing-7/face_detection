from __future__ import annotations

import cv2
import numpy as np

from .cascade import CascadeClassifier
from .integral_image import compute_integral_image
from .nms import non_max_suppression


def _predict_windows(cascade: CascadeClassifier, integral: np.ndarray, xs: np.ndarray, ys: np.ndarray):
    """批量预测同一尺度下的多个滑窗。

    实时检测时窗口数量很多，所以这里按 stage 批量计算，避免每个窗口
    都单独走一遍 Python 循环导致速度过慢。
    """
    alive = np.ones(len(xs), dtype=bool)
    scores = np.zeros(len(xs), dtype=np.float64)
    for stage in cascade.stages:
        alive_indices = np.flatnonzero(alive)
        if alive_indices.size == 0:
            break
        stage_scores = np.zeros(alive_indices.size, dtype=np.float64)
        active_xs = xs[alive_indices]
        active_ys = ys[alive_indices]
        for weak in stage.weak_classifiers:
            values = weak.feature.values_at(integral, active_xs, active_ys)
            stage_scores += weak.alpha * weak.predict_values(values)
        scores[alive_indices] = stage_scores
        alive[alive_indices[stage_scores < stage.threshold]] = False
    return alive, scores


def detect_multiscale(
    gray: np.ndarray,
    cascade: CascadeClassifier,
    scale_factor: float = 1.2,
    step: int = 4,
    min_size: int = 30,
    nms_threshold: float = 0.3,
) -> list[tuple[int, int, int, int]]:
    """在图像金字塔上用手写 cascade 进行多尺度人脸检测。"""
    if scale_factor <= 1.0:
        raise ValueError("scale_factor must be greater than 1.0")
    if step < 1:
        raise ValueError("step must be at least 1")

    h, w = gray.shape[:2]
    window = int(cascade.window_size)
    boxes: list[tuple[int, int, int, int, float]] = []
    scale = max(float(min_size) / float(window), 1.0)

    while True:
        resized_w = int(round(w / scale))
        resized_h = int(round(h / scale))
        if resized_w < window or resized_h < window:
            break

        resized = cv2.resize(gray, (resized_w, resized_h), interpolation=cv2.INTER_AREA)
        integral = compute_integral_image(resized)
        ys_grid, xs_grid = np.mgrid[0 : resized_h - window + 1 : step, 0 : resized_w - window + 1 : step]
        xs = xs_grid.ravel().astype(np.int64)
        ys = ys_grid.ravel().astype(np.int64)
        passed, scores = _predict_windows(cascade, integral, xs, ys)
        for x, y, score in zip(xs[passed], ys[passed], scores[passed], strict=False):
            boxes.append(
                (
                    int(round(int(x) * scale)),
                    int(round(int(y) * scale)),
                    int(round(window * scale)),
                    int(round(window * scale)),
                    float(score),
                )
            )
        scale *= scale_factor

    return [tuple(map(int, box[:4])) for box in non_max_suppression(boxes, threshold=nms_threshold)]
