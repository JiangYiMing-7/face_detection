"""Haar-like feature definitions, generation, computation, and visualization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .integral_image import batch_rect_sum, rect_sum, rect_sums_at


@dataclass(frozen=True)
class HaarFeature:
    """固定检测窗口中的 Haar-like 矩形特征。

    kind 决定矩形组合方式；白色区域权重为正，黑色区域权重为负。
    特征值就是若干矩形区域像素和的加权差。
    """

    kind: str
    x: int
    y: int
    width: int
    height: int

    def weighted_rectangles(self, offset_x: int = 0, offset_y: int = 0) -> list[tuple[int, int, int, int, float]]:
        """返回该 Haar 特征拆成的若干加权矩形区域。"""
        x = self.x + int(offset_x)
        y = self.y + int(offset_y)
        w = self.width
        h = self.height

        if self.kind == "two_horizontal":
            half = w // 2
            return [(x, y, half, h, 1.0), (x + half, y, half, h, -1.0)]
        if self.kind == "two_vertical":
            half = h // 2
            return [(x, y, w, half, 1.0), (x, y + half, w, half, -1.0)]
        if self.kind == "three_horizontal":
            third = w // 3
            return [
                (x, y, third, h, 1.0),
                (x + third, y, third, h, -1.0),
                (x + 2 * third, y, third, h, 1.0),
            ]
        if self.kind == "three_vertical":
            third = h // 3
            return [
                (x, y, w, third, 1.0),
                (x, y + third, w, third, -1.0),
                (x, y + 2 * third, w, third, 1.0),
            ]
        if self.kind == "four":
            half_w = w // 2
            half_h = h // 2
            return [
                (x, y, half_w, half_h, 1.0),
                (x + half_w, y, half_w, half_h, -1.0),
                (x, y + half_h, half_w, half_h, -1.0),
                (x + half_w, y + half_h, half_w, half_h, 1.0),
            ]
        raise ValueError(f"Unknown Haar feature kind: {self.kind}")

    def value(self, integral: np.ndarray, offset_x: int = 0, offset_y: int = 0) -> float:
        """在单张积分图上计算一个窗口位置的特征值。"""
        total = 0.0
        for rx, ry, rw, rh, weight in self.weighted_rectangles(offset_x, offset_y):
            total += weight * rect_sum(integral, rx, ry, rw, rh)
        return total

    def values(self, integrals: np.ndarray) -> np.ndarray:
        """在一批训练窗口的积分图上批量计算该特征。"""
        total = np.zeros(integrals.shape[0], dtype=np.float64)
        for rx, ry, rw, rh, weight in self.weighted_rectangles():
            total += weight * batch_rect_sum(integrals, rx, ry, rw, rh)
        return total

    def values_at(self, integral: np.ndarray, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        """在同一张积分图的多个滑窗位置批量计算该特征。"""
        total = np.zeros(len(xs), dtype=np.float64)
        for rx, ry, rw, rh, weight in self.weighted_rectangles():
            total += weight * rect_sums_at(integral, xs + rx, ys + ry, rw, rh)
        return total

    def to_dict(self) -> dict:
        """把 Haar 特征转换成 JSON 可序列化的字典。"""
        return {
            "kind": self.kind,
            "x": self.x,
            "y": self.y,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "HaarFeature":
        """从 JSON 字典恢复 Haar 特征对象。"""
        return cls(
            kind=str(data["kind"]),
            x=int(data["x"]),
            y=int(data["y"]),
            width=int(data["width"]),
            height=int(data["height"]),
        )


def _all_features(window_size: int) -> Iterable[HaarFeature]:
    """枚举 24x24 等固定窗口内所有合法的二/三/四矩形特征。"""
    kinds = (
        ("two_horizontal", 2, 1),
        ("two_vertical", 1, 2),
        ("three_horizontal", 3, 1),
        ("three_vertical", 1, 3),
        ("four", 2, 2),
    )
    for kind, width_divisor, height_divisor in kinds:
        min_w = width_divisor
        min_h = height_divisor
        for width in range(min_w, window_size + 1):
            if width % width_divisor != 0:
                continue
            for height in range(min_h, window_size + 1):
                if height % height_divisor != 0:
                    continue
                for y in range(0, window_size - height + 1):
                    for x in range(0, window_size - width + 1):
                        yield HaarFeature(kind, x, y, width, height)


def generate_haar_features(
    window_size: int = 24,
    max_features: int | None = 8000,
    seed: int = 42,
) -> list[HaarFeature]:
    """生成 Haar 特征集合。

    完整特征数量很大，课程设计版本默认固定随机种子抽取一部分，
    这样既保留特征选择过程，又能把训练时间控制在本机可承受范围内。
    """
    features = list(_all_features(window_size))
    if max_features is None or max_features <= 0 or max_features >= len(features):
        return features
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(features), size=max_features, replace=False)
    indices.sort()
    return [features[int(index)] for index in indices]


def compute_feature_matrix(
    windows: np.ndarray,
    features: list[HaarFeature],
    dtype: np.dtype = np.float32,
) -> np.ndarray:
    """为训练窗口计算特征矩阵：一行一个样本，一列一个 Haar 特征。"""
    from .integral_image import compute_integral_images

    integrals = compute_integral_images(windows)
    matrix = np.empty((len(windows), len(features)), dtype=dtype)
    for column, feature in enumerate(features):
        matrix[:, column] = feature.values(integrals)
    return matrix


def render_feature(
    feature: HaarFeature,
    window_size: int = 24,
    scale: int = 10,
) -> np.ndarray:
    """把 Haar 特征画成小图，用于报告或 PPT 展示最强特征。"""
    import cv2

    canvas = np.full((window_size * scale, window_size * scale, 3), 245, dtype=np.uint8)
    for rx, ry, rw, rh, weight in feature.weighted_rectangles():
        color = (245, 245, 245) if weight > 0 else (60, 60, 60)
        p1 = (rx * scale, ry * scale)
        p2 = ((rx + rw) * scale - 1, (ry + rh) * scale - 1)
        cv2.rectangle(canvas, p1, p2, color, thickness=-1)
        cv2.rectangle(canvas, p1, p2, (0, 0, 220), thickness=1)
    cv2.rectangle(canvas, (0, 0), (window_size * scale - 1, window_size * scale - 1), (0, 0, 0), 1)
    return canvas
