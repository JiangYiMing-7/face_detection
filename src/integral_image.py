"""用于 O(1) 计算 Haar 矩形区域和的积分图工具。"""

from __future__ import annotations

import numpy as np


def compute_integral_image(image: np.ndarray) -> np.ndarray:
    """计算单张灰度图的积分图。

    返回结果多补一行一列 0，后续矩形求和时不需要单独处理边界。
    """
    gray = np.asarray(image, dtype=np.float64)
    if gray.ndim != 2:
        raise ValueError("compute_integral_image expects a 2D grayscale image")
    integral = gray.cumsum(axis=0).cumsum(axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


def compute_integral_image_sq(image: np.ndarray) -> np.ndarray:
    """计算单张灰度图的像素平方积分图。

    用于在滑窗检测时 O(1) 计算每个窗口的方差，实现与训练时一致的方差归一化。
    Var(window) = E[x²] - E[x]²，两个积分图各查一次即可。
    """
    gray = np.asarray(image, dtype=np.float64)
    if gray.ndim != 2:
        raise ValueError("compute_integral_image_sq expects a 2D grayscale image")
    sq = gray ** 2
    integral = sq.cumsum(axis=0).cumsum(axis=1)
    return np.pad(integral, ((1, 0), (1, 0)), mode="constant")


def compute_integral_images(images: np.ndarray) -> np.ndarray:
    """批量计算训练窗口的积分图。"""
    batch = np.asarray(images, dtype=np.float64)
    if batch.ndim != 3:
        raise ValueError("compute_integral_images expects shape (n, h, w)")
    integrals = batch.cumsum(axis=1).cumsum(axis=2)
    return np.pad(integrals, ((0, 0), (1, 0), (1, 0)), mode="constant")


def rect_sum(integral: np.ndarray, x: int, y: int, width: int, height: int) -> float:
    """用积分图在 O(1) 时间内计算一个矩形区域的像素和。"""
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + int(width)
    y2 = y1 + int(height)
    return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])


def batch_rect_sum(
    integrals: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
) -> np.ndarray:
    """在一批积分图上批量计算同一矩形区域的像素和。"""
    x1 = int(x)
    y1 = int(y)
    x2 = x1 + int(width)
    y2 = y1 + int(height)
    return integrals[:, y2, x2] - integrals[:, y1, x2] - integrals[:, y2, x1] + integrals[:, y1, x1]


def rect_sums_at(
    integral: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    width: int,
    height: int,
) -> np.ndarray:
    """在同一张积分图的多个位置批量计算矩形区域和。"""
    x1 = np.asarray(xs, dtype=np.int64)
    y1 = np.asarray(ys, dtype=np.int64)
    x2 = x1 + int(width)
    y2 = y1 + int(height)
    return integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1]
