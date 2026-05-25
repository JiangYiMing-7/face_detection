from __future__ import annotations

import cv2
import numpy as np

from .cascade import CascadeClassifier
from .integral_image import compute_integral_image, compute_integral_image_sq
from .nms import non_max_suppression


def _window_stds(
    integral: np.ndarray,
    integral_sq: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """用积分图和平方积分图批量计算各滑窗的像素标准差。

    Std = sqrt(E[x²] - E[x]²)，结果用于方差归一化——与训练时 normalize_window 保持一致。
    """
    from .integral_image import rect_sums_at

    n = window_size * window_size
    mean = rect_sums_at(integral, xs, ys, window_size, window_size) / n
    mean_sq = rect_sums_at(integral_sq, xs, ys, window_size, window_size) / n
    var = mean_sq - mean ** 2
    return np.sqrt(np.maximum(var, 1e-8))


def _predict_windows(
    cascade: CascadeClassifier,
    integral: np.ndarray,
    integral_sq: np.ndarray | None,
    xs: np.ndarray,
    ys: np.ndarray,
):
    """批量预测同一尺度下的多个滑窗。

    integral_sq 不为 None 时对每个窗口做方差归一化（论文 Section 2.2），
    使检测时的特征值分布与训练时一致，提升精确率。
    """
    alive = np.ones(len(xs), dtype=bool)
    cumulative_scores = np.zeros(len(xs), dtype=np.float64)

    # 预计算方差归一化因子（每个窗口的像素 std）
    if integral_sq is not None:
        stds = _window_stds(integral, integral_sq, xs, ys, cascade.window_size)
    else:
        stds = None

    for stage in cascade.stages:
        alive_indices = np.flatnonzero(alive)
        if alive_indices.size == 0:
            break
        stage_scores = np.zeros(alive_indices.size, dtype=np.float64)
        active_xs = xs[alive_indices]
        active_ys = ys[alive_indices]
        for weak in stage.weak_classifiers:
            values = weak.feature.values_at(integral, active_xs, active_ys)
            if stds is not None:
                # 特征值除以窗口 std，与训练时 normalize_window 一致
                values = values / stds[alive_indices]
            stage_scores += weak.alpha * weak.predict_values(values)
        cumulative_scores[alive_indices] += stage_scores
        alive[alive_indices[stage_scores < stage.threshold]] = False
    return alive, cumulative_scores


def detect_multiscale(
    gray: np.ndarray,
    cascade: CascadeClassifier,
    scale_factor: float = 1.2,
    step: int = 4,
    min_size: int = 30,
    nms_threshold: float = 0.3,
    min_neighbors: int = 0,
    score_threshold: float = 0.0,
    variance_normalize: bool = True,
) -> list[tuple[int, int, int, int]]:
    """在图像金字塔上用手写 cascade 进行多尺度人脸检测。

    variance_normalize: 启用方差归一化（对应论文 Section 2.2），使检测时
        特征分布与训练时一致，有效降低误报率。
    score_threshold: 级联累积分数必须超过该值才保留检测结果（0 = 不过滤）。
    min_neighbors: 至少需要多少个重叠窗口才算真实人脸（0 = 只做 NMS）。
    """
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
        integral_sq = compute_integral_image_sq(resized) if variance_normalize else None

        ys_grid, xs_grid = np.mgrid[0 : resized_h - window + 1 : step, 0 : resized_w - window + 1 : step]
        xs = xs_grid.ravel().astype(np.int64)
        ys = ys_grid.ravel().astype(np.int64)
        passed, cum_scores = _predict_windows(cascade, integral, integral_sq, xs, ys)

        # 分数阈值：过滤低置信度检测
        if score_threshold > 0.0:
            passed = passed & (cum_scores >= score_threshold)

        for x, y, score in zip(xs[passed], ys[passed], cum_scores[passed], strict=False):
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

    if not boxes:
        return []

    if min_neighbors > 0:
        merged = _cluster_weighted_merge(boxes, iou_merge=0.18, min_count=min_neighbors)
        return [tuple(map(int, box[:4])) for box in non_max_suppression(merged, threshold=nms_threshold)]
    else:
        return [tuple(map(int, box[:4])) for box in non_max_suppression(boxes, threshold=nms_threshold)]


def _score_weighted_merge(
    boxes: list[tuple[int, int, int, int, float]],
    iou_merge: float = 0.5,
    min_count: int = 2,
) -> list[tuple[int, int, int, int, float]]:
    """NMS + 分数加权微调：先用 NMS 选出最高分候选，再用周围重叠框微调位置。

    流程：
    1. NMS 选出每个区域的最高分框作为锚点
    2. 对每个锚点，找所有与它 IoU >= 阈值的原始框
    3. 邻居数 < min_count 的锚点丢弃（可能是孤立误检）
    4. 用分数加权平均微调锚点位置，高置信度检测贡献更大
    """
    if not boxes:
        return []

    from .nms import intersection_over_union

    # Step 1: NMS 选出候选锚点（高分优先，IoU>0.3 的重叠框被抑制）
    sorted_boxes = sorted(boxes, key=lambda b: b[4], reverse=True)
    anchors = []
    for box in sorted_boxes:
        if all(intersection_over_union(box, a) < 0.3 for a in anchors):
            anchors.append(box)

    results = []
    for anchor in anchors:
        # Step 2: 找所有与锚点重叠度 >= iou_merge 的原始框
        neighbors = [b for b in boxes if intersection_over_union(anchor, b) >= iou_merge]

        # Step 3: 邻居不够多就丢弃
        if len(neighbors) < min_count:
            continue

        # Step 4: 分数加权平均微调位置
        min_score = min(b[4] for b in neighbors)
        weights = [max(b[4] - min_score + 1e-3, 1e-3) for b in neighbors]
        total_weight = sum(weights)
        total_score = sum(b[4] for b in neighbors)
        wx = sum(b[0] * weight for b, weight in zip(neighbors, weights, strict=False)) / total_weight
        wy = sum(b[1] * weight for b, weight in zip(neighbors, weights, strict=False)) / total_weight
        ww = sum(b[2] * weight for b, weight in zip(neighbors, weights, strict=False)) / total_weight
        wh = sum(b[3] * weight for b, weight in zip(neighbors, weights, strict=False)) / total_weight

        results.append((int(round(wx)), int(round(wy)), int(round(ww)), int(round(wh)), total_score))

    return results


def _cluster_weighted_merge(
    boxes: list[tuple[int, int, int, int, float]],
    iou_merge: float = 0.18,
    min_count: int = 2,
) -> list[tuple[int, int, int, int, float]]:
    """Merge duplicate windows before NMS.

    Adjacent scales often produce several boxes for the same face. IoU alone is
    not enough when one box is slightly larger or shifted, so this also merges
    boxes that contain each other or have close centers and similar sizes.
    """
    if not boxes:
        return []

    from .nms import intersection_over_union

    def center_distance(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a[:4]
        bx, by, bw, bh = b[:4]
        acx, acy = ax + aw / 2, ay + ah / 2
        bcx, bcy = bx + bw / 2, by + bh / 2
        return float(((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5)

    def containment(a: tuple, b: tuple) -> float:
        ax, ay, aw, ah = a[:4]
        bx, by, bw, bh = b[:4]
        x1, y1 = max(ax, bx), max(ay, by)
        x2, y2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        return inter / max(1, min(aw * ah, bw * bh))

    def same_face_cluster(a: tuple, b: tuple) -> bool:
        area_a = a[2] * a[3]
        area_b = b[2] * b[3]
        ref_size = max(a[2], a[3], b[2], b[3], 1)
        size_ratio = min(area_a, area_b) / max(area_a, area_b, 1)
        return (
            intersection_over_union(a, b) >= iou_merge
            or containment(a, b) >= 0.65
            or (center_distance(a, b) <= ref_size * 0.38 and size_ratio >= 0.35)
        )

    clusters: list[list[tuple[int, int, int, int, float]]] = []
    for box in sorted(boxes, key=lambda item: item[4], reverse=True):
        for cluster in clusters:
            if any(same_face_cluster(box, member) for member in cluster):
                cluster.append(box)
                break
        else:
            clusters.append([box])

    results = []
    for cluster in clusters:
        if len(cluster) < min_count:
            continue

        min_score = min(box[4] for box in cluster)
        weights = [max(box[4] - min_score + 1e-3, 1e-3) for box in cluster]
        total_weight = sum(weights)
        total_score = sum(box[4] for box in cluster)
        cx = sum((box[0] + box[2] / 2) * weight for box, weight in zip(cluster, weights, strict=False)) / total_weight
        cy = sum((box[1] + box[3] / 2) * weight for box, weight in zip(cluster, weights, strict=False)) / total_weight

        # Do not average the size downward. High-score inner-face windows can be
        # smaller than the visual face, so keep a large representative size from
        # the cluster and let NMS remove remaining duplicates.
        sorted_widths = sorted(box[2] for box in cluster)
        sorted_heights = sorted(box[3] for box in cluster)
        size_index = max(0, int(round(0.75 * (len(cluster) - 1))))
        w = sorted_widths[size_index]
        h = sorted_heights[size_index]
        x = cx - w / 2
        y = cy - h / 2
        results.append((int(round(x)), int(round(y)), int(round(w)), int(round(h)), total_score))

    return results
