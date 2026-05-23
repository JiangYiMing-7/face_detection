from __future__ import annotations


def intersection_over_union(a: tuple, b: tuple) -> float:
    """计算两个框的 IoU，用于检测结果匹配和 NMS。"""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if intersection <= 0:
        return 0.0
    union = aw * ah + bw * bh - intersection
    return intersection / union if union else 0.0


def non_max_suppression(boxes: list[tuple], threshold: float = 0.3) -> list[tuple]:
    """非极大值抑制：保留高分框，去掉与其高度重叠的重复框。"""
    if not boxes:
        return []

    def score(box: tuple) -> float:
        if len(box) >= 5:
            return float(box[4])
        return float(box[2] * box[3])

    candidates = sorted(boxes, key=score, reverse=True)
    selected: list[tuple] = []
    for box in candidates:
        if all(intersection_over_union(box, kept) < threshold for kept in selected):
            selected.append(box)
    return selected
