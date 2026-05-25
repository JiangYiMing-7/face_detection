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
    # 去掉被大框包含的小框（眼睛/鼻子等子区域误检）
    return remove_contained_boxes(selected)


def remove_contained_boxes(boxes: list[tuple], coverage: float = 0.5) -> list[tuple]:
    """如果小框的大部分面积被某个大框覆盖，删掉小框。"""
    if len(boxes) <= 1:
        return boxes
    keep = []
    for i, a in enumerate(boxes):
        ax, ay, aw, ah = a[:4]
        a_area = aw * ah
        contained = False
        for j, b in enumerate(boxes):
            if i == j:
                continue
            bx, by, bw, bh = b[:4]
            b_area = bw * bh
            if b_area <= a_area:
                continue  # 只看比自己大的框
            # 计算交集
            ix1, iy1 = max(ax, bx), max(ay, by)
            ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if a_area > 0 and inter / a_area >= coverage:
                contained = True
                break
        if not contained:
            keep.append(a)
    return keep
