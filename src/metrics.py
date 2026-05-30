"""Evaluation metrics for face-detection bounding boxes."""

from __future__ import annotations

from dataclasses import dataclass

from .nms import intersection_over_union


@dataclass
class MatchResult:
    """一张图上的预测框与真实框匹配统计。"""

    true_positive: int
    false_positive: int
    false_negative: int
    matches: list[tuple[int, int, float]]


def match_detections(
    ground_truth: list[tuple[int, int, int, int]],
    predictions: list[tuple[int, int, int, int]],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """用 IoU 阈值把预测框和真实框做一对一匹配。"""
    used_gt: set[int] = set()
    matches: list[tuple[int, int, float]] = []

    for pred_index, prediction in enumerate(predictions):
        best_gt = -1
        best_iou = 0.0
        for gt_index, gt_box in enumerate(ground_truth):
            if gt_index in used_gt:
                continue
            iou = intersection_over_union(prediction, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_gt = gt_index
        if best_gt >= 0 and best_iou >= iou_threshold:
            used_gt.add(best_gt)
            matches.append((pred_index, best_gt, best_iou))

    tp = len(matches)
    fp = max(0, len(predictions) - tp)
    fn = max(0, len(ground_truth) - tp)
    return MatchResult(tp, fp, fn, matches)


def summarize_counts(true_positive: int, false_positive: int, false_negative: int) -> dict:
    """由 TP/FP/FN 汇总 Precision、Recall 和 F1。"""
    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }
