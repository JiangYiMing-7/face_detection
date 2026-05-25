from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import time

import cv2

from src.annotations import load_annotations
from src.detectors import create_detector
from src.metrics import match_detections, summarize_counts


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate OpenCV or custom face detector on annotated images.")
    parser.add_argument("--detector", choices=["opencv", "custom"], default="opencv")
    parser.add_argument("--model", default="models/custom_cascade_v3_hnm.json", help="Custom cascade model path.")
    parser.add_argument("--image-dir", default="data/test/images")
    parser.add_argument("--annotations", default="data/test/annotations.json")
    parser.add_argument("--output-dir", default="results/eval")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--scale-factor", type=float, default=1.2)
    parser.add_argument("--min-neighbors", type=int, default=5,
                        help="groupRectangles 最小邻居数（0=只用NMS，>0更严格）")
    parser.add_argument("--min-size", type=int, default=100)
    parser.add_argument("--window-step", type=int, default=4)
    parser.add_argument("--nms-threshold", type=float, default=0.3)
    parser.add_argument("--score-threshold", type=float, default=0.0, help="累积分数阈值，过滤低置信度检测（0=不过滤）")
    parser.add_argument("--no-variance-normalize", action="store_true", default=True,
                        help="关闭方差归一化（默认开启；模型在原始像素值上训练）")
    parser.add_argument("--equalize", action="store_true")
    parser.add_argument("--clahe", action="store_true", help="使用 CLAHE 自适应直方图均衡化预处理")
    return parser.parse_args()


def image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def draw_boxes(image, boxes, color, label: str) -> None:
    """在测评可视化图上绘制真实框或预测框。"""
    for x, y, w, h in boxes:
        cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
        cv2.putText(image, label, (x, max(18, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA)


def main() -> None:
    """批量评估检测器，并输出逐图指标、汇总指标和可视化图片。"""
    args = parse_args()
    image_dir = Path(args.image_dir)
    annotations = load_annotations(args.annotations)
    detector = create_detector(args.detector, args.model)
    output_dir = Path(args.output_dir)
    visualization_dir = output_dir / "visualizations"
    visualization_dir.mkdir(parents=True, exist_ok=True)

    options = {
        "scale_factor": args.scale_factor,
        "min_neighbors": args.min_neighbors,
        "min_size": args.min_size,
        "window_step": args.window_step,
        "nms_threshold": args.nms_threshold,
        "score_threshold": args.score_threshold,
        "variance_normalize": not args.no_variance_normalize,
        "equalize": args.equalize,
        "clahe": args.clahe,
    }

    rows = []
    total_tp = 0
    total_fp = 0
    total_fn = 0
    total_time = 0.0
    evaluated = 0

    all_paths = image_paths(image_dir)
    total_imgs = len(all_paths)
    for image_path in all_paths:
        frame = cv2.imread(str(image_path))
        if frame is None:
            continue
        gt_boxes = annotations.get(image_path.name, annotations.get(str(image_path.relative_to(image_dir)), []))
        print(f"\r  [{evaluated+1}/{total_imgs}] {image_path.name[:40]:<40}", end="", flush=True)

        start = time.perf_counter()
        _, predictions = detector.detect(frame, options)
        elapsed = time.perf_counter() - start
        total_time += elapsed
        evaluated += 1

        result = match_detections(gt_boxes, predictions, args.iou_threshold)
        total_tp += result.true_positive
        total_fp += result.false_positive
        total_fn += result.false_negative
        per_image = summarize_counts(result.true_positive, result.false_positive, result.false_negative)
        rows.append(
            {
                "image": image_path.name,
                "ground_truth": len(gt_boxes),
                "predictions": len(predictions),
                "true_positive": result.true_positive,
                "false_positive": result.false_positive,
                "false_negative": result.false_negative,
                "precision": f"{per_image['precision']:.4f}",
                "recall": f"{per_image['recall']:.4f}",
                "f1": f"{per_image['f1']:.4f}",
                "time_ms": f"{elapsed * 1000:.2f}",
            }
        )

        visual = frame.copy()
        draw_boxes(visual, gt_boxes, (0, 255, 0), "GT")
        draw_boxes(visual, predictions, (0, 255, 255), "Pred")
        cv2.imwrite(str(visualization_dir / image_path.name), visual)

    summary = summarize_counts(total_tp, total_fp, total_fn)
    summary.update(
        {
            "detector": args.detector,
            "images": evaluated,
            "average_time_ms": (total_time / evaluated * 1000) if evaluated else 0.0,
            "average_fps": (evaluated / total_time) if total_time > 0 else 0.0,
            "iou_threshold": args.iou_threshold,
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "image",
            "ground_truth",
            "predictions",
            "true_positive",
            "false_positive",
            "false_negative",
            "precision",
            "recall",
            "f1",
            "time_ms",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print()  # 换行，结束进度条
    print(f"[INFO] Evaluated {evaluated} image(s) with {args.detector} detector")
    print(f"[INFO] Precision={summary['precision']:.4f} Recall={summary['recall']:.4f} F1={summary['f1']:.4f}")
    print(f"[INFO] Average FPS={summary['average_fps']:.2f}")
    print(f"[INFO] Saved metrics to: {output_dir}")


if __name__ == "__main__":
    main()
