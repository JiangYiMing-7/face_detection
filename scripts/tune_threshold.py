"""快速扫描 score_threshold，找到 Precision-Recall 最优点"""
import cv2
from pathlib import Path
from project_paths import project_path
from src.detectors import create_detector
from src.annotations import load_annotations
from src.metrics import match_detections, summarize_counts

det = create_detector("custom", project_path("models/custom_cascade_v3_hnm.json"))
ann = load_annotations(project_path("data/test_lfw/annotations.json"))

IMAGE_DIR = project_path("data/test_lfw/images")
image_paths = sorted(p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".png"})[:200]

base_options = {
    "scale_factor": 1.2,
    "min_size": 30,
    "window_step": 4,
    "nms_threshold": 0.3,
    "variance_normalize": False,
    "min_neighbors": 0,
}

for thresh in [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0]:
    options = {**base_options, "score_threshold": thresh}
    total_tp = total_fp = total_fn = 0
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None:
            continue
        gt = ann.get(img_path.name, [])
        _, preds = det.detect(frame, options)
        r = match_detections(gt, preds, 0.5)
        total_tp += r.true_positive
        total_fp += r.false_positive
        total_fn += r.false_negative
    s = summarize_counts(total_tp, total_fp, total_fn)
    print(f"score_thresh={thresh:5.1f}  TP={total_tp:3d}  FP={total_fp:5d}  FN={total_fn:3d}  P={s['precision']:.3f}  R={s['recall']:.3f}  F1={s['f1']:.4f}")
