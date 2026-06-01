"""测试大 min_size 的效果"""
import time
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

for min_size, step in [(60, 4), (60, 8), (80, 4), (80, 8), (100, 4)]:
    options = {
        "scale_factor": 1.2,
        "min_size": min_size,
        "window_step": step,
        "nms_threshold": 0.3,
        "variance_normalize": False,
        "min_neighbors": 0,
        "score_threshold": 0.0,
    }
    total_tp = total_fp = total_fn = 0
    t0 = time.perf_counter()
    for img_path in image_paths:
        frame = cv2.imread(str(img_path))
        if frame is None: continue
        gt = ann.get(img_path.name, [])
        _, preds = det.detect(frame, options)
        r = match_detections(gt, preds, 0.5)
        total_tp += r.true_positive
        total_fp += r.false_positive
        total_fn += r.false_negative
    elapsed = time.perf_counter() - t0
    s = summarize_counts(total_tp, total_fp, total_fn)
    fps = len(image_paths) / elapsed
    print(f"ms={min_size:3d} step={step}  TP={total_tp:3d}  FP={total_fp:5d}  FN={total_fn:3d}  P={s['precision']:.4f}  R={s['recall']:.3f}  F1={s['f1']:.4f}  FPS={fps:.1f}")
