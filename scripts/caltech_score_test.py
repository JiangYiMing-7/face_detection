"""Sweep custom detector score thresholds on the prepared Caltech test split."""

import sys, time
sys.path.insert(0, ".")
import cv2
from pathlib import Path
from src.detectors import create_detector
from src.annotations import load_annotations
from src.metrics import match_detections, summarize_counts

det = create_detector("custom", "models/custom_cascade_v3_hnm.json")
ann = load_annotations("data/test/caltech/annotations.json")
IMAGE_DIR = Path("data/test/caltech/images")
image_paths = sorted(p for p in IMAGE_DIR.rglob("*") if p.suffix.lower() in {".jpg", ".png"})

for score_thresh, min_size in [(12, 120), (15, 120), (20, 120), (25, 120), (30, 120)]:
    options = {
        "scale_factor": 1.2,
        "min_size": min_size,
        "window_step": 4,
        "nms_threshold": 0.3,
        "variance_normalize": False,
        "min_neighbors": 0,
        "score_threshold": score_thresh,
    }
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
    print("score=%2d ms=%d  TP=%4d  FP=%5d  FN=%4d  P=%.4f  R=%.3f  F1=%.4f" % (
        score_thresh, min_size, total_tp, total_fp, total_fn,
        s['precision'], s['recall'], s['f1']))
