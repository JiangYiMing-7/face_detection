"""调试LFW评估：看看检测到的框和GT框的IoU"""
import cv2
import json
from project_paths import project_path
from src.detectors import create_detector
from src.nms import intersection_over_union

img_path = project_path("data/test_lfw/images/Abba_Eban_0001.jpg")
img = cv2.imread(str(img_path))
print("Image shape:", img.shape)

with project_path("data/test_lfw/annotations.json").open(encoding="utf-8") as handle:
    ann = json.load(handle)
gt = ann.get("Abba_Eban_0001.jpg")
print("GT box:", gt)

det = create_detector("custom", project_path("models/custom_cascade_v3_hnm.json"))

gray, boxes = det.detect(img, {
    "scale_factor": 1.2,
    "min_size": 30,
    "window_step": 4,
    "nms_threshold": 0.3,
    "variance_normalize": False,
})
print(f"Detected {len(boxes)} boxes:")
for box in boxes[:10]:
    iou = intersection_over_union(box, gt[0])
    print(f"  {box} - IoU with GT: {iou:.3f}")
