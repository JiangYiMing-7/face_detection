"""调试LFW评估：看看检测到的框和GT框的IoU"""
import sys
sys.path.insert(0, ".")
import cv2
import json
from src.detectors import create_detector
from src.nms import intersection_over_union

img_path = "data/test_lfw/images/Abba_Eban_0001.jpg"
img = cv2.imread(img_path)
print("Image shape:", img.shape)

ann = json.load(open("data/test_lfw/annotations.json"))
gt = ann.get("Abba_Eban_0001.jpg")
print("GT box:", gt)

det = create_detector("custom", "models/custom_cascade_v3_hnm.json")

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
