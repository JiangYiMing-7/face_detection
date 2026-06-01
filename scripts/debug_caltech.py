"""快速诊断：在 Caltech faces_0001.jpg 上测试检测效果"""
import cv2
import json
from project_paths import project_path
from src.detectors import create_detector

img_path = project_path("data/test/caltech/images/faces_0001.jpg")
img = cv2.imread(str(img_path))
print("Image shape:", img.shape)

with project_path("data/test/caltech/annotations.json").open(encoding="utf-8") as handle:
    ann = json.load(handle)
print("GT box:", ann.get("faces_0001.jpg"))

det = create_detector("custom", project_path("models/custom_cascade_v1_no_hnm.json"))

# 用宽松参数
gray, boxes = det.detect(img, {
    "scale_factor": 1.2,
    "min_size": 20,
    "window_step": 4,
    "nms_threshold": 0.3,
    "variance_normalize": True,
})
print("Detected boxes:", boxes)

# 再试不归一化
gray2, boxes2 = det.detect(img, {
    "scale_factor": 1.2,
    "min_size": 20,
    "window_step": 4,
    "nms_threshold": 0.3,
    "variance_normalize": False,
})
print("Detected boxes (no VN):", boxes2)
