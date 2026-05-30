"""Smoke tests for the core face-detection math utilities."""

import unittest

import numpy as np

from src.annotations import load_annotations
from src.haar_features import HaarFeature
from src.integral_image import compute_integral_image, rect_sum
from src.metrics import match_detections, summarize_counts
from src.nms import intersection_over_union, non_max_suppression


class CoreAlgorithmTests(unittest.TestCase):
    """Regression tests for integral images, Haar features, NMS, and metrics."""

    def test_integral_image_rect_sum(self):
        """Integral image rectangle sums should match direct NumPy sums."""
        image = np.arange(1, 17, dtype=np.float64).reshape(4, 4)
        integral = compute_integral_image(image)
        self.assertEqual(rect_sum(integral, 1, 1, 2, 2), float(image[1:3, 1:3].sum()))

    def test_haar_feature_value(self):
        """A two-rectangle Haar feature should produce the expected contrast."""
        image = np.array([[1, 1, 5, 5], [1, 1, 5, 5]], dtype=np.float64)
        integral = compute_integral_image(image)
        feature = HaarFeature("two_horizontal", 0, 0, 4, 2)
        self.assertEqual(feature.value(integral), -16.0)

    def test_iou_nms_and_metrics(self):
        """Bounding-box overlap, NMS, matching, and summary metrics should agree."""
        self.assertAlmostEqual(intersection_over_union((0, 0, 10, 10), (5, 5, 10, 10)), 25 / 175)
        kept = non_max_suppression([(0, 0, 10, 10, 0.9), (1, 1, 10, 10, 0.8), (30, 30, 5, 5, 0.7)])
        self.assertEqual(len(kept), 2)
        result = match_detections([(0, 0, 10, 10)], [(1, 1, 10, 10), (30, 30, 5, 5)], 0.5)
        self.assertEqual((result.true_positive, result.false_positive, result.false_negative), (1, 1, 0))
        summary = summarize_counts(1, 1, 0)
        self.assertAlmostEqual(summary["precision"], 0.5)


if __name__ == "__main__":
    unittest.main()
