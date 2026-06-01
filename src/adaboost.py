"""自实现 Viola-Jones cascade 使用的 AdaBoost 训练基础组件。"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .haar_features import HaarFeature


@dataclass
class WeakClassifier:
    """基于单个 Haar 特征的弱分类器。

    polarity 表示不等号方向：同一个特征既可能是“小于阈值为人脸”，
    也可能是“大于阈值为人脸”，训练时会自动选择误差更小的方向。
    """

    feature: HaarFeature
    threshold: float
    polarity: int
    alpha: float = 0.0
    error: float = 0.0

    def predict_values(self, values: np.ndarray) -> np.ndarray:
        """根据一列特征值输出 +1/−1 标签。"""
        values = np.asarray(values)
        if self.polarity == 1:
            return np.where(values <= self.threshold, 1, -1)
        return np.where(values > self.threshold, 1, -1)

    def predict_integral(self, integral: np.ndarray, x: int = 0, y: int = 0) -> int:
        """在单张积分图的指定窗口位置预测 +1/-1。"""
        value = self.feature.value(integral, x, y)
        if self.polarity == 1:
            return 1 if value <= self.threshold else -1
        return 1 if value > self.threshold else -1

    def to_dict(self) -> dict:
        """序列化弱分类器，供 JSON 模型文件保存。"""
        return {
            "feature": self.feature.to_dict(),
            "threshold": self.threshold,
            "polarity": self.polarity,
            "alpha": self.alpha,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "WeakClassifier":
        """从 JSON 字典恢复弱分类器。"""
        return cls(
            feature=HaarFeature.from_dict(data["feature"]),
            threshold=float(data["threshold"]),
            polarity=int(data["polarity"]),
            alpha=float(data.get("alpha", 0.0)),
            error=float(data.get("error", 0.0)),
        )


@dataclass
class StrongClassifier:
    """AdaBoost 强分类器，由多个弱分类器加权相加得到。"""

    weak_classifiers: list[WeakClassifier] = field(default_factory=list)
    threshold: float = 0.0

    def decision_function(self, feature_matrix: np.ndarray) -> np.ndarray:
        """计算强分类器分数，分数越高越像正样本人脸。"""
        scores = np.zeros(feature_matrix.shape[0], dtype=np.float64)
        for weak in self.weak_classifiers:
            column_values = feature_matrix[:, getattr(weak, "feature_index")]
            scores += weak.alpha * weak.predict_values(column_values)
        return scores

    def predict_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """把强分类器分数转换为 +1/-1 预测标签。"""
        return np.where(self.decision_function(feature_matrix) >= self.threshold, 1, -1)

    def score_integral(self, integral: np.ndarray, x: int = 0, y: int = 0) -> float:
        """在单个窗口上累加所有弱分类器的加权得分。"""
        score = 0.0
        for weak in self.weak_classifiers:
            score += weak.alpha * weak.predict_integral(integral, x, y)
        return score

    def predict_integral(self, integral: np.ndarray, x: int = 0, y: int = 0) -> int:
        """在单个窗口上输出强分类器预测标签。"""
        return 1 if self.score_integral(integral, x, y) >= self.threshold else -1

    def to_dict(self) -> dict:
        """序列化强分类器及其包含的弱分类器。"""
        return {
            "threshold": self.threshold,
            "weak_classifiers": [weak.to_dict() for weak in self.weak_classifiers],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StrongClassifier":
        """从 JSON 字典恢复强分类器。"""
        return cls(
            weak_classifiers=[WeakClassifier.from_dict(item) for item in data["weak_classifiers"]],
            threshold=float(data.get("threshold", 0.0)),
        )


def _find_best_weak_classifier(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    features: list[HaarFeature],
    sorted_indices: np.ndarray,
    selected: np.ndarray,
    batch_size: int = 512,
) -> tuple[int, float, int, float]:
    """在当前样本权重下寻找加权错误率最小的单特征弱分类器。

    这里先按每个特征的取值排序，再用累积正负样本权重快速扫描阈值，
    避免对每个阈值都重新遍历全部样本。
    """
    best_feature = -1
    best_threshold = 0.0
    best_polarity = 1
    best_error = float("inf")
    total_positive = float(weights[labels == 1].sum())
    total_negative = float(weights[labels == -1].sum())

    n_features = feature_matrix.shape[1]
    for start in range(0, n_features, batch_size):
        end = min(start + batch_size, n_features)
        batch_columns = np.arange(start, end)
        active_mask = ~selected[batch_columns]
        if not np.any(active_mask):
            continue

        order = sorted_indices[:, start:end]
        sorted_labels = labels[order]
        sorted_weights = weights[order]
        sorted_values = np.take_along_axis(feature_matrix[:, start:end], order, axis=0)

        positive_weights = np.where(sorted_labels == 1, sorted_weights, 0.0)
        negative_weights = np.where(sorted_labels == -1, sorted_weights, 0.0)
        positive_left = np.cumsum(positive_weights, axis=0)
        negative_left = np.cumsum(negative_weights, axis=0)

        error_positive_left = negative_left + (total_positive - positive_left)
        error_positive_right = positive_left + (total_negative - negative_left)
        errors = np.minimum(error_positive_left, error_positive_right)
        errors[:, ~active_mask] = np.inf

        flat_index = int(np.argmin(errors))
        row_index, local_feature = np.unravel_index(flat_index, errors.shape)
        error = float(errors[row_index, local_feature])
        if error < best_error:
            best_error = error
            best_feature = start + int(local_feature)
            best_threshold = float(sorted_values[row_index, local_feature])
            if error_positive_left[row_index, local_feature] <= error_positive_right[row_index, local_feature]:
                best_polarity = 1
            else:
                best_polarity = -1

    if best_feature < 0:
        raise RuntimeError("No available Haar feature could be selected")
    return best_feature, best_threshold, best_polarity, best_error


def train_adaboost(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    features: list[HaarFeature],
    num_weaks: int,
    batch_size: int = 512,
) -> StrongClassifier:
    """训练一个 AdaBoost 强分类器。

    输入是已经计算好的特征矩阵，每一轮选择一个最优 Haar 弱分类器，
    然后提高被错分样本的权重，让后续弱分类器更关注难样本。
    """
    labels = np.asarray(labels, dtype=np.int8)
    if set(np.unique(labels).tolist()) != {-1, 1}:
        raise ValueError("AdaBoost labels must contain both -1 and 1")

    feature_matrix = np.asarray(feature_matrix, dtype=np.float32)
    _, n_features = feature_matrix.shape
    sorted_indices = np.argsort(feature_matrix, axis=0)
    selected = np.zeros(n_features, dtype=bool)

    positive_count = max(1, int(np.sum(labels == 1)))
    negative_count = max(1, int(np.sum(labels == -1)))
    weights = np.where(labels == 1, 1.0 / (2 * positive_count), 1.0 / (2 * negative_count))
    weights = weights.astype(np.float64)
    strong = StrongClassifier()

    for _ in range(int(num_weaks)):
        weights /= weights.sum()
        feature_index, threshold, polarity, error = _find_best_weak_classifier(
            feature_matrix,
            labels,
            weights,
            features,
            sorted_indices,
            selected,
            batch_size=batch_size,
        )
        if error >= 0.5:
            break

        clipped_error = min(max(error, 1e-12), 1.0 - 1e-12)
        alpha = 0.5 * np.log((1.0 - clipped_error) / clipped_error)
        weak = WeakClassifier(
            feature=features[feature_index],
            threshold=threshold,
            polarity=polarity,
            alpha=float(alpha),
            error=float(error),
        )
        setattr(weak, "feature_index", feature_index)
        predictions = weak.predict_values(feature_matrix[:, feature_index])
        weights *= np.exp(-alpha * labels * predictions)
        selected[feature_index] = True
        strong.weak_classifiers.append(weak)

    return strong


def bind_feature_indices(strong: StrongClassifier, features: list[HaarFeature]) -> None:
    """加载模型后，为每个弱分类器找回它在特征矩阵中的列号。"""
    lookup = {feature: index for index, feature in enumerate(features)}
    for weak in strong.weak_classifiers:
        if weak.feature not in lookup:
            raise ValueError("A weak classifier references a feature that is not in the feature list")
        setattr(weak, "feature_index", lookup[weak.feature])
