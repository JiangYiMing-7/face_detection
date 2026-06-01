"""Cascade 分类器容器及 JSON 保存/加载工具。"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np

from .adaboost import StrongClassifier, bind_feature_indices
from .haar_features import HaarFeature


@dataclass
class CascadeClassifier:
    """简化版 Viola-Jones 级联分类器。

    每个 stage 都是一个 AdaBoost 训练出的强分类器。检测时采用串联逻辑：
    只要某一级判定窗口为负样本，就立即拒绝该窗口；只有通过所有 stage
    的窗口才会被当作人脸候选框。
    """

    stages: list[StrongClassifier] = field(default_factory=list)
    window_size: int = 24
    feature_count: int = 0

    def predict_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """对一批已经计算好的特征矩阵做级联预测。"""
        alive = np.ones(feature_matrix.shape[0], dtype=bool)
        for stage in self.stages:
            if not np.any(alive):
                break
            # 只让还没有被前面 stage 拒绝的样本继续进入下一关。
            stage_predictions = stage.predict_matrix(feature_matrix[alive])
            alive_indices = np.flatnonzero(alive)
            alive[alive_indices[stage_predictions == -1]] = False
        return np.where(alive, 1, -1)

    def score_matrix(self, feature_matrix: np.ndarray) -> np.ndarray:
        """返回样本在最后通过或被拒绝的 stage 上的分数。"""
        scores = np.zeros(feature_matrix.shape[0], dtype=np.float64)
        alive = np.ones(feature_matrix.shape[0], dtype=bool)
        for stage in self.stages:
            if not np.any(alive):
                break
            stage_scores = stage.decision_function(feature_matrix[alive])
            alive_indices = np.flatnonzero(alive)
            scores[alive_indices] = stage_scores
            alive[alive_indices[stage_scores < stage.threshold]] = False
        scores[~alive] -= 1e6
        return scores

    def predict_integral(self, integral: np.ndarray, x: int = 0, y: int = 0) -> tuple[bool, float]:
        """在一张积分图的指定窗口位置执行级联检测。"""
        score = 0.0
        for stage in self.stages:
            score = stage.score_integral(integral, x, y)
            if score < stage.threshold:
                return False, score
        return True, score

    def to_dict(self) -> dict:
        """把级联模型转换成可写入 JSON 的纯 Python 字典。"""
        return {
            "type": "teaching_viola_jones_cascade",
            "window_size": self.window_size,
            "feature_count": self.feature_count,
            "stages": [stage.to_dict() for stage in self.stages],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CascadeClassifier":
        """从 JSON 字典恢复级联分类器对象。"""
        return cls(
            stages=[StrongClassifier.from_dict(item) for item in data.get("stages", [])],
            window_size=int(data.get("window_size", 24)),
            feature_count=int(data.get("feature_count", 0)),
        )

    def save(self, path: str | Path) -> None:
        """把训练好的 cascade 保存为 JSON，方便演示和测评脚本加载。"""
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CascadeClassifier":
        """从 JSON 文件加载训练好的 cascade 模型。"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls.from_dict(data)


def bind_cascade_feature_indices(cascade: CascadeClassifier, features: list[HaarFeature]) -> None:
    """为从 JSON 反序列化的弱分类器重新绑定特征下标。"""
    for stage in cascade.stages:
        bind_feature_indices(stage, features)
