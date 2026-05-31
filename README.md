# Viola-Jones 实时人脸检测系统

计算机视觉课程大作业：从零实现 Viola-Jones 人脸检测算法，并与 OpenCV 预训练 Haar Cascade 进行对比。

## 核心成果

| 检测器 | LFW F1 | 说明 |
|--------|--------|------|
| OpenCV Haar Cascade | 0.980 | 预训练基线，Recall=0.960 |
| **自实现 v6.1** | **1.000** | LFW 常规操作点零漏检、零误报 |

## 项目结构

```
custom/
├── train.py                  # 训练脚本（v3 及以前，1000正样本）
├── train_v6.py               # v6/v6.1 训练脚本（3000正样本、x7增强）
├── evaluate.py               # 定量评估（支持 --clahe）
├── demo.py                   # 实时摄像头演示（双屏对比）
├── prepare_data.py           # 从 WIDER FACE 裁切训练正样本
├── audit_cascade.py          # 级联模型审计（分析每级 FPR/TPR）
├── main.py                   # 早期演示入口（单检测器）
├── src/
│   ├── integral_image.py     # 积分图（含平方积分图）
│   ├── haar_features.py      # Haar-like 特征定义与枚举
│   ├── adaboost.py           # AdaBoost 训练与弱/强分类器
│   ├── cascade.py            # 级联分类器（JSON 序列化）
│   ├── sliding_window.py     # 多尺度滑窗 + _cluster_weighted_merge
│   ├── nms.py                # NMS + 包含框过滤
│   ├── detectors.py          # 统一接口（OpenCV / Custom + CLAHE）
│   ├── metrics.py            # P/R/F1 评估指标
│   └── annotations.py        # 标注文件解析
├── models/                   # 训练好的级联模型（JSON）
│   ├── custom_cascade_v3_hnm.json      # v3: 8级/230弱分类器
│   ├── custom_cascade_v6_full.json     # v6: 10级/525弱 (x4增强)
│   └── custom_cascade_v6_full_1.json   # v6.1: 10级/525弱 (x7增强) ★
├── data/
│   ├── train/negatives/      # 负样本图片（300张）
│   └── test_lfw/             # LFW 测试集（200张 + annotations）
└── results/                  # 评估结果、可视化
```

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：Python 3.10+、numpy、opencv-python（不使用任何深度学习框架）。

## 快速开始

### 1. 定量评估

```bash
# 在 LFW 上评估 v6.1 模型（F1=1.000）
python evaluate.py --detector custom \
  --model models/custom_cascade_v6_full_1.json \
  --image-dir data/test_lfw/images \
  --annotations data/test_lfw/annotations.json \
  --min-neighbors 5 \
  --min-size 100 \
  --output-dir results/eval_lfw_v61

# 评估 v3 模型
python evaluate.py --detector custom \
  --model models/custom_cascade_v3_hnm.json \
  --image-dir data/test_lfw/images \
  --annotations data/test_lfw/annotations.json \
  --min-neighbors 5 \
  --min-size 100 \
  --output-dir results/eval_lfw_v3

# OpenCV 基线
python evaluate.py --detector opencv \
  --image-dir data/test_lfw/images \
  --annotations data/test_lfw/annotations.json \
  --output-dir results/eval_lfw_opencv

# 更严格的 LFW 操作点（更多小窗口，更考验误检抑制）
python evaluate.py --detector custom \
  --model models/custom_cascade_v6_full_1.json \
  --image-dir data/test_lfw/images \
  --annotations data/test_lfw/annotations.json \
  --min-neighbors 5 \
  --min-size 60 \
  --output-dir results/eval_lfw_strict_v61
```

### 2. 实时摄像头演示

```bash
# 双屏对比（左: OpenCV, 右: 自实现）
python demo.py --model models/custom_cascade_v6_full_1.json

# 常用参数
python demo.py \
  --model models/custom_cascade_v3_hnm.json \
  --scale-factor 1.2 \
  --min-size 40 \
  --window-step 6 \
  --min-neighbors-custom 3 \
  --score-threshold 5.0
```

**快捷键**：`q` 退出 | `r` 录制 | 空格 暂停

### 3. 训练模型

训练数据需要自行准备（正样本从 WIDER FACE 裁切，负样本为非人脸图片）：

```bash
# 准备正样本
python prepare_data.py

# 训练 v3 级别模型
python train.py \
  --positive-dir data/train/positives \
  --negative-dir data/train/negatives \
  --output models/my_cascade.json \
  --max-features 20000 \
  --max-positives 1000 \
  --stage-sizes 5,10,15,20,30,40,50,60 \
  --augment

# 训练 v6.1 级别模型（需要 3000+ 正样本）
python train_v6.py \
  --positive-dir data/train/positives \
  --negative-dir data/train/negatives \
  --output models/my_cascade_v61.json \
  --max-features 20000 \
  --max-positives 3000 \
  --stage-sizes 10,15,20,30,40,50,60,80,100,120 \
  --augment
```

## 算法亮点

### 自定义后处理：_cluster_weighted_merge

替代 `cv2.groupRectangles`，使用三条件聚类解决跨尺度检测框合并：
- IoU >= 0.18（传统重叠条件）
- 包含度 >= 0.65（小框被大框覆盖）
- 中心距离近 + 尺寸相近（放宽几何约束）

### 多目标跟踪：TrackingBoxSmoother

实时演示中的多人脸独立跟踪，支持：
- IoU + 距离组合匹配评分
- 自适应平滑系数（静止 alpha=0.65，移动 alpha=0.9）
- 最多同时跟踪 8 张脸

### CLAHE 预处理

可选的自适应直方图均衡化。实验中它对较弱的 v3 模型有帮助，但对 v6/v6.1 会放大背景纹理并增加误报；最终 v6.1 默认关闭 CLAHE。

### x7 数据增强（v6.1）

翻转 + 亮度扰动 + gamma 校正 + 高斯噪声，3000 正样本扩充到 21000。

## 模型对比

| 模型 | 正样本 | 增强 | Haar 特征候选 | 级联 | LFW 常规 F1 | LFW 严格 F1 |
|------|--------|------|----------------|------|-------------|-------------|
| v3 | 1000 | x4 | 20000 | 8级/230弱 | 1.000 | 0.919 |
| v6 | 3000 | x4 | 20000 | 10级/525弱 | 1.000 | 0.964 |
| **v6.1** | **3000** | **x7** | **20000** | **10级/525弱** | **1.000** | **0.959** |

说明：LFW 常规操作点使用 `min_size=100, min_neighbors=5`；严格操作点使用 `min_size=60, min_neighbors=5`，候选窗口更多，更能暴露误检抑制能力。

## 参考文献

Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR.
