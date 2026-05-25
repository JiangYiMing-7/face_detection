"""
审计级联分类器：
1. 检查每一级的阈值 vs 实际正/负样本分数分布
2. 验证训练路径 vs 检测路径的特征值是否一致
3. 定位 FP 过高的根因
"""
import sys, json
sys.path.insert(0, ".")
import numpy as np
import cv2
from src.cascade import CascadeClassifier
from src.haar_features import generate_haar_features, compute_feature_matrix
from src.integral_image import compute_integral_image

# ===== 1. 加载模型，检查每一级的阈值和alpha范围 =====
cascade = CascadeClassifier.load("models/custom_cascade_v3_hnm.json")
print("="*60)
print("1. 模型结构分析")
print("="*60)
for i, stage in enumerate(cascade.stages):
    alphas = [w.alpha for w in stage.weak_classifiers]
    max_score = sum(alphas)  # 所有 weak 都预测 +1 时的最大分
    min_score = -sum(alphas)  # 所有 weak 都预测 -1 时的最小分
    print(f"  Stage {i+1}: {len(stage.weak_classifiers)} weaks, "
          f"threshold={stage.threshold:.4f}, "
          f"score_range=[{min_score:.2f}, {max_score:.2f}], "
          f"相对阈值位置={(stage.threshold - min_score)/(max_score - min_score)*100:.1f}%")

# ===== 2. 用一个训练正样本验证特征值一致性 =====
print("\n" + "="*60)
print("2. 训练路径 vs 检测路径 特征值一致性验证")
print("="*60)

# 造一个 24x24 的假窗口
np.random.seed(42)
window = np.random.randint(0, 256, (24, 24), dtype=np.uint8).astype(np.float32)

features = generate_haar_features(24, 8000, seed=42)

# 训练路径：compute_feature_matrix
train_feats = compute_feature_matrix(window[np.newaxis, ...], features)[0]

# 检测路径：values_at on integral image
integral = compute_integral_image(window.astype(np.uint8))
detect_feats = np.array([f.values_at(integral, np.array([0]), np.array([0]))[0] for f in features])

diff = np.abs(train_feats - detect_feats)
print(f"  最大差异: {diff.max():.6f}")
print(f"  平均差异: {diff.mean():.6f}")
print(f"  差异>0.01的特征数: {(diff > 0.01).sum()}/{len(features)}")

if diff.max() > 1.0:
    bad_idx = np.argmax(diff)
    print(f"  !! 差异最大的特征 #{bad_idx}: train={train_feats[bad_idx]:.2f} detect={detect_feats[bad_idx]:.2f}")
    print(f"     feature: {features[bad_idx]}")

# ===== 3. 在固定负样本上检验级联的 FP 率 =====
print("\n" + "="*60)
print("3. 逐级 FP 率分析（在随机负样本上）")
print("="*60)

from src.cascade import bind_cascade_feature_indices
bind_cascade_feature_indices(cascade, features)

# 生成一些随机纹理 patches 作为负样本
neg_windows = []
for _ in range(500):
    patch = np.random.randint(0, 256, (24, 24), dtype=np.uint8).astype(np.float32)
    neg_windows.append(patch)
neg_windows = np.stack(neg_windows)
neg_matrix = compute_feature_matrix(neg_windows, features)

# 逐级追踪
alive = np.ones(len(neg_matrix), dtype=bool)
for i, stage in enumerate(cascade.stages):
    alive_indices = np.flatnonzero(alive)
    if len(alive_indices) == 0:
        break
    scores = stage.decision_function(neg_matrix[alive])
    rejected = scores < stage.threshold
    alive[alive_indices[rejected]] = False
    pass_count = alive.sum()
    print(f"  Stage {i+1}: pass={pass_count}/{len(neg_matrix)} ({pass_count/len(neg_matrix)*100:.1f}%), "
          f"score: mean={scores.mean():.3f}, std={scores.std():.3f}, "
          f"min={scores.min():.3f}, threshold={stage.threshold:.3f}")

# ===== 4. 在真实图像 patch 上测试 =====
print("\n" + "="*60)
print("4. 真实图像 patch 上的逐级 FP 率")
print("="*60)

img = cv2.imread("data/test_lfw/images/Abba_Eban_0001.jpg", cv2.IMREAD_GRAYSCALE)
if img is not None:
    # 随机采样 200 个 24x24 patches
    real_patches = []
    h, w = img.shape
    for _ in range(200):
        x = np.random.randint(0, w - 24)
        y = np.random.randint(0, h - 24)
        real_patches.append(img[y:y+24, x:x+24].astype(np.float32))
    real_patches = np.stack(real_patches)
    real_matrix = compute_feature_matrix(real_patches, features)

    alive = np.ones(len(real_matrix), dtype=bool)
    for i, stage in enumerate(cascade.stages):
        alive_indices = np.flatnonzero(alive)
        if len(alive_indices) == 0:
            break
        scores = stage.decision_function(real_matrix[alive])
        rejected = scores < stage.threshold
        alive[alive_indices[rejected]] = False
        pass_count = alive.sum()
        print(f"  Stage {i+1}: pass={pass_count}/200 ({pass_count/200*100:.1f}%), "
              f"score: mean={scores.mean():.3f}, min={scores.min():.3f}, threshold={stage.threshold:.3f}")
