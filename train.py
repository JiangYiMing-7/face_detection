"""Train the custom teaching-oriented Viola-Jones cascade model."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import time

import cv2
import numpy as np

from src.adaboost import train_adaboost
from src.cascade import CascadeClassifier
from src.haar_features import compute_feature_matrix, generate_haar_features, render_feature


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_stage_sizes(raw: str) -> list[int]:
    """把形如 '10,20,40' 的参数解析为每级弱分类器数量列表。"""
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("stage sizes must be positive integers such as 10,20,40")
    return values


def parse_args() -> argparse.Namespace:
    """解析训练脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="Train a teaching-oriented Viola-Jones cascade.")
    parser.add_argument("--positive-dir", default="data/train/positives", help="Directory of face crop images.")
    parser.add_argument("--negative-dir", default="data/train/negatives", help="Directory of non-face images.")
    parser.add_argument("--output", default="models/custom_cascade.json", help="Path to save the trained model.")
    parser.add_argument("--window-size", type=int, default=24, help="Training window size. Default: 24")
    parser.add_argument("--max-features", type=int, default=8000, help="Number of sampled Haar features. Default: 8000")
    parser.add_argument("--stage-sizes", type=parse_stage_sizes, default=parse_stage_sizes("10,20,40"))
    parser.add_argument("--max-positives", type=int, default=1000, help="Maximum positive crops to load.")
    parser.add_argument("--negative-samples", type=int, default=2500, help="Total negative patches sampled.")
    parser.add_argument("--negatives-per-image", type=int, default=20, help="Random negative patches per source image.")
    parser.add_argument("--active-negatives", type=int, default=1200, help="Negative patches used in each stage.")
    parser.add_argument("--stage-detection-rate", type=float, default=0.995, help="Training positive pass-rate target per stage.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--feature-batch-size", type=int, default=512, help="Feature training batch size.")
    parser.add_argument("--augment", action="store_true", help="对正样本做翻转+亮度增强（×3 数量）。")
    parser.add_argument("--no-hnm", action="store_true", help="禁用 Hard Negative Mining，只用固定负样本池。")
    parser.add_argument("--fp-constraint", action="store_true",
                        help="启用 FP 率约束（每级阈值同时约束 ≤50%% FP 率）。不启用时只约束检测率，适合配合 min_neighbors 后处理。")
    parser.add_argument("--adaptive-stages", action="store_true",
                        help="原版 V-J 动态级联：每级动态增加弱分类器直到 FP 率 ≤ target_fp_rate，"
                             "阈值只按检测率设定。--stage-sizes 变为每级的最大弱分类器上限。")
    parser.add_argument("--target-fp-rate", type=float, default=0.5,
                        help="adaptive-stages 模式下每级的目标 FP 率（默认 0.5）")
    parser.add_argument("--num-stages", type=int, default=15,
                        help="adaptive-stages 模式下的级联总级数（默认 15）")
    parser.add_argument("--max-weaks-per-stage", type=int, default=200,
                        help="adaptive-stages 模式下每级最多弱分类器数（默认 200）")
    return parser.parse_args()


def image_paths(directory: str | Path) -> list[Path]:
    """递归列出目录下支持格式的图片文件。"""
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def read_gray(path: Path) -> np.ndarray | None:
    """以灰度模式读取图片，读取失败时返回 None。"""
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    return image


def normalize_window(gray: np.ndarray, window_size: int) -> np.ndarray:
    """把任意大小的训练 patch 统一缩放为固定窗口大小。"""
    resized = cv2.resize(gray, (window_size, window_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


def augment_windows(windows: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """数据增强：水平翻转 + 亮度扰动，正样本量 ×3。

    - 水平翻转：模拟不同朝向的正脸（左右对称）
    - 亮度扰动：×0.82 和 ×1.18，模拟不同光照条件
    所有增强版本 clip 到 [0, 255]，保持与原始像素相同量级。
    """
    flipped   = windows[:, :, ::-1].copy()
    darker    = np.clip(windows * 0.82, 0.0, 255.0).astype(np.float32)
    brighter  = np.clip(windows * 1.18, 0.0, 255.0).astype(np.float32)
    return np.concatenate([windows, flipped, darker, brighter], axis=0)


def load_positive_windows(directory: str | Path, window_size: int, max_count: int) -> np.ndarray:
    """读取正样本：目录中的每张图都被视作一张人脸 crop。"""
    windows = []
    for path in image_paths(directory):
        gray = read_gray(path)
        if gray is None:
            continue
        windows.append(normalize_window(gray, window_size))
        if max_count > 0 and len(windows) >= max_count:
            break
    if not windows:
        raise RuntimeError(f"No positive images found in {directory}")
    return np.stack(windows, axis=0)


def sample_negative_windows(
    directory: str | Path,
    window_size: int,
    total_samples: int,
    per_image: int,
    seed: int,
) -> np.ndarray:
    """从非人脸图片中随机裁剪负样本 patch。"""
    rng = np.random.default_rng(seed)
    windows = []
    paths = image_paths(directory)
    if not paths:
        raise RuntimeError(f"No negative images found in {directory}")

    shuffled = paths.copy()
    rng.shuffle(shuffled)
    while len(windows) < total_samples:
        made_progress = False
        for path in shuffled:
            gray = read_gray(path)
            if gray is None:
                continue
            h, w = gray.shape[:2]
            if min(h, w) < 8:
                continue
            for _ in range(max(1, per_image)):
                side = int(rng.integers(max(8, min(h, w) // 5), min(h, w) + 1))
                x = int(rng.integers(0, w - side + 1))
                y = int(rng.integers(0, h - side + 1))
                patch = gray[y : y + side, x : x + side]
                windows.append(normalize_window(patch, window_size))
                made_progress = True
                if len(windows) >= total_samples:
                    break
            if len(windows) >= total_samples:
                break
        if not made_progress:
            break

    if not windows:
        raise RuntimeError(f"Could not sample negative patches from {directory}")
    return np.stack(windows[:total_samples], axis=0)


def mine_hard_negatives(
    directory: str | Path,
    cascade: CascadeClassifier,
    features: list,
    window_size: int,
    target_count: int,
    per_image: int,
    rng: np.random.Generator,
) -> np.ndarray | None:
    """Hard Negative Mining（自举负样本挖掘）——论文核心流程之一。

    从负样本图片中大量随机采样 patch，通过当前已训练的级联的 patch
    才是"难例"（hard negative）：模型已经不容易拒绝它们。
    把这些难例加入下一 stage 的训练，迫使后续 stage 学习更细致的判别边界。

    返回难例的特征矩阵（行 = 样本，列 = 特征），若一个也没找到则返回 None。
    """
    paths = image_paths(directory)
    if not paths:
        return None

    shuffled = list(paths)
    rng.shuffle(shuffled)
    candidate_windows: list[np.ndarray] = []

    # 过采样：多采一些候选，再用级联过滤
    oversample = max(target_count * 5, 1000)
    for path in shuffled:
        if len(candidate_windows) >= oversample:
            break
        gray = read_gray(path)
        if gray is None:
            continue
        h, w = gray.shape
        if min(h, w) < window_size:
            continue
        for _ in range(per_image):
            side = int(rng.integers(window_size, min(h, w) + 1))
            x = int(rng.integers(0, max(1, w - side + 1)))
            y = int(rng.integers(0, max(1, h - side + 1)))
            patch = gray[y : y + side, x : x + side]
            candidate_windows.append(normalize_window(patch, window_size))
            if len(candidate_windows) >= oversample:
                break

    if not candidate_windows:
        return None

    candidates_arr = np.stack(candidate_windows)
    feat_mat = compute_feature_matrix(candidates_arr, features)
    predictions = cascade.predict_matrix(feat_mat)
    hard_indices = np.flatnonzero(predictions == 1)

    if len(hard_indices) == 0:
        return None

    chosen = rng.choice(
        hard_indices,
        size=min(target_count, len(hard_indices)),
        replace=False,
    )
    return feat_mat[chosen]


def choose_stage_threshold(
    scores: np.ndarray,
    target_detection_rate: float,
    negative_scores: np.ndarray | None = None,
    max_fp_rate: float = 0.5,
) -> float:
    """按目标通过率选择 stage 阈值，可选地约束 FP 率。

    原版 Viola-Jones 论文要求每级同时满足:
      1. 正样本检测率 >= target_detection_rate (如 99.5%)
      2. 负样本误报率 <= max_fp_rate (如 50%)
    当 negative_scores 为 None 时只约束检测率（更宽松，适合配合 min_neighbors 后处理）。
    """
    if scores.size == 0:
        return 0.0
    keep_rate = min(max(float(target_detection_rate), 0.5), 1.0)
    quantile = max(0.0, 1.0 - keep_rate)
    threshold_dr = float(np.quantile(scores, quantile))

    if negative_scores is not None and negative_scores.size > 0:
        threshold_fp = float(np.quantile(negative_scores, max_fp_rate))
        # 不能超过正样本 5% 分位数，保证检测率不低于 95%
        safe_upper = float(np.quantile(scores, 0.05))
        threshold = max(threshold_dr, min(threshold_fp, safe_upper))
        return threshold
    return threshold_dr


def save_training_log(rows: list[dict], output_model: Path) -> None:
    """保存每级训练指标，供后续报告画图和误差分析使用。"""
    log_dir = Path("results/logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "custom_training_log.csv"
    with log_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Saved training log: {log_path}")


def save_top_features(cascade: CascadeClassifier, output_dir: str | Path = "results/features") -> None:
    """保存每一级权重较大的 Haar 特征可视化图，方便报告展示。"""
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    rank = 1
    for stage_index, stage in enumerate(cascade.stages, start=1):
        sorted_weaks = sorted(stage.weak_classifiers, key=lambda weak: abs(weak.alpha), reverse=True)
        for weak in sorted_weaks[:3]:
            image = render_feature(weak.feature)
            path = directory / f"stage{stage_index}_feature{rank}.png"
            cv2.imwrite(str(path), image)
            rank += 1
    if rank > 1:
        print(f"[INFO] Saved feature visualizations to: {directory}")


def main() -> None:
    """训练简化 Cascade：每一级训练后收集 hard negatives 供下一级使用。"""
    args = parse_args()
    start_time = time.perf_counter()
    rng = np.random.default_rng(args.seed)

    print("[INFO] Loading training windows...")
    positives = load_positive_windows(args.positive_dir, args.window_size, args.max_positives)
    if args.augment:
        print(f"[INFO] Augmenting positives: {len(positives)} → ", end="", flush=True)
        positives = augment_windows(positives, rng)
        print(f"{len(positives)} (flip + brightness ×3)")
    negatives = sample_negative_windows(
        args.negative_dir,
        args.window_size,
        args.negative_samples,
        args.negatives_per_image,
        args.seed,
    )
    print(f"[INFO] Positives: {len(positives)} | Negative pool: {len(negatives)}")

    print("[INFO] Generating Haar features...")
    features = generate_haar_features(args.window_size, args.max_features, args.seed)
    print(f"[INFO] Haar features: {len(features)}")

    print("[INFO] Computing feature matrices...")
    positive_matrix = compute_feature_matrix(positives, features)
    negative_matrix = compute_feature_matrix(negatives, features)

    cascade = CascadeClassifier(window_size=args.window_size, feature_count=len(features))
    labels_positive = np.ones(len(positive_matrix), dtype=np.int8)
    log_rows: list[dict] = []

    # 初始 active 负样本：从预计算矩阵里随机取
    init_count = min(args.active_negatives, len(negative_matrix))
    init_indices = rng.choice(len(negative_matrix), size=init_count, replace=False)
    active_negative_matrix = negative_matrix[init_indices]

    # ── 确定每级弱分类器数量 ──────────────────────────────
    if args.adaptive_stages:
        # 原版 V-J：动态决定每级需要多少弱分类器
        stage_plan = list(range(1, args.num_stages + 1))  # 占位，实际数量动态决定
        total_stages = args.num_stages
    else:
        stage_plan = args.stage_sizes
        total_stages = len(stage_plan)

    for stage_index_0, planned_weaks in enumerate(stage_plan):
        stage_index = stage_index_0 + 1

        if args.adaptive_stages:
            # ── 自适应模式：逐步加弱分类器直到 FP 率 ≤ target ──
            print(f"[INFO] Training adaptive stage {stage_index}/{total_stages} "
                  f"(target FP rate ≤ {args.target_fp_rate:.0%}, max {args.max_weaks_per_stage} weaks)...")
            train_matrix = np.vstack([positive_matrix, active_negative_matrix])
            labels_negative = -np.ones(len(active_negative_matrix), dtype=np.int8)
            labels = np.concatenate([labels_positive, labels_negative])

            # 从少量弱分类器开始，逐步加到 FP 率达标
            best_stage = None
            for try_weaks in [5, 10, 15, 20, 30, 40, 60, 80, 100, 130, 160, 200]:
                if try_weaks > args.max_weaks_per_stage:
                    break
                stage = train_adaboost(train_matrix, labels, features,
                                       num_weaks=try_weaks, batch_size=args.feature_batch_size)
                if not stage.weak_classifiers:
                    continue
                pos_scores = stage.decision_function(positive_matrix)
                neg_scores = stage.decision_function(active_negative_matrix)
                stage.threshold = choose_stage_threshold(pos_scores, args.stage_detection_rate)
                fp_rate = float(np.mean(neg_scores >= stage.threshold))
                print(f"[INFO]   try {try_weaks:3d} weaks → FP rate={fp_rate:.3f}", flush=True)
                best_stage = stage
                if fp_rate <= args.target_fp_rate:
                    break
            stage = best_stage
            weak_count = len(stage.weak_classifiers)
        else:
            # ── 固定模式 ──
            weak_count = planned_weaks
            print(f"[INFO] Training stage {stage_index}/{total_stages} with {weak_count} weak classifiers...")
            train_matrix = np.vstack([positive_matrix, active_negative_matrix])
            labels_negative = -np.ones(len(active_negative_matrix), dtype=np.int8)
            labels = np.concatenate([labels_positive, labels_negative])

            stage = train_adaboost(
                train_matrix, labels, features,
                num_weaks=weak_count, batch_size=args.feature_batch_size,
            )

        if not stage.weak_classifiers:
            raise RuntimeError(f"Stage {stage_index} failed to select any weak classifier")

        positive_scores = stage.decision_function(positive_matrix)
        negative_scores = stage.decision_function(active_negative_matrix)
        stage.threshold = choose_stage_threshold(
            positive_scores, args.stage_detection_rate,
            negative_scores=negative_scores if args.fp_constraint else None,
            max_fp_rate=0.5,
        )
        cascade.stages.append(stage)

        neg_fp_rate = float(np.mean(negative_scores >= stage.threshold))
        train_predictions = stage.predict_matrix(train_matrix)
        train_tp = int(np.sum(train_predictions[: len(positive_matrix)] == 1))
        train_fp = int(np.sum(train_predictions[len(positive_matrix) :] == 1))

        # ── 更新下一 stage 的 active 负样本 ──────────────────────────────
        # 1. 从固定负样本池里找通过当前级联的"难例"
        pool_predictions = cascade.predict_matrix(negative_matrix)
        pool_hard_indices = np.flatnonzero(pool_predictions == 1)
        pool_hard = negative_matrix[pool_hard_indices] if len(pool_hard_indices) > 0 else None
        hard_count = len(pool_hard_indices)
        mined_count = 0

        if not args.no_hnm:
            # 2. Hard Negative Mining：从真实图片中挖掘额外难例
            print(f"[INFO]   Mining hard negatives from images...", flush=True)
            mined_hard = mine_hard_negatives(
                args.negative_dir,
                cascade,
                features,
                args.window_size,
                target_count=args.active_negatives,
                per_image=args.negatives_per_image * 3,
                rng=rng,
            )
            mined_count = len(mined_hard) if mined_hard is not None else 0
        else:
            mined_hard = None

        # 3. 合并难例，不足时用简单负例填充
        hard_parts = [p for p in [mined_hard, pool_hard] if p is not None and len(p) > 0]
        if hard_parts:
            all_hard = np.vstack(hard_parts) if len(hard_parts) > 1 else hard_parts[0]
            if len(all_hard) >= args.active_negatives:
                chosen = rng.choice(len(all_hard), args.active_negatives, replace=False)
                active_negative_matrix = all_hard[chosen]
            else:
                remaining_count = args.active_negatives - len(all_hard)
                easy_indices = np.setdiff1d(np.arange(len(negative_matrix)), pool_hard_indices)
                if len(easy_indices) > 0 and remaining_count > 0:
                    fill = negative_matrix[rng.choice(easy_indices, min(remaining_count, len(easy_indices)), replace=False)]
                    active_negative_matrix = np.vstack([all_hard, fill])
                else:
                    active_negative_matrix = all_hard
        else:
            active_negative_matrix = negative_matrix[rng.choice(len(negative_matrix), init_count, replace=False)]

        log_rows.append(
            {
                "stage": stage_index,
                "weak_classifiers": len(stage.weak_classifiers),
                "threshold": f"{stage.threshold:.6f}",
                "stage_fp_rate": f"{neg_fp_rate:.4f}",
                "train_positive_pass": train_tp,
                "train_negative_pass": train_fp,
                "hard_negatives_pool": hard_count,
                "hard_negatives_mined": mined_count,
            }
        )
        hnm_str = f", mined hard={mined_count}" if not args.no_hnm else ""
        print(
            f"[INFO] Stage {stage_index}: {len(stage.weak_classifiers)} weaks, "
            f"DR={train_tp}/{len(positive_matrix)} ({train_tp/len(positive_matrix)*100:.1f}%), "
            f"FPR={neg_fp_rate:.1%}, "
            f"pool hard={hard_count}{hnm_str}"
        )

    output_path = Path(args.output)
    cascade.save(output_path)
    print(f"[INFO] Saved custom cascade model: {output_path}")
    if log_rows:
        save_training_log(log_rows, output_path)
    save_top_features(cascade)
    elapsed = time.perf_counter() - start_time
    print(f"[INFO] Training complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
