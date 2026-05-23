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
    values = [int(item.strip()) for item in raw.split(",") if item.strip()]
    if not values or any(value <= 0 for value in values):
        raise argparse.ArgumentTypeError("stage sizes must be positive integers such as 10,20,40")
    return values


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def image_paths(directory: str | Path) -> list[Path]:
    root = Path(directory)
    if not root.exists():
        return []
    return sorted(path for path in root.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES)


def read_gray(path: Path) -> np.ndarray | None:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None or image.size == 0:
        return None
    return image


def normalize_window(gray: np.ndarray, window_size: int) -> np.ndarray:
    """把任意大小的训练 patch 统一缩放为 24x24 等固定窗口。"""
    resized = cv2.resize(gray, (window_size, window_size), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32)


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


def choose_stage_threshold(scores: np.ndarray, target_detection_rate: float) -> float:
    """按目标通过率选择 stage 阈值，优先减少正样本漏检。"""
    if scores.size == 0:
        return 0.0
    keep_rate = min(max(float(target_detection_rate), 0.5), 1.0)
    quantile = max(0.0, 1.0 - keep_rate)
    return float(np.quantile(scores, quantile))


def save_training_log(rows: list[dict], output_model: Path) -> None:
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
    active_negative_indices = rng.choice(
        len(negative_matrix),
        size=min(args.active_negatives, len(negative_matrix)),
        replace=False,
    )

    for stage_index, weak_count in enumerate(args.stage_sizes, start=1):
        print(f"[INFO] Training stage {stage_index}/{len(args.stage_sizes)} with {weak_count} weak classifiers...")
        active_negative_matrix = negative_matrix[active_negative_indices]
        train_matrix = np.vstack([positive_matrix, active_negative_matrix])
        labels_negative = -np.ones(len(active_negative_matrix), dtype=np.int8)
        labels = np.concatenate([labels_positive, labels_negative])

        stage = train_adaboost(
            train_matrix,
            labels,
            features,
            num_weaks=weak_count,
            batch_size=args.feature_batch_size,
        )
        if not stage.weak_classifiers:
            raise RuntimeError(f"Stage {stage_index} failed to select any weak classifier")

        positive_scores = stage.decision_function(positive_matrix)
        stage.threshold = choose_stage_threshold(positive_scores, args.stage_detection_rate)
        cascade.stages.append(stage)

        train_predictions = stage.predict_matrix(train_matrix)
        train_tp = int(np.sum(train_predictions[: len(positive_matrix)] == 1))
        train_fp = int(np.sum(train_predictions[len(positive_matrix) :] == 1))

        pool_predictions = cascade.predict_matrix(negative_matrix)
        hard_negative_indices = np.flatnonzero(pool_predictions == 1)
        hard_count = len(hard_negative_indices)
        if hard_count >= args.active_negatives:
            active_negative_indices = rng.choice(hard_negative_indices, size=args.active_negatives, replace=False)
        else:
            remaining = np.setdiff1d(np.arange(len(negative_matrix)), hard_negative_indices, assume_unique=False)
            fill_count = min(args.active_negatives - hard_count, len(remaining))
            fill_indices = rng.choice(remaining, size=fill_count, replace=False) if fill_count else np.array([], dtype=int)
            active_negative_indices = np.concatenate([hard_negative_indices, fill_indices]).astype(int)

        log_rows.append(
            {
                "stage": stage_index,
                "weak_classifiers": len(stage.weak_classifiers),
                "threshold": f"{stage.threshold:.6f}",
                "train_positive_pass": train_tp,
                "train_negative_pass": train_fp,
                "hard_negatives": hard_count,
            }
        )
        print(
            f"[INFO] Stage {stage_index}: positive pass {train_tp}/{len(positive_matrix)}, "
            f"stage false positives {train_fp}/{len(active_negative_matrix)}, hard negatives {hard_count}"
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
