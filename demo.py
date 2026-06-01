"""
实时人脸检测演示脚本
====================
左半屏：OpenCV 预训练 Haar Cascade（基线）
右半屏：自实现 Viola-Jones Cascade

操作：
  q        退出
  r        开始 / 停止录制（保存到 results/demo_output.mp4）
  空格      暂停 / 继续
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Windows 控制台 UTF-8
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.detectors import create_detector


def project_path(path: str | Path) -> Path:
    """Resolve relative CLI paths from the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def parse_args() -> argparse.Namespace:
    """解析双屏实时演示所需的摄像头、模型和后处理参数。"""
    p = argparse.ArgumentParser(description="实时人脸检测演示（双屏对比）")
    p.add_argument("--model", default="models/custom_cascade_v3_hnm.json",
                   help="自实现级联模型路径")
    p.add_argument("--camera", type=int, default=0, help="摄像头编号")
    p.add_argument("--scale-factor", type=float, default=1.2)
    p.add_argument("--min-size", type=int, default=40,
                   help="最小检测窗口（越大越快）")
    p.add_argument("--window-step", type=int, default=6,
                   help="滑窗步长（越大越快，4=精细 8=实时）")
    p.add_argument("--nms-threshold", type=float, default=0.45)
    p.add_argument("--variance-normalize", action="store_true",
                   help="开启方差归一化（默认关闭；模型在原始像素值上训练）")
    p.add_argument("--output", default="results/demo_output.mp4",
                   help="录制输出路径")
    p.add_argument("--width", type=int, default=640, help="摄像头宽度")
    p.add_argument("--height", type=int, default=480, help="摄像头高度")
    p.add_argument("--min-neighbors-custom", type=int, default=3,
                   help="自实现检测器 min_neighbors（v3建议3，v6建议10）")
    p.add_argument("--score-threshold", type=float, default=5.0,
                   help="累积分数阈值（v3建议5.0，v6建议0）")
    p.add_argument("--opencv-gate", action="store_true",
                   help="仅把 OpenCV 基线作为自实现检测框的可选调试门控。")
    return p.parse_args()


class BoxSmoother:
    """时间平滑 + 跳变抑制：指数移动平均减少抖动，大幅跳变视为误检。"""
    def __init__(self, alpha: float = 0.4, max_lost: int = 5, max_jump: float = 0.4):
        """初始化单目标框平滑器。"""
        self.alpha = alpha        # 平滑系数，越小越平滑
        self.max_lost = max_lost  # 连续多少帧没检测到就清空
        self.max_jump = max_jump  # 中心点跳变超过框尺寸的此比例则视为误检
        self.smooth_box = None
        self.lost_count = 0

    def update(self, boxes: list) -> list:
        """输入当前帧候选框，输出平滑后的主框列表。"""
        if not boxes:
            self.lost_count += 1
            if self.lost_count >= self.max_lost:
                self.smooth_box = None
                return []
            # 短暂丢失时保持上一次的框
            return [self.smooth_box] if self.smooth_box else []

        self.lost_count = 0
        # 取面积最大的框作为主检测
        best = max(boxes, key=lambda b: b[2] * b[3])
        if self.smooth_box is None:
            self.smooth_box = best
        else:
            sx, sy, sw, sh = self.smooth_box
            bx, by, bw, bh = best
            # 跳变抑制：新框中心与旧框中心距离过大则忽略
            old_cx, old_cy = sx + sw / 2, sy + sh / 2
            new_cx, new_cy = bx + bw / 2, by + bh / 2
            dist = ((new_cx - old_cx) ** 2 + (new_cy - old_cy) ** 2) ** 0.5
            ref_size = max(sw, sh)
            if dist > ref_size * self.max_jump:
                # 跳变太大，视为误检，保持旧框
                return [self.smooth_box]
            # 指数移动平均
            a = self.alpha
            self.smooth_box = (
                int(sx + a * (bx - sx)),
                int(sy + a * (by - sy)),
                int(sw + a * (bw - sw)),
                int(sh + a * (bh - sh)),
            )
        return [self.smooth_box]


class TrackingBoxSmoother:
    """多目标人脸框的时间平滑器。"""

    def __init__(self, alpha: float = 0.65, max_lost: int = 2, max_tracks: int = 8):
        """初始化多目标轨迹平滑器。"""
        self.alpha = alpha
        self.max_lost = max_lost
        self.max_tracks = max_tracks
        self.tracks: list[dict] = []

    @staticmethod
    def _center_distance(a: tuple, b: tuple) -> float:
        """计算两个框中心点之间的距离。"""
        ax, ay, aw, ah = a
        bx, by, bw, bh = b
        acx, acy = ax + aw / 2, ay + ah / 2
        bcx, bcy = bx + bw / 2, by + bh / 2
        return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5

    @staticmethod
    def _smooth(old_box: tuple, new_box: tuple, alpha: float) -> tuple:
        """用指数移动平均把新检测框融合到旧轨迹框。"""
        sx, sy, sw, sh = old_box
        bx, by, bw, bh = new_box
        return (
            int(round(sx + alpha * (bx - sx))),
            int(round(sy + alpha * (by - sy))),
            int(round(sw + alpha * (bw - sw))),
            int(round(sh + alpha * (bh - sh))),
        )

    @staticmethod
    def _match_score(track_box: tuple, detection: tuple) -> float:
        """综合 IoU 和中心距离，衡量检测框与已有轨迹的匹配程度。"""
        iou = box_iou(track_box, detection)
        dist = TrackingBoxSmoother._center_distance(track_box, detection)
        ref_size = max(track_box[2], track_box[3], detection[2], detection[3], 1)
        return iou - 0.25 * (dist / ref_size)

    def update(self, boxes: list) -> list:
        """根据当前帧检测结果更新多目标轨迹并返回稳定框。"""
        detections = sorted(boxes, key=lambda b: b[2] * b[3], reverse=True)[: self.max_tracks]
        unmatched = set(range(len(detections)))

        for track in self.tracks:
            track["matched"] = False
            if not unmatched:
                break
            best_idx = max(unmatched, key=lambda idx: self._match_score(track["box"], detections[idx]))
            best_box = detections[best_idx]
            iou = box_iou(track["box"], best_box)
            dist = self._center_distance(track["box"], best_box)
            ref_size = max(track["box"][2], track["box"][3], best_box[2], best_box[3], 1)
            if iou >= 0.08 or dist <= ref_size * 0.9:
                alpha = 0.9 if dist > ref_size * 0.45 else self.alpha
                track["box"] = self._smooth(track["box"], best_box, alpha)
                track["lost"] = 0
                track["matched"] = True
                unmatched.remove(best_idx)

        for idx in unmatched:
            self.tracks.append({"box": tuple(detections[idx][:4]), "lost": 0, "matched": True})

        kept = []
        for track in self.tracks:
            if not track["matched"]:
                track["lost"] += 1
            if track["lost"] <= self.max_lost:
                kept.append(track)
        self.tracks = kept[: self.max_tracks]
        return [track["box"] for track in self.tracks if track["lost"] == 0]


def box_iou(a: tuple, b: tuple) -> float:
    """计算两个框的 IoU。"""
    ax, ay, aw, ah = a[:4]
    bx, by, bw, bh = b[:4]
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = aw * ah + bw * bh - inter
    return inter / union if union else 0.0


def center_in_expanded_box(box: tuple, reference: tuple, expand: float = 0.65) -> bool:
    """判断候选框中心是否落在扩大后的参考框范围内。"""
    x, y, w, h = box[:4]
    rx, ry, rw, rh = reference[:4]
    cx, cy = x + w / 2, y + h / 2
    margin_x = rw * expand
    margin_y = rh * expand
    return (
        rx - margin_x <= cx <= rx + rw + margin_x
        and ry - margin_y <= cy <= ry + rh + margin_y
    )


def filter_custom_boxes(custom_boxes: list, opencv_boxes: list, frame_shape: tuple) -> list:
    """过滤越界、过小或与 OpenCV 调试门控不一致的自实现检测框。"""
    h, w = frame_shape[:2]
    filtered = []
    for box in custom_boxes:
        x, y, bw, bh = box[:4]
        if bw < 24 or bh < 24:
            continue
        if x < 0 or y < 0 or x + bw > w or y + bh > h:
            continue
        if opencv_boxes:
            matched = any(
                box_iou(box, ref) >= 0.05 or center_in_expanded_box(box, ref)
                for ref in opencv_boxes
            )
            if not matched:
                continue
        filtered.append(box)
    return filtered


def draw_detections(frame: np.ndarray, boxes: list, color: tuple, label: str) -> None:
    """在半屏画面上绘制检测框和数量标签。"""
    for x, y, w, h in boxes:
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(frame, f"{label}: {len(boxes)} face(s)",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)


def draw_fps(frame: np.ndarray, fps: float, color=(255, 255, 255)) -> None:
    """在画面左下角绘制当前 FPS。"""
    cv2.putText(frame, f"FPS: {fps:.1f}",
                (10, frame.shape[0] - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)


def put_title(frame: np.ndarray, title: str, color: tuple) -> None:
    """在半屏底部绘制检测器标题。"""
    h, w = frame.shape[:2]
    cv2.putText(frame, title,
                (w // 2 - len(title) * 7, h - 12),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def main() -> None:
    """运行 OpenCV 与自实现 cascade 的双屏实时对比演示。"""
    args = parse_args()
    output_path = project_path(args.output)

    # ── 加载检测器 ──────────────────────────────────────────
    print("[INFO] 加载 OpenCV 检测器...")
    opencv_det = create_detector("opencv")

    model_path = project_path(args.model)
    if not model_path.exists():
        # 自动寻找可用模型
        candidates = list((ROOT / "models").glob("*.json"))
        if candidates:
            model_path = sorted(candidates)[-1]
            print(f"[WARN] 指定模型不存在，改用 {model_path}")
        else:
            print("[FAIL] 未找到任何模型文件，请先运行 train.py")
            sys.exit(1)

    print(f"[INFO] 加载自实现模型: {model_path}")
    custom_det = create_detector("custom", model_path)

    options_opencv = {
        "scale_factor": args.scale_factor,
        "min_neighbors": 4,
        "min_size": args.min_size,
    }
    options_custom = {
        "scale_factor": args.scale_factor,
        "min_size": args.min_size,
        "window_step": args.window_step,
        "nms_threshold": args.nms_threshold,
        "variance_normalize": args.variance_normalize,
        "min_neighbors": args.min_neighbors_custom,
        "score_threshold": args.score_threshold,
    }

    # ── 打开摄像头 ──────────────────────────────────────────
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    if not cap.isOpened():
        print(f"[FAIL] 无法打开摄像头 {args.camera}")
        sys.exit(1)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[INFO] 摄像头分辨率: {actual_w}×{actual_h}")
    print("[INFO] 按 r 录制 | 空格 暂停 | q 退出")

    # ── 录制设置 ─────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_w = actual_w * 2  # 左右拼接
    writer: cv2.VideoWriter | None = None
    recording = False

    # ── 主循环 ───────────────────────────────────────────────
    fps_smooth = 30.0
    paused = False
    frame_cache = None
    custom_smoother = TrackingBoxSmoother(alpha=0.65, max_lost=2, max_tracks=8)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            paused = not paused
            print("[INFO]", "暂停" if paused else "继续")
        if key == ord("r"):
            if not recording:
                writer = cv2.VideoWriter(str(output_path), fourcc, 20,
                                         (out_w, actual_h))
                recording = True
                print(f"[INFO] 开始录制 -> {output_path}")
            else:
                if writer:
                    writer.release()
                    writer = None
                recording = False
                print(f"[INFO] 录制停止，保存至 {output_path}")

        if paused:
            if frame_cache is not None:
                cv2.imshow("Face Detection Demo", frame_cache)
            continue

        ret, frame = cap.read()
        if not ret:
            print("[WARN] 读取帧失败")
            continue

        t0 = time.perf_counter()

        # ── OpenCV 检测 ──────────────────────────────────────
        left = frame.copy()
        _, opencv_boxes = opencv_det.detect(frame, options_opencv)
        draw_detections(left, opencv_boxes, (0, 220, 0), "OpenCV")
        put_title(left, "OpenCV Haar Cascade (Baseline)", (0, 220, 0))

        # ── 自实现检测 ───────────────────────────────────────
        right = frame.copy()
        _, custom_boxes = custom_det.detect(frame, options_custom)
        if args.opencv_gate:
            custom_boxes = filter_custom_boxes(custom_boxes, opencv_boxes, frame.shape)
        custom_boxes = custom_smoother.update(custom_boxes)
        draw_detections(right, custom_boxes, (0, 200, 255), "Custom")
        put_title(right, "Custom Viola-Jones (Ours)", (0, 200, 255))

        elapsed = time.perf_counter() - t0
        fps_inst = 1.0 / elapsed if elapsed > 0 else 999
        fps_smooth = 0.85 * fps_smooth + 0.15 * fps_inst

        draw_fps(left, fps_smooth)
        draw_fps(right, fps_smooth)

        # ── 分隔线 + 拼接 ────────────────────────────────────
        combined = np.hstack([left, right])
        cv2.line(combined, (actual_w, 0), (actual_w, actual_h), (180, 180, 180), 2)

        # 录制状态指示
        if recording:
            cv2.circle(combined, (out_w - 20, 20), 8, (0, 0, 255), -1)
            cv2.putText(combined, "REC", (out_w - 55, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
            if writer:
                writer.write(combined)

        frame_cache = combined
        cv2.imshow("Face Detection Demo", combined)

    # ── 清理 ─────────────────────────────────────────────────
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    print("[INFO] 演示结束")


if __name__ == "__main__":
    main()
