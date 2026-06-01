"""带 OpenCV 调参控件的单检测器交互式人脸检测演示。"""

from __future__ import annotations

import argparse
import csv
import platform
import time
from pathlib import Path

import cv2

from src.cascade_paths import haarcascade_path
from src.detectors import create_detector


WINDOW_NAME = "Viola-Jones Face Detection"
CONTROL_WINDOW = "Controls"
PROJECT_ROOT = Path(__file__).resolve().parent


def project_path(path: str | Path) -> Path:
    """Resolve relative CLI paths from the repository root."""
    candidate = Path(path)
    return candidate if candidate.is_absolute() else PROJECT_ROOT / candidate


def parse_args() -> argparse.Namespace:
    """解析交互式演示脚本的命令行参数。"""
    parser = argparse.ArgumentParser(description="使用 OpenCV 或自训练 cascade 进行实时人脸检测。")
    parser.add_argument("--source", default="0", help="摄像头编号（如 0），或视频/图片路径。默认：0")
    parser.add_argument("--detector", choices=["opencv", "custom"], default="opencv", help="检测器后端。")
    parser.add_argument("--model", default="models/custom_cascade.json", help="自训练 cascade 模型路径。")
    parser.add_argument("--scale-factor", type=float, default=None, help="图像金字塔缩放因子。")
    parser.add_argument("--min-neighbors", type=int, default=4, help="OpenCV Haar minNeighbors。默认：4")
    parser.add_argument("--min-size", type=int, default=30, help="最小人脸尺寸（像素）。默认：30")
    parser.add_argument("--window-step", type=int, default=4, help="自实现检测器滑窗步长。默认：4")
    parser.add_argument("--nms-threshold", type=float, default=0.3, help="自实现检测器 NMS IoU 阈值。默认：0.3")
    parser.add_argument("--equalize", action="store_true", help="检测前启用直方图均衡化。")
    parser.add_argument("--no-display", action="store_true", help="不打开 OpenCV 窗口，只保存结果。")
    parser.add_argument("--width", type=int, default=960, help="摄像头帧宽度。默认：960")
    parser.add_argument("--height", type=int, default=540, help="摄像头帧高度。默认：540")
    return parser.parse_args()


def resolve_defaults(args: argparse.Namespace) -> argparse.Namespace:
    """根据检测器类型补默认参数：OpenCV 和手写检测器使用不同默认尺度。"""
    if args.scale_factor is None:
        args.scale_factor = 1.1 if args.detector == "opencv" else 1.2
    return args


def open_source(source: str, width: int, height: int):
    """打开摄像头、视频或图片输入源。"""
    if source.isdigit():
        camera_index = int(source)
        if platform.system() == "Windows":
            cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        return cap, False

    path = project_path(source)
    if not path.exists():
        raise FileNotFoundError(f"Input source does not exist: {source}")

    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        image = cv2.imread(str(path))
        if image is None:
            raise RuntimeError(f"Failed to read image: {source}")
        return image, True

    return cv2.VideoCapture(str(path)), False


def ensure_output_dirs():
    """创建截图、人脸裁剪和运行日志输出目录。"""
    root = PROJECT_ROOT / "results"
    screenshots = root / "screenshots"
    faces = root / "faces"
    logs = root / "logs"
    screenshots.mkdir(parents=True, exist_ok=True)
    faces.mkdir(parents=True, exist_ok=True)
    logs.mkdir(parents=True, exist_ok=True)
    return screenshots, faces, logs


def create_controls(args: argparse.Namespace) -> None:
    """创建 OpenCV 滑条窗口，用于实时调参。"""
    cv2.namedWindow(CONTROL_WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(CONTROL_WINDOW, 500, 260)
    cv2.createTrackbar("scale x100", CONTROL_WINDOW, int(args.scale_factor * 100), 160, lambda _: None)
    cv2.createTrackbar("minNeighbors", CONTROL_WINDOW, args.min_neighbors, 20, lambda _: None)
    cv2.createTrackbar("minSize", CONTROL_WINDOW, args.min_size, 240, lambda _: None)
    cv2.createTrackbar("window step", CONTROL_WINDOW, args.window_step, 16, lambda _: None)
    cv2.createTrackbar("equalize", CONTROL_WINDOW, int(args.equalize), 1, lambda _: None)
    cv2.createTrackbar("eyes", CONTROL_WINDOW, 1, 1, lambda _: None)
    cv2.createTrackbar("privacy blur", CONTROL_WINDOW, 0, 1, lambda _: None)


def read_controls(args: argparse.Namespace) -> dict:
    """从滑条窗口读取当前参数，并统一成检测器 options。"""
    scale_raw = cv2.getTrackbarPos("scale x100", CONTROL_WINDOW)
    min_neighbors = cv2.getTrackbarPos("minNeighbors", CONTROL_WINDOW)
    min_size = cv2.getTrackbarPos("minSize", CONTROL_WINDOW)
    window_step = cv2.getTrackbarPos("window step", CONTROL_WINDOW)
    return {
        "detector": args.detector,
        "scale_factor": max(1.05, scale_raw / 100.0),
        "min_neighbors": max(1, min_neighbors),
        "min_size": max(20, min_size),
        "window_step": max(1, window_step),
        "nms_threshold": args.nms_threshold,
        "equalize": cv2.getTrackbarPos("equalize", CONTROL_WINDOW) == 1,
        "detect_eyes": cv2.getTrackbarPos("eyes", CONTROL_WINDOW) == 1,
        "privacy_blur": cv2.getTrackbarPos("privacy blur", CONTROL_WINDOW) == 1,
    }


def static_options(args: argparse.Namespace) -> dict:
    """在无显示模式或单张图片模式下生成固定检测参数。"""
    return {
        "detector": args.detector,
        "scale_factor": args.scale_factor,
        "min_neighbors": args.min_neighbors,
        "min_size": args.min_size,
        "window_step": args.window_step,
        "nms_threshold": args.nms_threshold,
        "equalize": args.equalize,
        "detect_eyes": True,
        "privacy_blur": False,
    }


def load_eye_classifier():
    """加载 OpenCV 眼睛检测器，用于在人脸框内做辅助标记。"""
    path = haarcascade_path("haarcascade_eye.xml")
    classifier = cv2.CascadeClassifier(path)
    if classifier.empty():
        raise RuntimeError(f"Failed to load eye cascade: {path}")
    return classifier


def detect_eyes(gray, faces, eye_classifier, enabled: bool):
    """在每个人脸 ROI 内检测最多两个眼睛框。"""
    if not enabled:
        return []

    eyes_by_face = []
    for x, y, w, h in faces:
        face_gray = gray[y : y + h, x : x + w]
        eyes = eye_classifier.detectMultiScale(
            face_gray,
            scaleFactor=1.1,
            minNeighbors=8,
            minSize=(12, 12),
            flags=cv2.CASCADE_SCALE_IMAGE,
        )
        absolute_eyes = [(x + ex, y + ey, ew, eh) for ex, ey, ew, eh in eyes[:2]]
        eyes_by_face.append(absolute_eyes)
    return eyes_by_face


def blur_faces(frame, faces) -> None:
    """对检测到的人脸区域做高斯模糊，演示隐私遮挡效果。"""
    for x, y, w, h in faces:
        roi = frame[y : y + h, x : x + w]
        if roi.size == 0:
            continue
        k = max(21, (min(w, h) // 3) | 1)
        frame[y : y + h, x : x + w] = cv2.GaussianBlur(roi, (k, k), 0)


def draw_overlay(frame, faces, eyes_by_face, fps: float, options: dict) -> None:
    """在视频帧上绘制检测框、眼睛位置和当前运行参数。"""
    face_color = (0, 255, 255)
    eye_color = (255, 180, 0)
    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), face_color, 2)
        cv2.putText(frame, "Face", (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, face_color, 2, cv2.LINE_AA)

    for eyes in eyes_by_face:
        for ex, ey, ew, eh in eyes:
            center = (ex + ew // 2, ey + eh // 2)
            radius = max(6, min(ew, eh) // 2)
            cv2.circle(frame, center, radius, eye_color, 2)

    lines = [
        f"Detector: {options['detector']}",
        f"Faces: {len(faces)} | FPS: {fps:.1f}",
        f"scaleFactor: {options['scale_factor']:.2f}",
        f"minSize: {options['min_size']} | step: {options['window_step']}",
        f"equalize: {'on' if options['equalize'] else 'off'} | eyes: {'on' if options['detect_eyes'] else 'off'}",
        f"privacy: {'on' if options['privacy_blur'] else 'off'}",
        "s: screenshot | f: save faces | q/esc: quit",
    ]

    y = 28
    for line in lines:
        cv2.putText(frame, line, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 255, 255), 2, cv2.LINE_AA)
        y += 27


def save_face_crops(frame, faces, faces_dir: Path) -> int:
    """把当前帧中的人脸裁剪保存到 results/faces。"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    saved = 0
    for index, (x, y, w, h) in enumerate(faces, start=1):
        crop = frame[y : y + h, x : x + w]
        if crop.size == 0:
            continue
        cv2.imwrite(str(faces_dir / f"face_{timestamp}_{index}.jpg"), crop)
        saved += 1
    return saved


def write_log_header(log_path: Path):
    """创建 CSV 日志文件，并写入统一表头。"""
    handle = log_path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    writer.writerow(
        [
            "timestamp",
            "detector",
            "fps",
            "face_count",
            "scale_factor",
            "min_neighbors",
            "min_size",
            "window_step",
            "equalize",
            "eyes",
            "privacy_blur",
        ]
    )
    return handle, writer


def run_detection(detector, frame, eye_classifier, options: dict, fps: float):
    """执行一次检测，并生成带可视化叠加信息的显示帧。"""
    gray, faces = detector.detect(frame, options)
    eyes_by_face = detect_eyes(gray, faces, eye_classifier, options["detect_eyes"])
    display_frame = frame.copy()
    if options["privacy_blur"]:
        blur_faces(display_frame, faces)
    draw_overlay(display_frame, faces, eyes_by_face, fps, options)
    return display_frame, faces


def main() -> None:
    """运行实时或单图检测演示主循环。"""
    args = resolve_defaults(parse_args())
    screenshots_dir, faces_dir, logs_dir = ensure_output_dirs()

    print(f"[INFO] Loading detector: {args.detector}", flush=True)
    detector = create_detector(args.detector, project_path(args.model))
    eye_classifier = load_eye_classifier()

    print(f"[INFO] Opening source: {args.source}", flush=True)
    source, is_image = open_source(args.source, args.width, args.height)
    log_path = logs_dir / f"{args.detector}_log.csv"
    log_handle, log_writer = write_log_header(log_path)

    try:
        if is_image:
            frame = source.copy()
            options = static_options(args)
            start = time.perf_counter()
            display_frame, faces = run_detection(detector, frame, eye_classifier, options, 0.0)
            elapsed = time.perf_counter() - start
            log_writer.writerow(
                [
                    time.strftime("%Y-%m-%d %H:%M:%S"),
                    args.detector,
                    f"{(1.0 / elapsed) if elapsed > 0 else 0.0:.2f}",
                    len(faces),
                    f"{options['scale_factor']:.2f}",
                    options["min_neighbors"],
                    options["min_size"],
                    options["window_step"],
                    int(options["equalize"]),
                    int(options["detect_eyes"]),
                    int(options["privacy_blur"]),
                ]
            )
            log_handle.flush()
            filename = time.strftime(f"{args.detector}_image_result_%Y%m%d_%H%M%S.jpg")
            output_path = screenshots_dir / filename
            cv2.imwrite(str(output_path), display_frame)
            print(f"[INFO] Saved image result: {output_path}", flush=True)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, display_frame)
                cv2.waitKey(0)
            return

        if not source.isOpened():
            raise RuntimeError(f"Failed to open video source: {args.source}")

        print("[INFO] Source opened. Use the Controls window to tune parameters.", flush=True)
        print("[INFO] Press s to save a screenshot, f to save face crops, q or Esc to quit.", flush=True)

        if not args.no_display:
            cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
            create_controls(args)
        last_time = time.perf_counter()
        fps = 0.0
        frame_index = 0

        while True:
            ok, frame = source.read()
            if not ok:
                print("[INFO] End of source or frame read failed; exiting.", flush=True)
                break

            now = time.perf_counter()
            elapsed = now - last_time
            last_time = now
            if elapsed > 0:
                fps = 0.9 * fps + 0.1 * (1.0 / elapsed) if fps > 0 else 1.0 / elapsed

            options = read_controls(args) if not args.no_display else static_options(args)
            display_frame, faces = run_detection(detector, frame, eye_classifier, options, fps)
            if not args.no_display:
                cv2.imshow(WINDOW_NAME, display_frame)

            if frame_index % 15 == 0:
                log_writer.writerow(
                    [
                        time.strftime("%Y-%m-%d %H:%M:%S"),
                        args.detector,
                        f"{fps:.2f}",
                        len(faces),
                        f"{options['scale_factor']:.2f}",
                        options["min_neighbors"],
                        options["min_size"],
                        options["window_step"],
                        int(options["equalize"]),
                        int(options["detect_eyes"]),
                        int(options["privacy_blur"]),
                    ]
                )
                log_handle.flush()
            frame_index += 1

            if args.no_display:
                continue

            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
            if key == ord("s"):
                filename = time.strftime(f"{args.detector}_screenshot_%Y%m%d_%H%M%S.jpg")
                cv2.imwrite(str(screenshots_dir / filename), display_frame)
                print(f"[INFO] Saved screenshot: {screenshots_dir / filename}", flush=True)
            if key == ord("f"):
                saved = save_face_crops(frame, faces, faces_dir)
                print(f"[INFO] Saved {saved} face crop(s) to: {faces_dir}", flush=True)

    finally:
        log_handle.close()
        if not is_image:
            source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
