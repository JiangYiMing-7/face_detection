"""
prepare_caltech.py
==================
将 Caltech-101 Faces / Faces_easy 类别整理为本项目的测试集格式：
  data/test/caltech/images/    —— 图片（扁平化命名，避免重名）
  data/test/caltech/annotations.json  —— 标注（{filename: [[x,y,w,h]]}）

.mat 标注格式：box_coord = [[min_y, max_y, min_x, max_x]]  (行优先，1-indexed)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import scipy.io


ROOT = Path(__file__).resolve().parent
CALTECH_ROOT = Path(os.environ.get("CALTECH_ROOT", ROOT / "data" / "caltech-101" / "caltech-101"))

# (图片目录名, 标注目录名, 输出文件前缀)
CATEGORIES = [
    ("Faces",      "Faces_2", "faces"),
    ("Faces_easy", "Faces_3", "faces_easy"),
]


def parse_args() -> argparse.Namespace:
    """解析 Caltech 数据集根目录和输出目录参数。"""
    p = argparse.ArgumentParser(description="整理 Caltech-101 人脸测试集。")
    p.add_argument("--output-dir", default="data/test/caltech",
                   help="输出目录（图片+标注）")
    p.add_argument("--caltech-root", default=str(CALTECH_ROOT),
                   help="Caltech-101 根目录（含 101_ObjectCategories/ 和 Annotations/）")
    return p.parse_args()


def read_box(mat_path: Path) -> tuple[int, int, int, int] | None:
    """读取 .mat 标注，返回 (x, y, w, h)。

    Caltech .mat 格式：box_coord = [[min_y, max_y, min_x, max_x]]
    注意：原始坐标是 1-indexed，转 0-indexed 需减 1。
    """
    try:
        mat = scipy.io.loadmat(str(mat_path))
        coord = mat["box_coord"].flatten()          # [min_y, max_y, min_x, max_x]
        min_y, max_y, min_x, max_x = (int(v) for v in coord[:4])
        # 转为 0-indexed
        min_y -= 1; min_x -= 1
        x, y = min_x, min_y
        w = max(1, max_x - min_x)
        h = max(1, max_y - min_y)
        return (x, y, w, h)
    except Exception as exc:
        print(f"  [WARN] 无法读取 {mat_path.name}: {exc}")
        return None


def main() -> None:
    """把 Caltech 图片和 .mat 标注转换为项目统一测试集格式。"""
    args = parse_args()
    root = Path(args.caltech_root)
    out_dir = Path(args.output_dir)
    img_out = out_dir / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    annotations: dict[str, list] = {}
    total_images = 0
    total_boxes = 0

    for img_cat, ann_cat, prefix in CATEGORIES:
        img_src = root / "101_ObjectCategories" / img_cat
        ann_src = root / "Annotations" / ann_cat

        if not img_src.exists():
            print(f"[WARN] 图片目录不存在: {img_src}")
            continue
        if not ann_src.exists():
            print(f"[WARN] 标注目录不存在: {ann_src}")

        img_files = sorted(img_src.glob("*.jpg"))
        print(f"[INFO] {img_cat}: {len(img_files)} 张图片 (标注目录: {ann_cat})")

        for img_path in img_files:
            # 解析序号，例如 image_0001.jpg → 0001
            stem = img_path.stem  # 例如 "image_0001"
            number = stem.split("_")[-1]  # "0001"

            # 查找对应标注 .mat 文件
            ann_path = ann_src / f"annotation_{number}.mat"

            # 输出文件名：faces_0001.jpg / faces_easy_0001.jpg
            out_name = f"{prefix}_{number}.jpg"
            out_path = img_out / out_name

            # 复制图片
            shutil.copy2(img_path, out_path)

            # 解析标注
            if ann_path.exists():
                box = read_box(ann_path)
                if box is not None:
                    annotations[out_name] = [list(box)]
                    total_boxes += 1
                else:
                    annotations[out_name] = []
            else:
                # 没有标注文件：视作全图为人脸（近似）
                import cv2
                img = cv2.imread(str(img_path))
                if img is not None:
                    h, w = img.shape[:2]
                    annotations[out_name] = [[0, 0, w, h]]
                    total_boxes += 1
                else:
                    annotations[out_name] = []

            total_images += 1

    # 保存 annotations.json
    ann_out = out_dir / "annotations.json"
    ann_out.write_text(json.dumps(annotations, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"\n[INFO] 共处理 {total_images} 张图片，{total_boxes} 个标注框")
    print(f"[INFO] 图片已复制到: {img_out}")
    print(f"[INFO] 标注已保存到: {ann_out}")
    print(f"\n评估命令示例（Custom v1）：")
    print(f"  python evaluate.py --detector custom \\")
    print(f"    --model models/custom_cascade_v1_no_hnm.json \\")
    print(f"    --image-dir {out_dir}/images \\")
    print(f"    --annotations {out_dir}/annotations.json \\")
    print(f"    --output-dir results/eval_caltech_custom \\")
    print(f"    --min-size 30 --window-step 4 --scale-factor 1.2")


if __name__ == "__main__":
    main()
