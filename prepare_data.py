"""
数据准备脚本 — Viola-Jones 人脸检测大作业
==========================================
数据来源（与设计文档一致）：

  训练正样本  → WIDER FACE val 中裁出的人脸 crop（~363 MB，一次下载）
  训练负样本  → Oxford DTD 纹理数据集（已完成）
  测试图片    → WIDER FACE val 另一子集（同一 zip，无需额外下载）
  测试标注    → WIDER FACE split annotations（3.6 MB，已完成）

运行：
  python prepare_data.py
"""

from __future__ import annotations

import json
import random
import shutil
import sys
import tarfile
import time
import urllib.request
import zipfile
from pathlib import Path

import cv2
import numpy as np

# Windows 控制台 UTF-8
if sys.platform == "win32":
    import io as _io
    sys.stdout = _io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = _io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# ──────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
TRAIN_POS = ROOT / "data/train/positives"
TRAIN_NEG = ROOT / "data/train/negatives"
TEST_IMGS = ROOT / "data/test/images"
TEST_ANN  = ROOT / "data/test/annotations.json"
TMP       = ROOT / ".data_tmp"

for d in (TRAIN_POS, TRAIN_NEG, TEST_IMGS, TMP):
    d.mkdir(parents=True, exist_ok=True)

HF_BASE = "https://huggingface.co/datasets"

# ── 规模参数 ──────────────────────────────────────────────
MAX_POSITIVES    = 3000   # 训练正样本数量上限（LFW + WIDER FACE 各一半）
MAX_TEST_IMAGES  = 200    # 测试集图片数量上限
TARGET_NEGATIVES = 300    # 目标负样本数量

LFW_TAR   = Path(r"D:\aaa大三下作业\计算机视觉\face_detection\data\lfw_aug.tar")
LFW_QUOTA = MAX_POSITIVES // 2   # LFW 占一半
WF_QUOTA  = MAX_POSITIVES - LFW_QUOTA  # WIDER FACE 大脸占一半


# ──────────────────────────────────────────────────────────
# 工具函数
# ──────────────────────────────────────────────────────────
def _progress(block_num, block_size, total_size):
    """打印 urllib 下载进度条。"""
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(downloaded / total_size * 100, 100)
        bar = "#" * int(pct / 2) + "-" * (50 - int(pct / 2))
        print(f"\r  [{bar}] {pct:5.1f}%  "
              f"{downloaded/1024/1024:.1f}/{total_size/1024/1024:.1f} MB",
              end="", flush=True)


def download(url: str, dest: Path, desc: str = "") -> bool:
    """下载远程文件到本地临时目录，失败时删除残留文件。"""
    print(f"\n[下载] {desc or dest.name}")
    print(f"  URL : {url}")
    try:
        req = urllib.request.Request(
            url, headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req, timeout=300) as resp, \
             open(dest, "wb") as f:
            total = int(resp.headers.get("Content-Length", 0))
            blk, n = 65536, 0
            while True:
                chunk = resp.read(blk)
                if not chunk:
                    break
                f.write(chunk)
                n += 1
                _progress(n, blk, total)
        print()
        return True
    except Exception as exc:
        print(f"\n  [FAIL] {exc}")
        dest.unlink(missing_ok=True)
        return False


def extract_zip(zip_path: Path, dest: Path) -> None:
    """解压 zip 文件到目标目录。"""
    print(f"[解压] {zip_path.name} -> {dest.name}/")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(dest)


def count_images(d: Path) -> int:
    """统计目录下常见图片格式文件数量。"""
    if not d.exists():
        return 0
    return sum(1 for f in d.rglob("*")
               if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"})


# ──────────────────────────────────────────────────────────
# 核心：下载 WIDER FACE val（正样本 + 测试集共用）
# ──────────────────────────────────────────────────────────
def ensure_wider_val() -> Path | None:
    """
    返回解压后的 WIDER_val/images/ 目录。
    已存在则直接返回，否则先下载 zip 再解压。
    """
    val_dir   = TMP / "WIDER_val_extracted"
    imgs_root = val_dir / "WIDER_val" / "images"

    if imgs_root.exists() and count_images(imgs_root) > 100:
        print(f"[跳过] WIDER FACE val 已解压（{count_images(imgs_root)} 张）")
        return imgs_root

    val_zip = TMP / "WIDER_val.zip"
    if not val_zip.exists():
        url = (f"{HF_BASE}/CUHK-CSE/wider_face/resolve/main"
               f"/data/WIDER_val.zip?download=true")
        if not download(url, val_zip, "WIDER FACE val 图片 (363 MB)"):
            return None

    extract_zip(val_zip, val_dir)
    val_zip.unlink(missing_ok=True)

    if not imgs_root.exists() or count_images(imgs_root) < 100:
        print(f"  [FAIL] 解压后未找到图片目录: {imgs_root}")
        return None
    return imgs_root


# ──────────────────────────────────────────────────────────
# 读取 WIDER FACE val 标注
# ──────────────────────────────────────────────────────────
def load_wider_annotations() -> list[tuple[str, list]]:
    """
    返回 [(img_rel_path, [[x,y,w,h], ...]), ...]
    img_rel_path 形如 "0--Parade/xxx.jpg"
    """
    ann_file = TMP / "wider_face_split" / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    if not ann_file.exists():
        ann_file = TMP / "wider_face_split" / "wider_face_val_bbx_gt.txt"
    if not ann_file.exists():
        print(f"  [FAIL] 找不到标注文件，请先确认 wider_face_split 已下载")
        return []

    results = []
    lines = ann_file.read_text(encoding="utf-8").splitlines()
    i = 0
    while i < len(lines):
        img_rel = lines[i].strip()
        i += 1
        if i >= len(lines):
            break
        try:
            n = int(lines[i].strip())
        except ValueError:
            continue
        i += 1
        boxes = []
        for _ in range(n):
            if i >= len(lines):
                break
            parts = lines[i].strip().split()
            i += 1
            if len(parts) < 4:
                continue
            try:
                x, y, w, h = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
                if w > 5 and h > 5:          # 过滤极小人脸
                    boxes.append([x, y, w, h])
            except ValueError:
                continue
        if boxes:
            results.append((img_rel, boxes))
    return results


# ══════════════════════════════════════════════════════════
# 步骤 1a：训练正样本 — LFW（优先）
# ══════════════════════════════════════════════════════════
def prepare_positives_lfw() -> bool:
    """从 LFW tar 提取对齐正脸图，返回 True 表示成功填满正样本。"""
    if not LFW_TAR.exists():
        return False

    existing = count_images(TRAIN_POS)
    if existing >= LFW_QUOTA:
        print(f"[跳过] LFW 正样本已有 {existing} 张（配额 {LFW_QUOTA}）")
        return True

    lfw_dir = TMP / "lfw_extracted"
    if not (lfw_dir / "lfw_aug").exists():
        print(f"[解压] {LFW_TAR.name} -> {lfw_dir.name}/  (约 1-2 分钟)")
        lfw_dir.mkdir(parents=True, exist_ok=True)
        with tarfile.open(LFW_TAR, "r") as tar:
            tar.extractall(lfw_dir, filter="data")
    else:
        print("[跳过] LFW 已解压")

    need   = LFW_QUOTA - existing
    copied = 0
    all_imgs = list((lfw_dir / "lfw_aug").rglob("*.jpg"))
    random.shuffle(all_imgs)

    for src in all_imgs:
        if copied >= need:
            break
        dst = TRAIN_POS / f"{existing + copied:05d}_{src.name}"
        if not dst.exists():
            shutil.copy2(src, dst)
            copied += 1
        if copied % 500 == 0 and copied > 0:
            print(f"  已复制 {copied}/{need} 张 LFW 正样本...", flush=True)

    print(f"[正样本-LFW] 共复制 {copied} 张，目录合计 {count_images(TRAIN_POS)} 张")
    return count_images(TRAIN_POS) >= LFW_QUOTA


# ══════════════════════════════════════════════════════════
# 步骤 1b：训练正样本 — 从 WIDER FACE val 裁人脸（备用）
# ══════════════════════════════════════════════════════════
def prepare_positives(imgs_root: Path, annotations: list) -> list[str]:
    """
    从 annotations 前半部分的图片里裁人脸，存入 TRAIN_POS。
    返回用于正样本的 img_rel 列表（不再用作测试集）。
    """
    existing = count_images(TRAIN_POS)
    if existing >= MAX_POSITIVES:
        print(f"[跳过] 训练正样本已有 {existing} 张")
        # 仍需返回被占用的图名，以便测试集排除
        half = len(annotations) // 2
        return [r for r, _ in annotations[:half]]

    # 用前半部分做正样本
    half     = len(annotations) // 2
    pos_anns = annotations[:half]

    random.seed(42)
    random.shuffle(pos_anns)

    need   = MAX_POSITIVES - existing
    saved  = 0

    for img_rel, boxes in pos_anns:
        if saved >= need:
            break
        src = imgs_root / img_rel
        if not src.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        for box in boxes:
            if saved >= need:
                break
            x, y, w, h = box
            # 只用清晰大脸（原图宽高均 >= 60px），过滤小脸/侧脸/遮挡脸
            if w < 60 or h < 60:
                continue
            # 稍微扩大一点，模拟真实 crop
            pad = int(min(w, h) * 0.1)
            x1  = max(0, x - pad)
            y1  = max(0, y - pad)
            x2  = min(img_w, x + w + pad)
            y2  = min(img_h, y + h + pad)
            crop = img[y1:y2, x1:x2]
            if crop.size == 0:
                continue
            fname = f"{saved:05d}_{src.stem}_x{x}y{y}.jpg"
            cv2.imencode('.jpg', crop)[1].tofile(str(TRAIN_POS / fname))
            saved += 1

        if saved % 200 == 0 and saved > 0:
            print(f"  已裁剪 {saved}/{need} 张人脸...", flush=True)

    print(f"[正样本] 裁剪完成，共 {count_images(TRAIN_POS)} 张")
    return [r for r, _ in pos_anns]


# ══════════════════════════════════════════════════════════
# 步骤 2：测试集 — WIDER FACE val 后半部分
# ══════════════════════════════════════════════════════════
def prepare_test_set(imgs_root: Path, annotations: list, used_for_train: set) -> None:
    """从未用于训练的 WIDER FACE 图片中构建测试集和标注 JSON。"""
    existing = count_images(TEST_IMGS)
    ann_count = 0
    if TEST_ANN.exists():
        try:
            ann_count = len(json.loads(TEST_ANN.read_text(encoding="utf-8")))
        except Exception:
            pass

    if existing >= MAX_TEST_IMAGES and ann_count >= MAX_TEST_IMAGES:
        print(f"[跳过] 测试集已有 {existing} 张 + {ann_count} 条标注")
        return

    test_annotations: dict[str, list] = {}
    copied = 0

    for img_rel, boxes in annotations:
        if img_rel in used_for_train:
            continue
        if copied >= MAX_TEST_IMAGES:
            break
        src = imgs_root / img_rel
        if not src.exists():
            continue
        fname = src.name
        dst   = TEST_IMGS / fname
        if not dst.exists():
            shutil.copy2(src, dst)
        test_annotations[fname] = boxes
        copied += 1

    TEST_ANN.write_text(
        json.dumps(test_annotations, indent=2, ensure_ascii=False),
        encoding="utf-8"
    )
    print(f"[测试集] {copied} 张图片，标注写入 {TEST_ANN.name}")


# ══════════════════════════════════════════════════════════
# 步骤 3：负样本 — 从 WIDER FACE val 图片避开人脸区域裁背景
# ══════════════════════════════════════════════════════════
def prepare_negatives(imgs_root: Path, annotations: list) -> None:
    """
    从 WIDER FACE val 图片中，避开所有人脸 bbox，随机裁出背景 patch。
    完全对应设计文档中"从 WIDER FACE 图片中避开人脸区域裁背景"方案。
    """
    existing = count_images(TRAIN_NEG)
    if existing >= TARGET_NEGATIVES:
        print(f"[跳过] 训练负样本已有 {existing} 张")
        return

    need = TARGET_NEGATIVES - existing
    saved = 0
    rng = random.Random(42)
    PATCH_SIZE = 64     # 裁出的背景 patch 尺寸
    TRIES_PER_IMG = 20  # 每张图最多尝试次数

    for img_rel, boxes in annotations:
        if saved >= need:
            break
        src = imgs_root / img_rel
        if not src.exists():
            continue
        img = cv2.imdecode(np.fromfile(str(src), dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]
        if img_h < PATCH_SIZE or img_w < PATCH_SIZE:
            continue

        for _ in range(TRIES_PER_IMG):
            if saved >= need:
                break
            # 随机选一个左上角
            px = rng.randint(0, img_w - PATCH_SIZE)
            py = rng.randint(0, img_h - PATCH_SIZE)
            px2, py2 = px + PATCH_SIZE, py + PATCH_SIZE

            # 检查是否与任何人脸框重叠（IoU 风格：只要相交就跳过）
            overlap = False
            for bx, by, bw, bh in boxes:
                ix1 = max(px, bx)
                iy1 = max(py, by)
                ix2 = min(px2, bx + bw)
                iy2 = min(py2, by + bh)
                if ix2 > ix1 and iy2 > iy1:
                    overlap = True
                    break
            if overlap:
                continue

            patch = img[py:py2, px:px2]
            if patch.size == 0:
                continue
            fname = f"{saved:05d}_wider_bg_{src.stem}_x{px}y{py}.jpg"
            cv2.imencode('.jpg', patch)[1].tofile(str(TRAIN_NEG / fname))
            saved += 1

        if saved % 50 == 0 and saved > 0:
            print(f"  已裁背景 {saved}/{need} 张...", flush=True)

    print(f"[负样本] 从 WIDER FACE 裁背景完成，共 {count_images(TRAIN_NEG)} 张")


# ══════════════════════════════════════════════════════════
# 主流程
# ══════════════════════════════════════════════════════════
def main() -> None:
    """执行下载、裁剪正负样本、生成测试集的完整数据准备流程。"""
    print("=" * 62)
    print("  Viola-Jones 数据准备脚本")
    print("=" * 62)

    # ── 确保 WIDER FACE val 已下载解压 ──
    print("\n[准备] 下载/检查 WIDER FACE val 图片...")
    imgs_root = ensure_wider_val()
    if imgs_root is None:
        print("\n[FAIL] WIDER FACE val 下载失败，无法继续。")
        return

    # ── 读取标注 ──
    print("[准备] 读取 WIDER FACE 标注...")
    annotations = load_wider_annotations()
    if not annotations:
        print("[FAIL] 未读到标注，请检查 .data_tmp/wider_face_split/ 目录。")
        return
    print(f"  共 {len(annotations)} 张带标注图片")

    # ── 步骤 1：正样本（LFW 前 1500 + WIDER FACE 大脸后 1500）──
    used_for_train = set()
    if LFW_TAR.exists():
        print(f"\n[步骤 1/3] 训练正样本 — LFW {LFW_QUOTA} 张 + WIDER FACE 大脸 {WF_QUOTA} 张")
        prepare_positives_lfw()
    else:
        print(f"\n[步骤 1/3] 训练正样本 — 仅从 WIDER FACE 裁人脸（未找到 LFW tar）")

    # 无论有无 LFW，都补充 WIDER FACE 大脸直到 MAX_POSITIVES
    if count_images(TRAIN_POS) < MAX_POSITIVES:
        print(f"  [补充] 继续从 WIDER FACE 裁大脸（目标 {MAX_POSITIVES} 张）...")
        used_for_train_list = prepare_positives(imgs_root, annotations)
        used_for_train = set(used_for_train_list)

    # ── 步骤 2：测试集 ──
    print("\n[步骤 2/3] 测试集 — WIDER FACE val 另一子集")
    prepare_test_set(imgs_root, annotations, used_for_train)

    # ── 步骤 3：负样本 ──
    print("\n[步骤 3/3] 训练负样本 — 从 WIDER FACE 避开人脸区域裁背景")
    prepare_negatives(imgs_root, annotations)

    # ── 汇总 ──
    n_pos  = count_images(TRAIN_POS)
    n_neg  = count_images(TRAIN_NEG)
    n_test = count_images(TEST_IMGS)
    n_ann  = 0
    if TEST_ANN.exists():
        try:
            n_ann = len(json.loads(TEST_ANN.read_text(encoding="utf-8")))
        except Exception:
            pass

    print("\n" + "=" * 62)
    print("  数据准备完成！")
    print(f"  训练正样本  {n_pos:>5} 张  ->  data/train/positives/")
    print(f"  训练负样本  {n_neg:>5} 张  ->  data/train/negatives/")
    print(f"  测试图片    {n_test:>5} 张  ->  data/test/images/")
    print(f"  测试标注    {n_ann:>5} 张  ->  data/test/annotations.json")
    print("=" * 62)

    if n_pos >= 100 and n_neg >= 100:
        print("\n[OK] 数据就绪！可以开始训练：\n")
        print("  # 快速验证（几分钟，先确认流程没问题）")
        print("  python train.py --max-features 500 --stage-sizes 2,4 --negative-samples 200\n")
        print("  # 正式训练")
        print("  python train.py\n")
        print("  # 训练完后测评：")
        print("  python evaluate.py --detector opencv")
        print("  python evaluate.py --detector custom --model models/custom_cascade.json")
    else:
        print("\n[!] 数据不完整，请检查网络后重新运行。")

    print(f"\n提示：临时缓存在 {TMP}，完成后可删除节省磁盘。")


if __name__ == "__main__":
    main()
