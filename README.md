# Viola-Jones 实时人脸检测系统

本项目用于计算机视觉大作业“目标检测系统”。系统包含两套非深度学习人脸检测器：

- `opencv`：调用 OpenCV 已经训练好的 Haar Cascade，作为基线方法。
- `custom`：本项目手写实现的教学版 Viola-Jones 级联分类器。

自定义检测器实现了核心算法链路：积分图、Haar-like 矩形特征、单特征弱分类器、AdaBoost 强分类器、简化 Cascade、多尺度滑窗、IoU/NMS 和检测指标评估。OpenCV 只用于图片/视频读取、缩放、窗口显示和画框。

## 1. 环境配置

建议使用 Python 3.10 或以上版本。可以任选一种方式创建环境，只要最终能安装 `requirements.txt` 中的依赖即可。

方式一：使用 `venv`：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

方式二：使用 conda：

```bash
conda create -n face-detection python=3.10
conda activate face-detection
pip install -r requirements.txt
```

如果你已经有可用的 Python 环境，也可以直接在该环境中安装依赖：

```bash
pip install -r requirements.txt
```

## 2. 实时演示

使用 OpenCV 预训练分类器：

```bash
python main.py --detector opencv
```

使用自己训练的分类器：

```bash
python main.py --detector custom --model models/custom_cascade.json
```

指定图片或视频输入：

```bash
python main.py --detector opencv --source data/test/images/example.jpg
python main.py --detector custom --source data/test/images/example.jpg
python main.py --detector opencv --source path/to/video.mp4
```

只保存结果、不弹出窗口：

```bash
python main.py --detector custom --source data/test/images/example.jpg --no-display
```

交互控制窗口说明：

- `scale x100`：图像金字塔缩放系数乘以 100，例如 110 表示 `1.10`。
- `minNeighbors`：OpenCV Haar 检测器的候选框过滤参数。
- `minSize`：最小人脸尺寸。
- `window step`：自定义检测器的滑窗步长，越小越细但越慢。
- `equalize`：是否开启直方图均衡。
- `eyes`：是否在人脸框内继续检测眼睛。
- `privacy blur`：是否对检测到的人脸区域做模糊处理。

快捷键：

- `s`：保存当前检测结果到 `results/screenshots/`。
- `f`：保存检测到的人脸裁剪到 `results/faces/`。
- `q` 或 `Esc`：退出程序。

## 3. 训练自定义 Cascade

准备训练数据：

```text
data/train/positives/   # 人脸裁剪图，每张图视为一个正样本
data/train/negatives/   # 非人脸图片，程序会从中随机裁剪负样本
```

正式训练命令：

```bash
python train.py \
  --positive-dir data/train/positives \
  --negative-dir data/train/negatives \
  --output models/custom_cascade.json
```

默认训练设置：

- 训练窗口：`24x24`
- Haar 特征候选数：`8000`
- Cascade 阶段：`10,20,40`，即 3 个 AdaBoost 强分类器，共 70 个 Haar 弱分类器
- 负样本 patch 数：`2500`
- 每一级训练后进行 hard negative mining

如果先验证流程，可以降低训练规模：

```bash
python train.py --max-features 500 --stage-sizes 2,4 --negative-samples 200 --active-negatives 100
```

训练完成后，模型保存到 `models/custom_cascade.json`，每一级最强的 Haar 特征可视化图保存到 `results/features/`。

## 4. 测评

准备测试集：

```text
data/test/images/
data/test/annotations.json
```

标注格式：

```json
{
  "image1.jpg": [[x, y, w, h], [x, y, w, h]],
  "image2.jpg": [[x, y, w, h]]
}
```

运行测评：

```bash
python evaluate.py --detector opencv
python evaluate.py --detector custom --model models/custom_cascade.json
```

输出文件：

- `results/eval/summary.json`：Precision、Recall、F1、TP、FP、FN、平均耗时和 FPS。
- `results/eval/metrics.csv`：逐图测评结果。
- `results/eval/visualizations/`：真实框和预测框的可视化对比图。

## 5. 代码结构

```text
main.py                  # 实时演示入口
train.py                 # 自定义 Cascade 训练入口
evaluate.py              # 测评入口
src/integral_image.py    # 积分图与矩形求和
src/haar_features.py     # Haar-like 特征生成和计算
src/adaboost.py          # AdaBoost 弱分类器选择和强分类器训练
src/cascade.py           # 多级强分类器串联
src/sliding_window.py    # 多尺度滑窗检测
src/nms.py               # IoU 和非极大值抑制
src/metrics.py           # Precision / Recall / F1 计算
```

## 6. 大作业说明

本项目对应 Viola 和 Jones 的论文《Robust Real-Time Face Detection》的核心思想：

- 使用积分图快速计算矩形区域像素和。
- 使用 Haar-like 矩形特征描述人脸局部结构。
- 使用 AdaBoost 从大量候选特征中选择有效弱分类器。
- 使用 Cascade 结构快速拒绝大量背景窗口。

当前实现是课程设计规模的简化版本，目标是完整展示算法训练、检测和评估闭环；它不追求论文原版工业级检测率，也不使用 CNN、YOLO、Transformer 或其他深度学习模型。
