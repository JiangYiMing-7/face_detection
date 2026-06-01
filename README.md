# Viola-Jones 实时人脸检测系统

计算机视觉课程大作业：从零实现 Viola-Jones 人脸检测流程，并与 OpenCV 预训练 Haar Cascade 进行对比。项目不使用深度学习框架，核心算法由 Python + NumPy 实现。

## 当前可复现结果

仓库已包含 LFW 测试子集、标注文件和训练好的 JSON 级联模型。使用 `models/custom_cascade_v6_full_1.json` 在 `data/test_lfw/` 上评估时，当前复现结果如下：

| 检测器 | 测试集 | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: |
| OpenCV Haar Cascade | LFW 200 张 | 1.0000 | 0.9600 | 0.9796 |
| 自实现 v6.1 | LFW 200 张 | 1.0000 | 1.0000 | 1.0000 |

自实现 v6.1 常规操作点使用 `min_size=100`、`min_neighbors=5`。严格操作点可把 `min_size` 调到 60，用于观察更多候选窗口下的误检抑制能力。

## 环境配置

建议使用 Python 3.10+。如果本机已有课程环境，例如 conda 的 `ml` 环境，可以先激活：

```bash
conda activate ml
```

也可以新建环境后安装依赖：

```bash
pip install -r requirements.txt
```

主要依赖：

- `numpy`：核心矩阵计算
- `opencv-python`：图像读写、摄像头、OpenCV 基线
- `matplotlib`：训练曲线和报告图表脚本
- `scipy`：Caltech `.mat` 标注整理脚本

## 最短复现

在项目根目录执行：

```bash
python -m unittest discover -s tests
python evaluate.py --detector custom --output-dir results/eval_reproduce
```

`evaluate.py` 的默认复现配置为：

- 模型：`models/custom_cascade_v6_full_1.json`
- 图片：`data/test_lfw/images`
- 标注：`data/test_lfw/annotations.json`
- 输出：`results/eval`

如果当前 shell 的 `python` 没有安装 OpenCV，请先进入包含 `opencv-python` 的环境再运行。

## 项目结构

```text
face_detection_baseline/
├── src/                         # 核心算法与检测器封装
│   ├── integral_image.py         # 积分图、平方积分图
│   ├── haar_features.py          # Haar-like 特征定义、枚举、可视化
│   ├── adaboost.py               # AdaBoost 弱/强分类器训练
│   ├── cascade.py                # 级联分类器与 JSON 序列化
│   ├── sliding_window.py         # 多尺度滑窗检测与聚类合并
│   ├── nms.py                    # IoU、NMS、包含框过滤
│   ├── detectors.py              # OpenCV / Custom 统一检测接口
│   ├── metrics.py                # Precision、Recall、F1 评估
│   └── annotations.py            # 标注 JSON 读取
├── train.py                      # v1-v3 风格训练入口
├── train_v6.py                   # v6/v6.1 训练入口，支持 x7 增强
├── evaluate.py                   # 批量评估入口
├── demo.py                       # 双屏实时摄像头演示
├── main.py                       # 单检测器交互式演示入口
├── prepare_data.py               # WIDER FACE/LFW 数据准备
├── prepare_caltech.py            # Caltech-101 Faces 数据整理
├── audit_cascade.py              # 级联模型审计脚本
├── scripts/                      # 辅助调参、调试、检查脚本
├── tests/                        # 核心数学模块冒烟测试
├── models/                       # 已训练 JSON 模型
├── data/
│   ├── test_lfw/                 # 已包含的 LFW 复现测试集
│   └── train/                    # 训练数据目录，positive 需自行准备
├── results/                      # 本地评估输出和运行结果
├── 训练数据与可视化/               # 训练日志、图表、模型备份
└── 报告.md                       # 课程结题报告
```

## 定量评估

复现自实现 v6.1：

```bash
python evaluate.py --detector custom \
  --model models/custom_cascade_v6_full_1.json \
  --image-dir data/test_lfw/images \
  --annotations data/test_lfw/annotations.json \
  --min-neighbors 5 \
  --min-size 100 \
  --output-dir results/eval_lfw_v61
```

复现 OpenCV 基线：

```bash
python evaluate.py --detector opencv \
  --image-dir data/test_lfw/images \
  --annotations data/test_lfw/annotations.json \
  --output-dir results/eval_lfw_opencv
```

严格操作点：

```bash
python evaluate.py --detector custom \
  --model models/custom_cascade_v6_full_1.json \
  --image-dir data/test_lfw/images \
  --annotations data/test_lfw/annotations.json \
  --min-neighbors 5 \
  --min-size 60 \
  --output-dir results/eval_lfw_strict_v61
```

评估输出包括：

- `metrics.csv`：逐图 TP/FP/FN、Precision、Recall、F1、耗时
- `summary.json`：整体汇总指标
- `visualizations/`：预测框和真实框可视化

## 实时演示

双屏摄像头对比，左侧 OpenCV，右侧自实现模型：

```bash
python demo.py --model models/custom_cascade_v6_full_1.json
```

常用调参示例：

```bash
python demo.py \
  --model models/custom_cascade_v3_hnm.json \
  --scale-factor 1.2 \
  --min-size 40 \
  --window-step 6 \
  --min-neighbors-custom 3 \
  --score-threshold 5.0
```

快捷键：

- `q`：退出
- 空格：暂停/继续
- `r`：开始/停止录制，默认保存到 `results/demo_output.mp4`

## 训练模型

仓库包含负样本和已训练模型，但正样本目录默认只保留占位文件。完整训练前需要先准备正样本。

```bash
python prepare_data.py
```

如果已有本地 LFW 增强包，不要把绝对路径写进代码，可通过环境变量传入：

```bash
LFW_TAR=/path/to/lfw_aug.tar python prepare_data.py
```

训练 v3 风格模型：

```bash
python train.py \
  --positive-dir data/train/positives \
  --negative-dir data/train/negatives \
  --output models/my_cascade.json \
  --max-features 20000 \
  --max-positives 1000 \
  --stage-sizes 5,10,15,20,30,40,50,60 \
  --augment
```

训练 v6/v6.1 风格模型：

```bash
python train_v6.py \
  --positive-dir data/train/positives \
  --negative-dir data/train/negatives \
  --output models/my_cascade_v61.json \
  --max-features 20000 \
  --max-positives 3000 \
  --stage-sizes 10,15,20,30,40,50,60,80,100,120 \
  --augment
```

训练会输出模型到 `models/`，日志到 `results/logs/`，代表性 Haar 特征图到 `results/features/`。

## Caltech 测试集整理

如果需要复现 Caltech-101 Faces 测试，可先准备 Caltech-101 数据，并通过参数或环境变量指定路径：

```bash
CALTECH_ROOT=/path/to/caltech-101 python prepare_caltech.py
```

整理后可评估：

```bash
python evaluate.py --detector custom \
  --model models/custom_cascade_v1_no_hnm.json \
  --image-dir data/test/caltech/images \
  --annotations data/test/caltech/annotations.json \
  --output-dir results/eval_caltech_custom \
  --min-size 30 \
  --window-step 4 \
  --scale-factor 1.2
```

## 路径与可移植性

项目入口脚本已按项目根目录解析默认相对路径，避免依赖本机绝对路径。提交或打包时需要保留这些目录：

- `src/`
- `models/`
- `data/test_lfw/`
- `tests/`
- `requirements.txt`
- `README.md`
- `报告.md`

本地数据集路径通过命令行参数或环境变量传入，不应写死到代码中。

## 辅助脚本

`scripts/` 下是实验和调参用脚本：

- `check_model.py`：查看模型 stage 数量、弱分类器数量和阈值
- `quick_lfw_test.py` / `quick_lfw_test2.py`：快速扫描 LFW 后处理参数
- `tune_threshold.py`：扫描 `score_threshold`
- `debug_lfw.py` / `debug_caltech.py`：单图诊断 IoU 和检测框
- `inspect_mat.py`：查看 Caltech `.mat` 标注内容

这些脚本会通过 `scripts/project_paths.py` 从项目根目录解析数据和模型路径。

## 核心实现特点

- 积分图实现 Haar 特征 O(1) 矩形求和
- AdaBoost 训练弱分类器并组合成强分类器
- 多级 Cascade 快速拒绝非人脸窗口
- Hard Negative Mining 提高背景抑制能力
- `_cluster_weighted_merge` 合并跨尺度候选框
- `TrackingBoxSmoother` 提升实时演示中多人脸框稳定性
- 可选 CLAHE 预处理用于对比实验

## 参考文献

Viola, P., & Jones, M. (2001). *Rapid Object Detection using a Boosted Cascade of Simple Features*. CVPR.
