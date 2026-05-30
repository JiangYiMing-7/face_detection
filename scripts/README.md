# scripts/ — 辅助调试脚本

本目录存放开发与调参过程中使用的**辅助脚本**，不属于核心交付流程（训练 / 评估 / 演示）。
核心入口仍在仓库根目录：`train.py`、`train_v6.py`、`evaluate.py`、`demo.py`、`prepare_data.py`、`prepare_caltech.py`、`audit_cascade.py`。

这些脚本依赖 `src/` 包，请**从仓库根目录**运行，并把根目录加入 `PYTHONPATH`：

```bash
PYTHONPATH=. python scripts/<脚本名>.py
```

| 脚本 | 用途 |
|------|------|
| `debug_lfw.py` | 在单张/少量 LFW 图片上可视化调试检测结果 |
| `debug_caltech.py` | 在 Caltech 图片上调试检测结果 |
| `quick_lfw_test.py` / `quick_lfw_test2.py` | 快速跑 LFW 子集，查看 P/R/F1 |
| `caltech_score_test.py` | 在 Caltech 上扫 `score_threshold` 参数 |
| `tune_threshold.py` | 调参辅助：扫描检测阈值 |
| `check_model.py` | 打印模型 JSON 的级联结构（级数、各级弱分类器数） |
| `inspect_mat.py` | 检查特征矩阵 / 中间张量的数值 |
