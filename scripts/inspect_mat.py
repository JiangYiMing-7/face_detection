"""查看 Caltech .mat 标注文件中的原始键和值。"""

import argparse

import scipy.io
from project_paths import project_path


parser = argparse.ArgumentParser(description="查看 Caltech .mat 标注文件中的原始键和值。")
parser.add_argument(
    "mat_path",
    nargs="?",
    default="data/caltech-101/caltech-101/Annotations/Faces_2/annotation_0001.mat",
    help="待检查的 .mat 文件路径，默认按项目 data/ 目录解析。",
)
args = parser.parse_args()

mat = scipy.io.loadmat(str(project_path(args.mat_path)))
print('Keys:', list(mat.keys()))
for k, v in mat.items():
    if not k.startswith('_'):
        print(k, ':', v)
