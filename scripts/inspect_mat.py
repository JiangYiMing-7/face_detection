"""查看 Caltech .mat 标注文件中的原始键和值。"""

import scipy.io
mat = scipy.io.loadmat(r'D:\aaa大三下作业\计算机视觉\face_detection\data\caltech-101\caltech-101\Annotations\Faces_2\annotation_0001.mat')
print('Keys:', list(mat.keys()))
for k, v in mat.items():
    if not k.startswith('_'):
        print(k, ':', v)
