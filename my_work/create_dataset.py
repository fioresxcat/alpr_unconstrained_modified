from pathlib import Path
import os
import shutil

p = Path(r'F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\training-dataset-annotations\aolp-le')
list_file = []
for file in p.glob('*.txt'):
    list_file.append(file.stem)
# print(list_file)

source_dir = r"F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\AOLP\Subset_LE\Subset_LE\Subset_LE\Subset_LE\Image"
des_dir = r"F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\Datasets\image"

for file in list_file:
    
    source = os.path.join(source_dir, file + '.jpg')
    des = os.path.join(des_dir, file + '.jpg')
    shutil.copyfile(source, des)