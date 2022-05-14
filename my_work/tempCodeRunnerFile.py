import shutil
import os
from pathlib import Path

csv_file = open(
    r"F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\alpr-unconstrained-master\samples\test\cd-hard-annotations.csv", 'r')
img_names = []
for line in csv_file:
    img_name = line.split(',')[0]
    source = os.path.join(
        r"F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\Original datasets\cars_train", f"{img_name}.jpg")
    des = os.path.join(
        r"F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\alpr-unconstrained-master\samples\test", f"{img_name}.jpg")
    shutil.copyfile(source, des)
    print(f'Done {img_name}')