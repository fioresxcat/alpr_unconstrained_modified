import cv2
import os
import matplotlib.pyplot as plt
from PIL import Image
from pathlib import Path

data_dir = Path(r'F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\alpr-unconstrained-master-final\samples\test')
des_dir = Path(r'F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\alpr-unconstrained-master-final\samples\resized_image\test_resized_600_500')

for img_path in data_dir.glob('vuong*.jpg'):
    img = Image.open(str(img_path))
    img = img.resize((600,500))
    img.save(des_dir / img_path.name)
