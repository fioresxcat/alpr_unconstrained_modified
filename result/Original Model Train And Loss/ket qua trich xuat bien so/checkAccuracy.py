import os
from pathlib import Path
from PIL import Image

test_dir = Path(r'C:\Users\Fiores\Desktop\Original Model Train And Loss\ket qua trich xuat bien so\test_resized_result')
val_dir = Path(r'C:\Users\Fiores\Desktop\Original Model Train And Loss\ket qua trich xuat bien so\val_resized_result')
# (470,110) (280,200)

long = 0
long_error = 0
square = 0
square_error = 0
correct_pred = 0
num_img = 0

for img_path in val_dir.glob('*.png'):
    num_img += 1
    img = Image.open(img_path)

    if 'vuong' in img_path.name:
        square += 1
        if img.size!=(280,200):
            square_error += 1
    else:
        long += 1
        if img.size!=(470,110):
            long_error += 1

correct_pred = num_img - long_error - square_error
accu = correct_pred / num_img

print(f'Tong so anh: {num_img}')
print(f'So bien dai: {long}')
print(f'So bien dai du doan dung: {long-long_error}')
print(f'So bien vuong: {square}')
print(f'So bien vuong du doan dung: {square-square_error}')
print(f'Tong so bien du doan dung: {correct_pred}')
print(f'Accuracy: {accu}')

# val_count = 0
# val_img_count = 0
# for img_path in val_dir.glob('*.png'):
#     val_img_count += 1
#     img = Image.open(img_path)
#     if img.size == (470,110) and 'vuong' not in str(img_path.name):
#         val_count += 1
#     if img.size == (280,200) and 'vuong' in str(img_path.name):
#         val_count += 1

# val_accu = val_count / val_img_count

# print(f'test_count / test_img : {test_count}/{test_img_count}')
# print(f'test accuracy: {test_accu}')
# print(f'val_count / val_img : {val_count}/{val_img_count}')
# print(f'val accuracy: {val_accu}')
