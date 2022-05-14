import os
from pathlib import Path

# file = open(r"F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\alpr-unconstrained-master\alpr-unconstrained-master\samples\train-detector\00030.txt", "r")
# lines = file.readlines()
# file.close()

# file = file = open(r"F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\alpr-unconstrained-master\alpr-unconstrained-master\samples\train-detector\00030.txt", "w")
# lines[0] = lines[0][:-4] + ','
# file.write(lines[0])
# file.close()

excluded = ['00011.txt', '00024.txt', '00029.txt']
dir = Path(r'F:\Desktop\Sách\20211\Nhập môn AI\Đồ án\alpr-unconstrained-master\alpr-unconstrained-master\samples\train-detector')
for file in dir.glob('*.txt'):
    if file.name not in excluded:
        file_obj = open(file, 'r')
        lines = file_obj.readlines()
        file_obj.close()

        file_obj = open(file, 'w')
        lines[0] = lines[0][:-4] + ','
        file_obj.write(lines[0])
        file_obj.close()

        print('Done')
