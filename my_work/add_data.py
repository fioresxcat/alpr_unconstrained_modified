import shutil
import os
from pathlib import Path

# --------------------------------------------------rename bien dai-------------------------------------------
import shutil
import os
from pathlib import Path

data_dir = Path(r"C:\Users\Fiores\Desktop\30 dài")
count = 1
for file in data_dir.glob('*.jpg'):
    os.rename(file, os.path.join(data_dir,f'dai_{count}.jpg'))
    count += 1

# -------------------------------------------------rename bien vuong---------------------------------------------------
import shutil
import os
from pathlib import Path

data_dir = Path(r"C:\Users\Fiores\Desktop\130 vuông")
count = 1
for file in data_dir.glob('*.jpg'):
    os.rename(file, os.path.join(data_dir,f'vuong_{count}.jpg'))
    count += 1

# ------------------------------------------------------copy anh test-------------------------------------------
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

# ------------------------------------------------code luu result ra file----------------------------------------

# load model
model_path = 'path_to_model'
wpod_net = load_model(model_path)

# Kích thước lớn nhất và nhỏ nhất của 1 chiều ảnh
Dmax = 608
Dmin = 288

result_file = open('result.txt', 'w')
for img_path in test_dir.glob('*.jpg'):
    img_path = str(img_path)
    Ivehicle = cv2.imread(img_path)

    # Lấy tỷ lệ giữa W và H của ảnh và tìm ra chiều nhỏ nhất
    ratio = float(max(Ivehicle.shape[:2])) / min(Ivehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)

    _, LpImg, lp_type = detect_lp(wpod_net, im2single(
        Ivehicle), bound_dim, lp_threshold=0.5)

    # Cau hinh tham so cho model SVM
    digit_w = 30  # Kich thuoc ki tu
    digit_h = 60  # Kich thuoc ki tu
    model_svm = cv2.ml.SVM_load('svm.xml')

    if (len(LpImg)):
        # Chuyen doi anh bien so
        LpImg[0] = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))

        roi = LpImg[0]

        # Chuyen anh bien so ve gray
        gray = cv2.cvtColor(LpImg[0], cv2.COLOR_BGR2GRAY)

        # Ap dung threshold de phan tach so va nen
        binary = cv2.threshold(gray, 127, 255,
                               cv2.THRESH_BINARY_INV)[1]

        # cv2.imshow("Anh bien so sau threshold", binary)
        # cv2.waitKey()

        # Segment kí tự
        kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)
        cont, _ = cv2.findContours(
            thre_mor, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        plate_info = ""
        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1.5 <= ratio <= 3.5:  # Chon cac contour dam bao ve ratio w/h
                if h/roi.shape[0] >= 0.6:  # Chon cac contour cao tu 60% bien so tro len

                    # Ve khung chu nhat quanh so
                    cv2.rectangle(roi, (x, y), (x + w, y + h), (0, 255, 0), 2)

                    # Tach so va predict
                    curr_num = thre_mor[y:y+h, x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(
                        curr_num, 30, 255, cv2.THRESH_BINARY)
                    curr_num = np.array(curr_num, dtype=np.float32)
                    curr_num = curr_num.reshape(-1, digit_w * digit_h)

                    # Dua vao model SVM
                    result = model_svm.predict(curr_num)[1]
                    result = int(result[0, 0])

                    if result <= 9:  # Neu la so thi hien thi luon
                        result = str(result)
                    else:  # Neu la chu thi chuyen bang ASCII
                        result = chr(result)

                    plate_info += result

        # cv2.imshow("Cac contour tim duoc", roi)
        # cv2.waitKey()

        # Viet bien so len anh
        cv2.putText(Ivehicle, fine_tune(plate_info), (50, 50),
                    cv2.FONT_HERSHEY_PLAIN, 3.0, (0, 0, 255), lineType=cv2.LINE_AA)

        # Hien thi anh
        print(f'Image name: {img_path.name} - Bien so: {plate_info}')
        result_file.write(f'{img_name}, {plate_info}\n')
        # cv2.imshow("Hinh anh output",Ivehicle)
        # cv2.waitKey()

    # cv2.destroyAllWindows()
result_file.close()

# ------------------------------------------------- load file result va test accuracy --------------------------
prediction_file = open('result.txt', 'r')
groundtruth_file = open('cd-hard-annotations.csv')

predictions = {}
groundtruths = {}
num_prediction, num_groundtruth = 0, 0
for line in prediction_file:
    name, lp = line.split(',')
    predictions[name] = lp.lower()
    num_prediction += 1

for line in groundtruth_file:
    name, lp, _ = line.split(',')
    groundtruths[name] = lp.lower()
    num_groundtruth += 1

assert num_prediction == num_groundtruth

accu = 0
for name in predictions:
    if predictions[name] == groundtruths[name]:
        accu += 1
accuracy = accu / num_prediction
print(f'Accuracy: {accuracy}')
