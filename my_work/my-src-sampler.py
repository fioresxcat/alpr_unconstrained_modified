import cv2
import numpy as np
import random

from src.utils import im2single, getWH, hsv_transform, IOU_centre_and_dims
from src.label import Label
from src.projection_utils import perspective_transform, find_T_matrix, getRectPts


def labels2output_map(lp_type, label, lppts, dim, stride):  ### fix
    '''
    label: là Label tương ứng với lppts
    lppts: là pts của ảnh gốc đã được rectified (và augmented), dưới dạng tương đối
    '''
    # 7.75 when dim = 208 and stride = 16
    side = ((float(dim) + 40.)/2.)/stride

    outsize = int(dim/stride)
    # 1 channel cho xác suất có object / ko object, 8 channel là 8 giá trị xác định 4 đỉnh của LP trong output feature map
    Y = np.zeros((outsize, outsize, 2*4+2), dtype='float32') ##### fix
    Y[..., 1] = lp_type  ###### fix

    MN = np.array([outsize, outsize])
    WH = np.array([dim, dim], dtype=float)

    # tl, br là vị trí trên ảnh gốc (vị trí tương đối)
    # => ta cũng chỉ xét những pixel ở vị trí tương đối như thế trên ảnh output
    tlx, tly = np.floor(np.maximum(label.tl(), 0.)*MN).astype(int).tolist()
    brx, bry = np.ceil(np.minimum(label.br(), 1.)*MN).astype(int).tolist()

    # duyệt hết những điểm có thể chứa LP. Xem điểm nào thực sự chứa thì gán Y[y,x,0] = 1
    for x in range(tlx, brx):
        for y in range(tly, bry):

            mn = np.array([float(x) + .5, float(y) + .5])
            iou = IOU_centre_and_dims(
                mn/MN, label.wh(), label.cc(), label.wh())  # label.cc(), label.wh() xác định vị trí tương đối của bb trong ảnh gốc
                # do đó, câu lệnh phía trên để tính xem vị trí tương đối của bb trong MN với tâm là mn và vị trí tương đối của bb trong ảnh gốc có giống nhau ko ?

            if iou > 0.5:

                p_WH = lppts*WH.reshape((2, 1))
                p_MN = p_WH/stride

                p_MN_center_mn = p_MN - mn.reshape((2, 1))

                p_side = p_MN_center_mn/side

                # nếu IoU giữa bb tương đối có tâm ở điểm (x,y) và Biển số thật > 0.5 thì xác suất để tại điểm (x,y) đó xuất hiện biển số là 1. Còn lại là 0 hết
                Y[y, x, 0] = 1.
                # 8 lớp phía dưới là 8 normalized annotated points of the LP. Chính là A_mn
                Y[y, x, 2:] = p_side.T.flatten()  ######### fix

    return Y


def pts2ptsh(pts):
    # pts la mot ma tran 2 chieu
    # them 1 dong toan so 1 o duoi matrix thoi
    return np.matrix(np.concatenate((pts, np.ones((1, pts.shape[1]))), 0))


def project(I, T, pts, dim):
    # 1. Riêng phần này là tính pts mới
    # thêm 1 dòng toàn 1 vào dưới pts thôi
    ptsh = np.matrix(np.concatenate((pts, np.ones((1, 4))), 0))

    # T.shape =  (3,3), ptsh.shape = (3,4) => new ptsh.shape = (3,4)
    ptsh = np.matmul(T, ptsh)

    ptsh = ptsh/ptsh[2]  # dòng cuối cùng của ptsh sau bước này lại toàn 1
    ptsret = ptsh[:2]  # lấy 2 dòng đầu là pts mới cho ảnh đã augment
    ptsret = ptsret/dim  # chuyển pts từ tuyệt đối sang tương đối

    # 2. Riêng phần này là tính ảnh mới đã augment
    Iroi = cv2.warpPerspective(
        I, T, (dim, dim), borderValue=.0, flags=cv2.INTER_LINEAR)

    # 3. Trả lại ảnh đã chiếu và pts mới tương ứng
    return Iroi, ptsret


def flip_image_and_pts(I, pts):
    I = cv2.flip(I, 1)  # flip I horizontally
    pts[0] = 1. - pts[0]
    idx = [1, 0, 3, 2]
    pts = pts[..., idx]
    return I, pts


def augment_sample(I, pts, dim):
    maxsum, maxangle = 120, np.array([80., 80., 45.])

    # np.random.rand(3) trả về 3 số trong khoảng (0,1)
    angles = np.random.rand(3)*maxangle

    if angles.sum() > maxsum:
        angles = (angles/angles.sum()) * (maxangle/maxangle.sum())

    I = im2single(I)  # normalize ảnh
    iwh = getWH(I.shape)  # get width and height

    whratio = random.uniform(2., 4.)
    wsiz = random.uniform(dim*.2, dim*1.)  # wsiz từ dim/5 đến dim
    hsiz = wsiz/whratio

    dx = random.uniform(0., dim - wsiz)
    dy = random.uniform(0., dim - hsiz)

    pph = getRectPts(dx, dy, dx+wsiz, dy+hsiz)
    # pts đưa vào là dưới dạng tương đối, ta đang biến nó về dạng tuyệt đối
    pts = pts*iwh.reshape((2, 1))
    # ma trận T biến vùng biển số về góc nhìn thẳng với 4 góc là pph
    T = find_T_matrix(pts2ptsh(pts), pph)

    # H cũng là một T_matrix
    H = perspective_transform((dim, dim), angles=angles)
    H = np.matmul(H, T)  # nhân 2 T_matrix lại với nhau. Vừa rectified, vừa xoay góc nhìn 3D

    # trả về ảnh đã crop chỉ còn vùng ROI và pts đc chuyển đổi tương ứng (pts là tương đối)
    Iroi, pts = project(I, H, pts, dim)

    # đoạn dưới này là mod màu sắc thôi
    hsv_mod = np.random.rand(3).astype('float32')
    hsv_mod = (hsv_mod - 0.5) * 0.3
    hsv_mod[0] *= 360
    Iroi = hsv_transform(Iroi, hsv_mod)
    Iroi = np.clip(Iroi, 0., 1.)

    pts = np.array(pts)

    # đoạn này là flip ảnh
    if random.random() > .5:
        Iroi, pts = flip_image_and_pts(Iroi, pts)

    # gán cái pts tìm được thành 1 cái Label
    tl, br = pts.min(1), pts.max(1)
    llp = Label(0, tl, br)

    return Iroi, llp, pts
