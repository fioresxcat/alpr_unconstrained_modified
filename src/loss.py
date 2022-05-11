
import tensorflow as tf


def logloss(Ptrue, Pred, szs, eps=10e-10):
    b, h, w, ch = szs   # batch, height, width, channel
    Pred = tf.clip_by_value(Pred, eps, 1.)
    Pred = -tf.math.log(Pred)
    Pred = Pred*Ptrue
    Pred = tf.reshape(Pred, (b, h*w*ch))
    Pred = tf.reduce_sum(Pred, 1)
    return Pred  # shape (b,1)


def l1(true, pred, szs):
    b, h, w, ch = szs
    res = tf.reshape(true-pred, (b, h*w*ch))
    res = tf.abs(res)
    res = tf.reduce_sum(res, 1)
    return res


def loss(Ytrue, Ypred):
    # Ytrue: M x N x 10
    # Ypred: M x N x 10

    b = tf.shape(Ytrue)[0]
    h = tf.shape(Ytrue)[1]
    w = tf.shape(Ytrue)[2]

    # shape (b,h,w). Mặt này toàn là 1,0,1,0,... thôi. Vì nó xác định ở mỗi pixel có biển số hay ko mà
    obj_probs_true = Ytrue[..., 0]
    obj_probs_pred = Ypred[..., 0]

    non_obj_probs_true = 1. - Ytrue[..., 0]
    non_obj_probs_pred = Ypred[..., 1]

    lp_long_true_probs = Ytrue[..., 1]  # fix
    lp_square_true_probs = 1. - lp_long_true_probs  # fix

    lp_long_pred_probs = Ypred[..., 8]  # fix
    lp_square_pred_probs = Ypred[..., 9]  # fix

    # cac he so cua affine transformation ma model tinh ra
    affine_pred = Ypred[..., 2:8]  # fix
    # vi tri cac pixel la dinh cua boundingbox. pts_true chính là A_mn rồi.
    pts_true = Ytrue[..., 2:]

    # từ affine_pred, ta tính ra pts là matrix có dạng giống với Ytrue, bao gồm các tọa độ của 4 đỉnh cho mỗi điểm (m,n)
    # sau đó, so sánh pts với pts_true để ra hàm loss.
    # điều đặc biệt ỏ đây, là Ytrue và Ypred 'ko giống dạng nhau'. Từ Ypred, ta mới đi tính toán ra pts. Cái pts này mới là giống dạng Ytrue này.
    # Ypred là ma trận các hệ số affine, còn Ytrue lại là ma trận các tọa độ đỉnh của bb
    affinex = tf.stack([tf.maximum(affine_pred[..., 0], 0.),
                        affine_pred[..., 1], affine_pred[..., 2]], 3)  # shape (b,h,w,3)
    affiney = tf.stack([affine_pred[..., 3], tf.maximum(
        affine_pred[..., 4], 0.), affine_pred[..., 5]], 3)

    v = 0.5
    # shape (1,1,1,12)
    base = tf.stack([[[[-v, -v, 1., v, -v, 1., v, v, 1., -v, v, 1.]]]])
    base = tf.tile(base, tf.stack([b, h, w, 1]))  # shape (b,h,w,12)

    pts = tf.zeros((b, h, w, 0))  # empty tensor

    for i in range(0, 12, 3):
        row = base[..., i:(i+3)]  # shape (b,h,w,3)
        ptsx = tf.reduce_sum(affinex*row, 3)  # ptsx co shape (b,h,w)
        ptsy = tf.reduce_sum(affiney*row, 3)  # ptsy co shape (b,h,w)

        pts_xy = tf.stack([ptsx, ptsy], 3)  # pts_xy co shape (b,h,w,2)
        pts = (tf.concat([pts, pts_xy], 3))
    # ket thuc vong for thi pts co shape (b,h,w,8)

    flags = tf.reshape(obj_probs_true, (b, h, w, 1))
    # *flags là để xem ô nào có LP thì nó mới thêm vào hàm loss.
    res = 1.*l1(pts_true*flags, pts*flags, (b, h, w, 4*2))
    res += 1.*logloss(obj_probs_true, obj_probs_pred, (b, h, w, 1))
    res += 1.*logloss(non_obj_probs_true, non_obj_probs_pred, (b, h, w, 1))
    res += 0.3*logloss(lp_long_true_probs*flags,
                       lp_long_pred_probs*flags, (b, h, w, 1))  # fix
    res += 0.3*logloss(lp_square_true_probs*flags,
                       lp_square_pred_probs*flags, (b, h, w, 1))  # fix

    return res
