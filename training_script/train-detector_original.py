import sys
import numpy as np
import cv2
import argparse
import tensorflow.keras as keras
import matplotlib.pyplot as plt

from random import choice
from os.path import isfile, isdir, basename, splitext
from os import makedirs

from src.keras_utils import save_model, load_model
from src.label import readShapes
from src.loss import loss
from src.utils import image_files_from_folder, show
from src.sampler import augment_sample, labels2output_map
from src.data_generator import DataGenerator

from pdb import set_trace as pause


def load_network(modelpath, input_dim):
    '''
    Hàm này load model và test luôn model xem nó có chạy đúng ko (có đúng là input shape > output shape 16 lần ko)
    '''
    model = load_model(modelpath)  # load eccv model theo mac dinh
    input_shape = (input_dim, input_dim, 3)  # cái này sẽ là (208,208,3)

    # Fixed input size for training
    inputs = keras.layers.Input(shape=(input_dim, input_dim, 3))
    outputs = model(inputs)

    output_shape = tuple([s for s in outputs.shape[1:]])
    output_dim = output_shape[1]
    model_stride = input_dim / output_dim

    assert input_dim % output_dim == 0, \
        'The output resolution must be divisible by the input resolution'

    assert model_stride == 2**4, \
        'Make sure your model generates a feature map with resolution ' \
        '16x smaller than the input'

    return model, model_stride, input_shape, output_shape


def process_data_item(data_item, dim, model_stride):
    # XX là ảnh đã augment
    # pts là 4 tọa độ của 4 đỉnh bounnding box đã chỉnh sửa theo ảnh augment. pts trong range(0,1)
    # llp là đối tượng Label được dựng lên từ pts thôi. thực ra label chính là pts mà
    # data_item[1].pts là dưới dạng tương đối
    XX, llp, pts = augment_sample(data_item[0], data_item[1].pts, dim) ## XX là (208,208,3)
    YY = labels2output_map(data_item[1].lp_type, llp, pts, dim, model_stride) ## YY là (13,13,9)  #### fix
    return XX, YY  # đây chính là cái mình dùng để train đây này.


def load_Data(data_dir):
    Files = image_files_from_folder(data_dir)  # Files là 1 list các img path

    Data = []
    for file in Files:
        # labfile là label file. File label của 1 ảnh có tên giống nó chỉ khác đuôi là txt
        labfile = splitext(file)[0] + '.txt'
        if isfile(labfile):
            L = readShapes(labfile)
            I = cv2.imread(file)
            Data.append([I, L[0]])  # I là ảnh, L[0] là label cho 4 góc của LP
            # L[0] thuộc class Shape()

    return Data


def process_Data(data, xshape=(208, 208, 3), yshape=(13, 13, 9)):
    X = np.empty((len(data),) + xshape)
    Y = np.empty((len(data),) + yshape)

    for (i, data_item) in enumerate(data):
        X[i], Y[i] = process_data_item(data_item, 208, 16)

    return X, Y


def plot_losses(train_loss, val_loss):
    plt.plot(train_loss)
    plt.plot(val_loss)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('iterations on batch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.show()


if __name__ == '__main__':
    # phần này phục vụ việc truyền các tham số từ dòng lệnh
    parser = argparse.ArgumentParser()
    parser.add_argument('-m' 		, '--model'			, type=str,
                        required=True		, help='Path to previous model')
    parser.add_argument('-n' 		, '--name'			, type=str,
                        required=True		, help='Model name')
    parser.add_argument('-tr'		, '--train-dir'		, type=str,
                        required=True		, help='Input data directory for training')
    parser.add_argument('-its'		, '--iterations'		, type=int, default=300000	,
                        help='Number of mini-batch iterations (default = 300.000)')
    parser.add_argument('-bs'		, '--batch-size'		, type=int,
                        default=32		, help='Mini-batch size (default = 32)')
    parser.add_argument('-od'		, '--output-dir'		, type=str,
                        default='./'		, help='Output directory (default = ./)')
    parser.add_argument('-op'		, '--optimizer'		, type=str,
                        default='Adam'	, help='Optmizer (default = Adam)')
    parser.add_argument('-lr'		, '--learning-rate'	, type=float,
                        default=.01		, help='Optmizer (default = 0.01)')
    args = parser.parse_args()

    # lấy giá trị các tham số từ command line
    netname = basename(args.name)
    train_dir = args.train_dir
    outdir = args.output_dir

    iterations = args.iterations
    batch_size = args.batch_size
    dim = 208  # input_shape là (208,208,3)

    if not isdir(outdir):
        makedirs(outdir)

    # file create_model.py tạo ra model rồi lưu vào 1 file
    # hàm này load lại model đó vào đây
    model, model_stride, xshape, yshape = load_network(args.model, dim) # xshape là (208,208,3), yshape là (13,13,8)
    opt = getattr(keras.optimizers, args.optimizer)(lr=args.learning_rate)
    model.compile(loss=loss, optimizer=opt)  # compile model

    print('Checking input directory...')
    Data = load_Data(train_dir)
    print('%d images with labels found' % len(Data))

    dg = DataGenerator(	data=Data,
                        process_data_item_func=lambda x: process_data_item(
                            x, dim, model_stride),
                        xshape=xshape,  # (208,208,3)
                        yshape=(yshape[0], yshape[1],
                                yshape[2]+1),  # (13,13,9)
                        nthreads=2,
                        pool_size=1000,
                        min_nsamples=100)
    dg.start()

    Xtrain = np.empty((batch_size, dim, dim, 3),
                      dtype='single')  # (batch,208,208,3)
    Ytrain = np.empty((batch_size, int(dim/model_stride),
                       int(dim/model_stride), 2*4+2))  ###### fix # (batch,13,13,10)

    val_dir = './samples/val'
    Val_Data = load_Data(val_dir)
    Xval, Yval = process_Data(Val_Data)  #### phải label cho cả tập val nữa dcm

    model_path_backup = '%s/%s_backup' % (outdir, netname)
    model_path_final = '%s/%s_final' % (outdir, netname)

    train_losses = []
    val_losses = []

    trainloss_txt = open('train_losses.txt', 'w')
    valloss_txt = open('val_losses.txt', 'w')

    for it in range(iterations):
        print('Iter. %d (of %d)' % (it+1, iterations))

        Xtrain, Ytrain = dg.get_batch(batch_size)
        train_loss = model.train_on_batch(Xtrain, Ytrain)
        val_loss = model.test_on_batch(
            Xval, Yval, reset_metrics=True, return_dict=False)

        print('\tLoss: %f' % train_loss)
        print('\tVal Loss: %f' % val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        trainloss_txt.write(str(train_loss)+'\n')
        valloss_txt.write(str(val_loss)+'\n')

        # Save model every 1000 iterations
        if (it+1) % 1000 == 0:
            print('Saving model (%s)' % model_path_backup)
            save_model(model, model_path_backup)

    print('Stopping data generator')
    dg.stop()

    trainloss_txt.close()
    valloss_txt.close()

    print('Saving model (%s)' % model_path_final)
    save_model(model, model_path_final)

    plot_losses(train_losses, val_losses)
