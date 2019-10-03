from __future__ import print_function
import os
import struct
import sys

import numpy as np
from logger import Logger
from PIL import Image
import pandas as pd
from sklearn.model_selection import train_test_split

# from keras.utils import to_categorical


def build_mnist_npz(mnist_dirpath):
    Logger()
    log = Logger.get_logger(__name__)
    log.info('Building mnist npz')
    training_set_images = os.path.join(mnist_dirpath,
                                       "train-images-idx3-ubyte")
    training_set_labels = os.path.join(mnist_dirpath,
                                       "train-labels-idx1-ubyte")
    test_set_images = os.path.join(mnist_dirpath, "t10k-images-idx3-ubyte")
    test_set_labels = os.path.join(mnist_dirpath, "t10k-labels-idx1-ubyte")

    with open(training_set_images, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        trn_imgs = np.fromfile(f, dtype=np.uint8).reshape(60000, 1, 28, 28)
    with open(training_set_labels, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        trn_lbls = np.fromfile(f, dtype=np.uint8)

    with open(test_set_images, "rb") as f:
        magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
        tst_imgs = np.fromfile(f, dtype=np.uint8).reshape(10000, 1, 28, 28)
    with open(test_set_labels, "rb") as f:
        magic, num = struct.unpack(">II", f.read(8))
        tst_lbls = np.fromfile(f, dtype=np.uint8)

    with open("vgg16.npz", "wb") as f:
        np.savez_compressed(
            f,
            trn_imgs=trn_imgs,
            trn_lbls=trn_lbls,
            tst_imgs=tst_imgs,
            tst_lbls=tst_lbls)


def transpose_images(image_set):
    return np.asarray([im.transpose(-1, 0, 1) for im in image_set])


def load_mnist_fashion_npz(path):
    log = Logger.get_logger(__name__)
    log.info('loading fashion npz')
    maximum = 60000
    train_data = pd.read_csv(path + '/fashion-mnist_train.csv')
    test_data = pd.read_csv(path + '/fashion-mnist_test.csv')
    log.info("train_data shape %s", train_data.shape)  #(60,000*785)
    log.info("test_data shape %s", test_data.shape)  #(10000,785)
    train_X = np.array(train_data.iloc[:, 1:])[:maximum]
    test_X = np.array(test_data.iloc[:, 1:])
    train_Y = np.array(train_data.iloc[:, 0])[:maximum]  # (60000,)
    test_Y = np.array(test_data.iloc[:, 0])  #(10000,)
    log.info("train_x shape %s test_x shape %s", train_X.shape, test_X.shape)
    # Convert the images into 3 channels
    train_X = np.dstack([train_X] * 3)
    test_X = np.dstack([test_X] * 3)
    log.info("train_x shape %s test_x shape %s", train_X.shape, test_X.shape)
    # Reshape images as per the tensor format required by tensorflow
    train_X = train_X.reshape(-1, 28, 28, 3)
    test_X = test_X.reshape(-1, 28, 28, 3)

    log.info("train_x shape %s test_x shape %s", train_X.shape, test_X.shape)
    train_X = resize_images(train_X)
    test_X = resize_images(test_X)
    train_X = transpose_images(train_X)
    test_X = transpose_images(test_X)
    log.info("train_x shape %s test_x shape %s", train_X.shape, test_X.shape)
    # Normalise the data and change data type
    train_X = train_X / 255.
    test_X = test_X / 255.
    train_X = train_X.astype('float32')
    test_X = test_X.astype('float32')

    train_Y_one_hot = get_categorical_transform(train_Y)
    test_Y_one_hot = get_categorical_transform(test_Y)
    # Splitting train data as train and validation data
    train_X, valid_X, train_label, valid_label = train_test_split(
        train_X, train_Y_one_hot, test_size=0.2, random_state=13)

    log.info(
        "train_x shape %s valid_x shape %s train_label shape %s valid_label shape %s",
        train_X.shape, valid_X.shape, train_label.shape, valid_label.shape)
    return (train_X, train_label), (valid_X, valid_label)


def resize_images(train_set):
    # Resize the images 48*48 as required by VGG16
    return np.asarray([
        np.array(Image.fromarray(im, mode="RGB").resize((48, 48)))
        for im in train_set
    ])


def get_categorical_transform(train_set):
    # Converting Labels to one hot encoded format
    def to_categorical(lbl):
        if lbl not in labels_to_categorical:
            y = np.zeros((10, 1), dtype=np.uint8)
            y[lbl] = 1
            labels_to_categorical[lbl] = y
        return labels_to_categorical[lbl]

    labels_to_categorical = dict()

    return np.asarray([to_categorical(lbl) for lbl in train_set],
                      dtype=np.int8)


def load_mnist_npz(mnist_npzpath):
    log = Logger.get_logger(__name__)
    log.info('loading mnist npz')

    def to_categorical(lbl):
        if lbl not in labels_to_categorical:
            y = np.zeros((10, 1), dtype=np.uint8)
            y[lbl] = 1
            labels_to_categorical[lbl] = y
        return labels_to_categorical[lbl]

    labels_to_categorical = dict()

    dataset = np.load(mnist_npzpath)

    trn_imgs = dataset["trn_imgs"]
    trn_lbls = dataset["trn_lbls"]
    # log.info("trn_images %s", trn_imgs.shape)
    # log.info("trn_lbls %s", trn_lbls.shape)
    trn_x = np.array([img / 255. for img in trn_imgs])
    trn_y = np.array([to_categorical(lbl) for lbl in trn_lbls]).astype(
        np.uint8)

    tst_imgs = dataset["tst_imgs"]
    tst_lbls = dataset["tst_lbls"]
    tst_x = np.array([img / 255. for img in tst_imgs]).astype(np.float32)
    tst_y = np.array([to_categorical(lbl) for lbl in tst_lbls]).astype(
        np.uint8)
    # log.info("shape of trn_x %s", trn_x[0][0])
    return (trn_x, trn_y), (tst_x, tst_y)


class bcolors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def bar(now, end):
    return "[%-10s]" % ("=" * int(10 * now / end))


def print(s="", bcolor=None, override=False):
    log = Logger.get_logger(__name__)
    log.info
    if print.last_override and not override:
        sys.stdout.write("\n")
    print.last_override = override

    if override:
        sys.stdout.write("\33[2K\r")
    if bcolor:
        sys.stdout.write(bcolor)
    sys.stdout.write(str(s))
    if bcolor:
        sys.stdout.write(bcolors.ENDC)
    if not override and s:
        sys.stdout.write("\n")
    sys.stdout.flush()


print.last_override = False
