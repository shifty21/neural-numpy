#!/usr/bin/env python3
import argparse
import inspect
import sys
import os
import numpy as np

from utils import functions as f
from utils import utils as u
import network as n
from logger import Logger
from train_utils import Train
from layers import *
from optimizers import *
from utils.functions import CustomDataType

def fcl01(regularization, epoch, batch_size,dropout=False, **kwargs):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        FullyConnectedLayer(100, init_func=f.glorot_uniform_initializer, act_func=f.sigmoid, dropout=dropout),
        FullyConnectedLayer(10, init_func=f.glorot_uniform_initializer, act_func=f.sigmoid, dropout=dropout)
    ], f.quadratic, regularization)
    # optimizer = o.SGD_Momentum(3.0,0.9)
    # optimizer = NAG(3.0,0.9)
    # optimizer = ADAM()
    # optimizer = ADAM_MAX()
    optimizer = SGD(3.0)
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size

def fcl02(regularization, epoch,batch_size,*_):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.softmax, dropout=dropout)
    ], f.log_likelihood, regularization)
    optimizer = SGD(0.1)
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size


def cnn01(regularization, epoch,batch_size,kernel_size,pool_size ):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        ConvolutionalLayer(2, kernel_size=kernel_size, init_func=f.glorot_uniform_initializer, act_func=f.sigmoid, dropout=dropout),
        MaxPoolingLayer(pool_size=pool_size, dropout=dropout),
        FullyConnectedLayer(height=10, init_func=f.glorot_uniform_initializer, act_func=f.softmax, dropout=dropout)
    ], f.log_likelihood, regularization)
    # optimizer = SGD(0.1)
    optimizer = ADAM()
    # optimizer = o.ADAM()
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size

def cnn02(regularization, epoch,batch_size,kernel_size,pool_size):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        ConvolutionalLayer(2, kernel_size=kernel_size, init_func=f.glorot_uniform, act_func=f.sigmoid, dropout=dropout),
        MaxPoolingLayer(pool_size=pool_size, dropout=dropout),
        FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax, dropout=dropout)
    ], f.categorical_crossentropy, regularization)
    # optimizer = o.SGD(0.1)
    optimizer = ADAM()
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d","--data", help="The path to the MNIST data set in .npz format (generated using utils.py)")
    parser.add_argument("-f","--func", help="The function name of the example to be run")
    parser.add_argument("-e","--epoch", help="No of passes over the data", type = int)
    parser.add_argument("-b","--batch_size", help="Size of each batch", type = int)
    parser.add_argument("-ks","--kernel_size", help="Size of each batch", type = int)
    parser.add_argument("-p","--pool_size", help="MaxPoolingLayer pool size", type = int)
    parser.add_argument("-r","--regularization", help="Regularization algo. Possible values L1, L2", type = str)
    parser.add_argument("-do","--dropout", help="Enable dropout", type = str)
    parser.add_argument("-dt","--data_type", help="Float type 32 or 64", type = int)

    args = parser.parse_args()
    Logger()
    train = Train()
    np.random.seed(314)
    dropout = args.dropout
    log = Logger.get_logger(__name__)
    f.InitCustomDataType(args.data_type)
    u.print("Loading '%s'..." % args.data, bcolor=u.bcolors.BOLD)
    trn_set, tst_set = u.load_mnist_npz(args.data)

    trn_set, vld_set = (trn_set[0][:50000], trn_set[1][:50000]), (trn_set[0][50000:], trn_set[1][50000:])

    u.print("Loading '%s'..." % args.func, bcolor=u.bcolors.BOLD)
    log.debug("args.func-- " + str(args.func) + " args.regularization-- " + str(args.regularization) + " args.epoch-- " + str(args.epoch)
                 + " args.batch_size-- " + str(args.batch_size) + " args.kernel_size-- " + str(args.kernel_size ) + " args.pool_size-- " + str(args.pool_size))
    net, optimizer, num_epochs, batch_size = locals()[args.func](args.regularization, args.epoch,args.batch_size, args.kernel_size,args.pool_size)
    log.debug(inspect.getsource(locals()[args.func]).strip())

    u.print("Training network...", bcolor=u.bcolors.BOLD)
    train.train(net, optimizer, num_epochs, batch_size, trn_set, tst_set, vld_set)
