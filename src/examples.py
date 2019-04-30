#!/usr/bin/env python3
import argparse
import inspect
import sys

import numpy as np

import functions as f
import network as n
import utils as u
from logger import Logger
from train_utils import Train
from layers import *
from optimizers import *

def fcl01(epoch, batch_size, *_):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        FullyConnectedLayer(100, init_func=f.glorot_uniform, act_func=f.sigmoid),
        FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.sigmoid)
    ], f.quadratic)
    # optimizer = o.SGD_Momentum(3.0,0.9)
    # optimizer = o.NAG(3.0,0.9)
    # optimizer = o.ADAM()
    optimizer = ADAM_MAX()
    # optimizer = o.SGD(3.0)
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size

def fcl02(epoch,batch_size,*_):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.log_likelihood)
    optimizer = SGD(0.1)
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size


def cnn01(epoch,batch_size,kernel_size,pool_size):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        ConvolutionalLayer(2, kernel_size=kernel_size, init_func=f.glorot_uniform, act_func=f.sigmoid),
        MaxPoolingLayer(pool_size=pool_size),
        FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.log_likelihood)
    # optimizer = o.SGD(0.1)
    optimizer = ADAM_MAX()
    # optimizer = o.ADAM()
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size

def cnn02(epoch,batch_size,kernel_size,pool_size):
    net = n.NeuralNetwork([
        InputLayer(height=28, width=28),
        ConvolutionalLayer(2, kernel_size=kernel_size, init_func=f.glorot_uniform, act_func=f.sigmoid),
        MaxPoolingLayer(pool_size=pool_size),
        FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.categorical_crossentropy)
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
    args = parser.parse_args()
    Logger()
    log = Logger.get_logger(__name__)
    train = Train()
    np.random.seed(314)

    u.print("Loading '%s'..." % args.data, bcolor=u.bcolors.BOLD)
    trn_set, tst_set = u.load_mnist_npz(args.data)

    trn_set, vld_set = (trn_set[0][:50000], trn_set[1][:50000]), (trn_set[0][50000:], trn_set[1][50000:])

    u.print("Loading '%s'..." % args.func, bcolor=u.bcolors.BOLD)
    net, optimizer, num_epochs, batch_size = locals()[args.func](args.epoch,args.batch_size, args.kernel_size,args.pool_size)
    # u.print(inspect.getsource(locals()[args.func]).strip())

    u.print("Training network...", bcolor=u.bcolors.BOLD)
    train.train(net, optimizer, num_epochs, batch_size, trn_set, tst_set, vld_set)
