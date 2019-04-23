#!/usr/bin/env python3
import argparse
import inspect
import sys

import numpy as np

import functions as f
import layers as l
import optimizers as o
import network as n
import utils as u
from logger import logger

def fcl01(epoch, batch_size, *_):
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.FullyConnectedLayer(100, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.sigmoid)
    ], f.quadratic)
    # optimizer = o.SGD_Momentum(3.0,0.9)
    # optimizer = o.NAG(3.0,0.9)
    # optimizer = o.ADAM()
    optimizer = o.ADAM_MAX()
    # optimizer = o.SGD(3.0)
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size

def fcl02(epoch,batch_size,*_):
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.FullyConnectedLayer(10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.log_likelihood)
    optimizer = o.SGD(0.1)
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size


def cnn01(epoch,batch_size,kernel_size,pool_size):
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.ConvolutionalLayer(2, kernel_size=kernel_size, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.MaxPoolingLayer(pool_size=pool_size),
        l.FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.log_likelihood)
    # optimizer = o.SGD(0.1)
    optimizer = o.ADAM_MAX()
    # optimizer = o.ADAM()
    num_epochs = epoch
    batch_size = batch_size
    return net, optimizer, num_epochs, batch_size

def cnn02(epoch,batch_size,kernel_size,pool_size):
    net = n.NeuralNetwork([
        l.InputLayer(height=28, width=28),
        l.ConvolutionalLayer(2, kernel_size=kernel_size, init_func=f.glorot_uniform, act_func=f.sigmoid),
        l.MaxPoolingLayer(pool_size=pool_size),
        l.FullyConnectedLayer(height=10, init_func=f.glorot_uniform, act_func=f.softmax)
    ], f.categorical_crossentropy)
    # optimizer = o.SGD(0.1)
    optimizer = o.ADAM()
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
    parser.add_argument("-l","--log_level", help="Logging level - DEBUG, INFO, WARNING, ERROR, CRITICAL")
    args = parser.parse_args()

    log_obj = logger(args.log_level.upper())
    log = logger.get_logger()

    np.random.seed(314)

    u.print("Loading '%s'..." % args.data, bcolor=u.bcolors.BOLD)
    trn_set, tst_set = u.load_mnist_npz(args.data)

    trn_set, vld_set = (trn_set[0][:50000], trn_set[1][:50000]), (trn_set[0][50000:], trn_set[1][50000:])

    u.print("Loading '%s'..." % args.func, bcolor=u.bcolors.BOLD)
    net, optimizer, num_epochs, batch_size = locals()[args.func](args.epoch,args.batch_size, args.kernel_size,args.pool_size)
    # u.print(inspect.getsource(locals()[args.func]).strip())

    u.print("Training network...", bcolor=u.bcolors.BOLD)
    n.train(net, optimizer, num_epochs, batch_size, trn_set, vld_set)

    u.print("Testing network...", bcolor=u.bcolors.BOLD)
    accuracy = n.test(net, tst_set)
    u.print("Test accuracy: %0.2f%%" % (accuracy*100))
