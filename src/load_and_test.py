#!/usr/bin/env python3
import argparse

import numpy as np

from utils import functions as f
from utils import utils as u
import network as n
from logger import Logger
from train_utils import Train
from layers import *
from optimizers import *
import examples as e



def fcl01(weights):
    dropout = False
    net,_,_,_ = e.fcl01(None,1,10)
    print (net)
    return net

################################################################################
if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="the path to the MNIST data set in .npz format (generated using utils.py)")
    parser.add_argument("-f", "--func", help="the function name of the test to be run")
    args = parser.parse_args()

    trn_set, tst_set = u.load_mnist_npz(args.data)
    weights_biases = np.load("np_weights.npz")
    print ("weight1========== " + str(weights_biases["w"][0].shape))
    print ("weight2========== " + str(weights_biases["w"][1].shape))
    print ("biases1========== " + str(weights_biases["b"][0].shape))
    print ("biases2========== " + str(weights_biases["b"][1].shape))
    net = locals()[args.func](weights_biases)

    print("Testing network...")
    train = Train()
    accuracy = train.test(net, tst_set)
    print("Test accuracy:", accuracy)
