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
    weights = np.load("np_weights.npz")
    print ("weight========== " + str(weights["w"][0][0].shape))
    
    print ("biases========== " + str(weights["b"].shape))
    net = locals()[args.func](weights)

    print("Testing network...")
    train = Train()
    accuracy = train.test(net, tst_set)
    print("Test accuracy:", accuracy)
