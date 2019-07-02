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

from utils import FixedPoint

def fcl01(weights_biases, fixed, convert):
    dropout = False
    net,_,_,_ = e.fcl01(None,1,10)
    print (net)
    temp = 1
    if convert == 1 :
        convert_to_fixed = True
    else:
        convert_to_fixed = False
    for layer,next_layer in net.layers:
        if temp==1:
            temp = temp +1
            continue
        else :
            if convert_to_fixed:
                 # print ("shape of values at start in weight matrix value",(weights_biases["w"][0].shape), layer.w[0][0])
                 # print ("shape of values at start in biase matrix value",(weights_biases["b"][0].shape), layer.b[0][0])
                 layer.w      = fixed.convert_float_to_fixed(weights_biases["w"][0])
                 layer.b      = fixed.convert_float_to_fixed(weights_biases["b"][0])
                 next_layer.w = fixed.convert_float_to_fixed(weights_biases["w"][1])
                 next_layer.b = fixed.convert_float_to_fixed(weights_biases["b"][1])
            else:
                layer.w      = weights_biases["w"][0]
                layer.b      = weights_biases["b"][0]
                next_layer.w = weights_biases["w"][1]
                next_layer.b = weights_biases["b"][1]
    return net

################################################################################
if __name__== "__main__":
    Logger()
    fixed = FixedPoint()
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--data", help="the path to the MNIST data set in .npz format (generated using utils.py)")
    parser.add_argument("-f", "--func", help="the function name of the test to be run")
    parser.add_argument("-c", "--convert", help="convert the weight and biases", type = int)
    args = parser.parse_args()

    trn_set, tst_set = u.load_mnist_npz(args.data)
    weights_biases = np.load("np_weights.npz")
    debug = False
    if debug:
        print ("weight1========== " + str(weights_biases["w"][0]))
        print ("weight2========== " + str(weights_biases["w"][1]))
        print ("biases1========== " + str(weights_biases["b"][0]))
        print ("biases2========== " + str(weights_biases["b"][1]))
    net = locals()[args.func](weights_biases, fixed, args.convert)

    print("Testing network...")
    train = Train()
    accuracy = train.test_inference(net, tst_set)
    print("Test accuracy:", 100*accuracy)
