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
import math
import time

from utils import FixedPoint
from utils import CustomMultiplier

import multiprocessing
from multiprocessing import Manager


def load_network(weights_biases, net, convert_to_fixed, log):
    fixed = FixedPoint()
    log.info(net)
    temp = -1
    last_layer = 0
    for layer, next_layer in net.layers:
        if temp == -1:
            temp = temp + 1
            continue
        elif isinstance(layer, MaxPoolingLayer):
            last_layer = next_layer
            continue
        else:
            if convert_to_fixed:
                log.debug(
                    "shape of values at start in weight matrix value - %s",
                    str(weights_biases["w"][temp].shape))
                log.debug(
                    "shape of values at start in biase matrix value - %s",
                    str(weights_biases["b"][temp].shape))
                layer.w = fixed.convert_float_to_fixed(
                    weights_biases["w"][temp])
                layer.b = fixed.convert_float_to_fixed(
                    weights_biases["b"][temp])
                # next_layer.w = fixed.convert_float_to_fixed(weights_biases["w"][temp])
                # next_layer.b = fixed.convert_float_to_fixed(weights_biases["b"][temp])
                last_layer = next_layer
            else:
                layer.w = weights_biases["w"][temp]
                layer.b = weights_biases["b"][temp]
                # next_layer.w = weights_biases["w"][temp]
                # next_layer.b = weights_biases["b"][temp]
                last_layer = next_layer
        temp += 1
    if convert_to_fixed:
        log.debug("value of temp - %s", temp)
        log.debug("shape of values at start in weight matrix value - %s",
                  str(weights_biases["w"][temp].shape))
        log.debug("shape of values at start in biase matrix value - %s",
                  str(weights_biases["b"][temp].shape))
        last_layer.w = fixed.convert_float_to_fixed(weights_biases["w"][temp])
        last_layer.b = fixed.convert_float_to_fixed(weights_biases["b"][temp])
    else:
        last_layer.w = weights_biases["w"][temp]
        last_layer.b = weights_biases["b"][temp]
    return net


def fcl01(weights_biases, convert_to_fixed, log):
    dropout = False
    net, _, _, _ = e.fcl010(None, 1, 10)
    return load_network(weights_biases, net, convert_to_fixed, log)

    ################################################################################


def cnn01(weight_biases, convert_to_fixed, log):
    dropout = False
    net, _, _, _ = e.cnn01(None, 1, 10, 5, 2, dropout)
    return load_network(weight_biases, net, convert_to_fixed, log)


def calculate_sigmoid_table():
    log = Logger.get_logger(__name__)
    start = int(round(time.time() * 1000))
    arr = {}
    for i in range(np.iinfo(np.int16).min, np.iinfo(np.int16).max):
        with np.errstate(over="ignore"):
            arr[i] = (1.0 / (1.0 + np.exp(-i)))
    log.info("length of dic generated normally %s", len(arr))
    log.info("Total time for sigmoid array %f", time.time() * 1000 - start)
    return arr


if __name__ == "__main__":
    l = Logger()
    log = Logger.get_logger(__name__)
    sigmoid_arr = calculate_sigmoid_table()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--data",
        help=
        "the path to the MNIST data set in .npz format (generated using utils.py)"
    )
    parser.add_argument(
        "-f", "--func", help="the function name of the test to be run")
    parser.add_argument(
        "-c", "--convert", help="convert the weight and biases", type=int)
    parser.add_argument("-sd", "--saved_data", help="Saved data")

    args = parser.parse_args()

    if args.convert == 1:
        convert_to_fixed = True
    else:
        convert_to_fixed = False
    trn_set, tst_set = u.load_mnist_npz(args.data)
    weights_biases = np.load(args.saved_data)
    log.debug("shape of weight %s", str(len(weights_biases["w"])))
    log.debug("shape of biases %s", str(len(weights_biases["w"])))
    net = locals()[args.func](weights_biases, convert_to_fixed, log)
    custom_multiplier = CustomMultiplier(sigmoid_arr)
    log.info("Testing network...")
    train = Train()
    accuracy = train.test_inference(net, tst_set, convert_to_fixed)
    log.info("Test accuracy: %f", 100 * accuracy)
