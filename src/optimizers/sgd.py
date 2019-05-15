import abc

import numpy as np
import math
from logger import Logger
from optimizers.interface_optimizers import Optimizer

class SGD(Optimizer):
    def __init__(self, lr):
        log = Logger.get_logger(__name__)
        self.lr = lr

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            # lr = learning rate  and gw,gb is gradient
            gw = sum_der_w[layer]/batch_len
            layer.w += -(self.lr*gw)
            gb = sum_der_b[layer]/batch_len
            layer.b += -(self.lr*gb)
