import abc

import numpy as np
import math
from logger import Logger
from optimizers.interface_optimizers import Optimizer

#Nesterov accelerated gradient
class NAG(Optimizer):
    def __init__(self, lr, m):
        log = Logger.get_logger(__name__)
        self.lr = lr
        self.m = m

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            # lr = learning rate  and gw,gb is gradient
            # V(t)=γV(t−1)+η∇J( θ−γV(t−1) ) - θ−γV(t−1)  gives us an approximation of the next position of the parameters which gives us a rough idea where our parameters are going to be.
            # θ=θ−V(t) - update the weights and biases
            gw = sum_der_w[layer]/batch_len
            layer.vw = self.m*layer.vw - self.lr*gw*(layer.w + self.m*layer.vw)
            layer.w += layer.vw
            gb = sum_der_b[layer]/batch_len
            layer.vb = self.m*layer.vb - self.lr*gb*(layer.b + self.m*layer.vb)
            layer.b += layer.vb
