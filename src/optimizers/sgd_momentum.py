import abc

import numpy as np
import math
from logger import Logger
from optimizers.interface_optimizers import Optimizer

class SGD_Momentum(Optimizer):
    def __init__(self, lr, m):
        log = Logger.get_logger(__name__)
        log.info('Using SGD_Momentum')
        self.lr = lr
        self.m = m

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            # lr = learning rate  and gw,gb is gradient
            # V(t)=γV(t−1)+η∇J(θ).  γV(t−1) - gives us the momentum
            # θ=θ−V(t) - update the weights and biases
            gw = sum_der_w[layer]/batch_len
            layer.vw = self.m*layer.vw + gw
            layer.w -= self.lr*layer.vw

            gb = sum_der_b[layer]/batch_len
            layer.vb = self.m*layer.vb + gb
            layer.b -= self.lr*layer.vb
