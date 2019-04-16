import abc

import numpy as np


class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        raise AssertionError


class SGD(Optimizer):
    def __init__(self, lr):
        self.lr = lr

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            # lr = learning rate  and gw,gb is gradient
            gw = sum_der_w[layer]/batch_len
            layer.w += -(self.lr*gw)
            gb = sum_der_b[layer]/batch_len
            layer.b += -(self.lr*gb)

class SGD_Momentum(Optimizer):
    def __init__(self, lr, m):
        self.lr = lr
        self.m = m

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            # lr = learning rate  and gw,gb is gradient
            import pdb; pdb.set_trace()
            gw = sum_der_w[layer]/batch_len
            layer.vw = self.m*layer.vw + gw
            layer.w += -(layer.vw)
            gb = sum_der_b[layer]/batch_len
            layer.vb = self.m*layer.vb + gb
            layer.b += -(layer.vb)
