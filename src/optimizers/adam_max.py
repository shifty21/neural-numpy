import abc

import numpy as np
import math
from logger import Logger
from optimizers.interface_optimizers import Optimizer

class ADAM_MAX(Optimizer):
    def __init__(self,beta1 =0.9, beta2 = 0.999, alpha = 0.002):
        log = Logger.get_logger(__name__)
        log.info('Using ADAM_MAX')
        self.beta1 = beta1
        self.beta2 = beta2
        self.alpha = alpha
        #according to algo this is the lr but its not used in parameter calculation instead alpha is used
        # self.lr = self.alpha/(1-math.pow(self.beta1,layer.t))
        #adamax says alpha/(1-beta1^t) is the lr (one below gives better accuracy of 88.75 where as one in algo(above) gives 61.92 %)
        self.lr = self.alpha/(1-self.beta1)
    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            with np.errstate(over='ignore'):
                layer.t = layer.t + 1

                gw = sum_der_w[layer]/batch_len
                gb = sum_der_b[layer]/batch_len
                layer.m0w = self.beta1*layer.m0w + (1-self.beta1)*gw
                layer.u0w = max(self.beta2*layer.u0w,np.linalg.norm(gw))
                #with self.lr on
                layer.w = layer.w - (self.lr/(1-math.pow(self.beta1,layer.t)))*layer.m0w/layer.u0w
                # print ("value of t :- " + str(layer.t) + " value of layer.m0w :-" + str(layer.m0w[0][0]) + " value of layer.u0w :- " + str(layer.u0w) + " layer.w :- " + str(layer.w[0][0]))
                layer.m0b = self.beta1*layer.m0b + (1-self.beta1)*gb
                layer.u0b = max(self.beta2*layer.u0b,np.linalg.norm(gb))
                layer.b = layer.b - (self.lr/(1-math.pow(self.beta1,layer.t)))*layer.m0b/layer.u0b
