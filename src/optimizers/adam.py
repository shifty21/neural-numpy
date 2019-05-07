import abc

import numpy as np
import math
from logger import Logger
from optimizers.interface_optimizers import Optimizer

#Adaptive Moment Estimation
class ADAM(Optimizer):
    def __init__(self, alpha = 0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        log = Logger.get_logger(__name__)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.alpha = alpha
    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            layer.t = layer.t + 1
            #according to tensorflow api this is whats used whereas in algo alpha (fixed and lower accuracy) is used in place of lr(giving better accuracy)
            self.lr = self.alpha*math.sqrt((1-math.pow(self.beta2,layer.t))/(1-math.pow(self.beta1,layer.t)))
            with np.errstate(over='ignore'):
                try:
                    gw = sum_der_w[layer]/batch_len
                    layer.mtw = layer.mtw*self.beta1 + (1-self.beta1)*gw
                    layer.vtw = layer.vtw*self.beta2 + (1-self.beta2)*np.square(gw)
                    mtw = layer.mtw/(1-math.pow(self.beta1,layer.t))
                    vtw = layer.vtw/(1-math.pow(self.beta2,layer.t))
                    layer.w += -((mtw*self.lr/(np.sqrt(vtw) + self.epsilon)))
                except Exception as e:
                    print (e)
                try:
                    gb = sum_der_b[layer]/batch_len
                    layer.mtb = layer.mtb*self.beta1 + (1-self.beta1)*gb
                    layer.vtb = layer.vtb*self.beta2 + (1-self.beta2)*np.square(gb)
                    mtb = layer.mtb/(1-math.pow(self.beta1,layer.t))
                    vtb = layer.vtb/(1-math.pow(self.beta2,layer.t))
                    layer.b += -((mtb*self.lr/(np.sqrt(vtb) + self.epsilon)))
                    # print ("value of t :- " + str(layer.t) + " value of layer.mtb :-" + str(layer.mtb[0][0]) + " value of layer.vtb :- " + str(layer.vtb[0][0]) + " layer.w :- " + str(layer.w[0][0]))
                except Exception as e:
                    print (e)
