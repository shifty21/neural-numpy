import abc

import numpy as np
import math
from logger import logger

class Optimizer(metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    @abc.abstractmethod
    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        raise AssertionError


class SGD(Optimizer):
    def __init__(self, lr):
        log = logger.get_logger()
        log.info('Using SGD')
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
        log = logger.get_logger()
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

#Nesterov accelerated gradient
class NAG(Optimizer):
    def __init__(self, lr, m):
        log = logger.get_logger()
        log.info('Using NAG')
        self.lr = lr
        self.m = m

    def apply(self, layers, sum_der_w, sum_der_b, batch_len):
        for _, layer in layers:
            # lr = learning rate  and gw,gb is gradient
            # V(t)=γV(t−1)+η∇J( θ−γV(t−1) ) - θ−γV(t−1)  gives us an approximation of the next position of the parameters which gives us a rough idea where our parameters are going to be.
            # θ=θ−V(t) - update the weights and biases
            gw = sum_der_w[layer]/batch_len
            layer.vw = self.m*layer.vw + self.lr*gw*(layer.w - self.m*layer.vw)
            layer.w += -(layer.vw)
            gb = sum_der_b[layer]/batch_len
            layer.vb = self.m*layer.vb + self.lr*gb*(layer.b - self.m*layer.vb)
            layer.b += -(layer.vb)

#Adaptive Moment Estimation
class ADAM(Optimizer):
    def __init__(self, alpha = 0.002, beta1=0.9, beta2=0.999, epsilon=1e-8):
        log = logger.get_logger()
        log.info('Using ADAM')
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

class ADAM_MAX(Optimizer):
    def __init__(self,beta1 =0.9, beta2 = 0.999, alpha = 0.002):
        log = logger.get_logger()
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
