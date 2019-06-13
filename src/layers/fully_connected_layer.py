import numpy as np

from utils import functions as f
from utils import utils as u
from logger import Logger
import os
from utils import FixedPoint
from layers.interface_layer import Layer

class FullyConnectedLayer(Layer):

    def __init__(self, height, init_func, act_func, dropout=False):
        super().__init__()
        self.log = Logger.get_logger(__name__)
        self.depth = 1
        self.height = height
        self.width = 1
        self.n_out = self.depth * self.height * self.width
        self.init_func = init_func
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)
        self.dropout = dropout
        self.dropout_rate = 0.9
        self.fixedConvert = FixedPoint()
    def get_weights(self):
        return self.w

    def get_biases(self):
        return self.b

    def connect_to(self, prev_layer):
        self.w = self.init_func((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.b = f.zero_custom((self.n_out, 1))
        self.vw = f.zeros_like(self.w)
        self.vb = f.zeros_like(self.b)

        self.m0w = f.zeros_like(self.w)
        self.m0b = f.zeros_like(self.b)
        self.mtw = f.zeros_like(self.w)
        self.vtw = f.zeros_like(self.w)
        self.mtb = f.zeros_like(self.b)
        self.vtb = f.zeros_like(self.b)

    def feedforward(self, prev_layer):

        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network
        """

        prev_a = prev_layer.a.reshape((prev_layer.a.size, 1))

        if (self.dropout):
            dropout_matrix = np.random.rand(prev_a.shape[0], prev_a.shape[1]) < self.dropout_rate
            prev_a = np.multiply(prev_a, dropout_matrix)
            prev_a = prev_a / self.dropout_rate

        self.z = (self.w @ prev_a) + self.b
        # print ("shape of values in weight matrix ",(self.w[0][0]))
        # print ("shape of values in baise matrix ",(self.b[0][0]))
        self.a = self.fixedConvert.convert_fixed_to_float(self.act_func(self.z))
        # self.a = self.act_func(self.z)
        assert self.z.shape == self.a.shape

    def backpropagate(self, prev_layer, delta):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network
        :param delta: the error propagated backward by the next layer of the network
        :returns: the amount of change of input weights of this layer, the amount of change of the biases of this layer
            and the error propagated by this layer
        """
        assert delta.shape == self.z.shape == self.a.shape

        prev_a = prev_layer.a.reshape((prev_layer.a.size, 1))

        if (self.dropout):
            dropout_matrix = np.random.rand(prev_a.shape[0], prev_a.shape[1]) < self.dropout_rate
            prev_a = np.multiply(prev_a, dropout_matrix)
            prev_a = prev_a / self.dropout_rate
        der_w = delta @ prev_a.T

        der_b = np.copy(delta)

        prev_delta = (self.w.T @ delta).reshape(prev_layer.z.shape) * prev_layer.der_act_func(prev_layer.z)

        return der_w, der_b, prev_delta
