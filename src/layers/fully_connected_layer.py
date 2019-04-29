import numpy as np

import functions as f
import utils as u
from logger import Logger

from layers.interface_layer import Layer

class FullyConnectedLayer(Layer):

    def __init__(self, height, init_func, act_func):
        super().__init__()
        self.log = Logger.get_logger(__name__)
        self.log.info('FullyConnectedLayer init')
        self.depth = 1
        self.height = height
        self.width = 1
        self.n_out = self.depth * self.height * self.width
        self.init_func = init_func
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)

    def connect_to(self, prev_layer):
        self.w = self.init_func((self.n_out, prev_layer.n_out), prev_layer.n_out, self.n_out)
        self.b = f.zero((self.n_out, 1))
        self.vw = np.zeros_like(self.w)
        self.vb = np.zeros_like(self.b)

        self.m0w = np.zeros_like(self.w)
        self.m0b = np.zeros_like(self.b)
        self.mtw = np.zeros_like(self.w)
        self.vtw = np.zeros_like(self.w)
        self.mtb = np.zeros_like(self.b)
        self.vtb = np.zeros_like(self.b)

    def feedforward(self, prev_layer):

        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network
        """
        prev_a = prev_layer.a.reshape((prev_layer.a.size, 1))

        self.z = (self.w @ prev_a) + self.b

        self.a = self.act_func(self.z)
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

        der_w = delta @ prev_a.T

        der_b = np.copy(delta)

        prev_delta = (self.w.T @ delta).reshape(prev_layer.z.shape) * prev_layer.der_act_func(prev_layer.z)

        return der_w, der_b, prev_delta
