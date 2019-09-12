import numpy as np

from utils import functions as f
from utils import utils as u
from logger import Logger

from layers.interface_layer import Layer
from layers.convolutional_layer import ConvolutionalLayer

from utils.fixed_point import FixedPoint


class MaxPoolingLayer(Layer):
    def __init__(self, pool_size, dropout=False):
        super().__init__()
        self.log = Logger.get_logger(__name__)
        self.log.info('MaxPoolingLayer init')
        self.pool_size = pool_size
        self.der_act_func = lambda x: x
        self.dropout = dropout
        self.dropout_rate = 0.9
        self.fixedConverter = FixedPoint()

    def get_weights(self, convert_to_float):
        if convert_to_float:
            return self.fixedConverter.convert_fixed_to_float(self.w)
        return self.w

    def get_biases(self, convert_to_float):
        if convert_to_float:
            return self.fixedConverter.convert_fixed_to_float(self.b)
        return self.b

    def connect_to(self, prev_layer):
        assert isinstance(prev_layer, ConvolutionalLayer)
        self.depth = prev_layer.depth
        self.height = (
            (prev_layer.height - self.pool_size) // self.pool_size) + 1
        self.width = (
            (prev_layer.width - self.pool_size) // self.pool_size) + 1
        self.n_out = self.depth * self.height * self.width

        self.w = np.empty((0))
        self.b = np.empty((0))
        self.vw = f.zeros_like(self.w)
        self.vb = f.zeros_like(self.b)

        self.m0w = f.zeros_like(self.w)
        self.m0b = f.zeros_like(self.b)
        self.mtw = f.zeros_like(self.w)
        self.vtw = f.zeros_like(self.w)
        self.mtb = f.zeros_like(self.b)
        self.vtb = f.zeros_like(self.b)

    def feedforward(self, prev_layer, inference):
        """
        Feedforward the observation through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer must be a list of
            feature maps, where each feature map is a 3d matrix
        """
        assert self.w.size == 0
        assert self.b.size == 0
        assert isinstance(prev_layer, ConvolutionalLayer)
        assert prev_layer.depth == self.depth
        assert prev_layer.a.ndim == 3

        prev_a = prev_layer.a
        if (self.dropout):
            dropout_matrix = np.random.rand(
                prev_a.shape[0], prev_a.shape[1]) < self.dropout_rate
            prev_a = np.multiply(prev_a, dropout_matrix)
            prev_a = prev_a / self.dropout_rate

        prev_layer_fmap_size = prev_layer.height
        assert prev_layer_fmap_size % self.pool_size == 0
        if inference:
            self.z = np.zeros((self.depth, self.height, self.width),
                              dtype=np.int8)
        else:
            self.z = np.zeros((self.depth, self.height, self.width))
        for r, t in zip(range(self.depth), range(prev_layer.depth)):
            assert r == t
            for i, m in enumerate(range(0, prev_layer.height, self.pool_size)):
                for j, n in enumerate(
                        range(0, prev_layer.width, self.pool_size)):
                    prev_a_window = prev_a[t, m:m + self.pool_size, n:n +
                                           self.pool_size]
                    assert prev_a_window.shape == (self.pool_size,
                                                   self.pool_size)
                    # downsampling
                    self.z[r, i, j] = np.max(prev_a_window)

        self.a = self.z

    def backpropagate(self, prev_layer, delta):
        """
        Backpropagate the error through the layer. Given any pair source(convolutional)/destination(pooling) feature
        maps, each unit of the destination feature map propagates an error to a window (self.pool_size, self.pool_size)
        of the source feature map

        :param prev_layer: the previous layer of the network
        :param delta: a tensor of shape (self.depth, self.height, self.width)
        """
        assert self.w.size == 0
        assert self.b.size == 0
        assert isinstance(prev_layer, ConvolutionalLayer)
        assert prev_layer.depth == self.depth
        assert prev_layer.a.ndim == 3
        assert delta.shape == (self.depth, self.height, self.width)

        prev_a = prev_layer.a

        if (self.dropout):
            dropout_matrix = np.random.rand(
                prev_a.shape[0], prev_a.shape[1]) < self.dropout_rate
            prev_a = np.multiply(prev_a, dropout_matrix)
            prev_a = prev_a / self.dropout_rate

        der_w = f.array([])

        der_b = f.array([])

        prev_delta = np.empty_like(prev_a)
        for r, t in zip(range(self.depth), range(prev_layer.depth)):
            assert r == t
            for i, m in enumerate(range(0, prev_layer.height, self.pool_size)):
                for j, n in enumerate(
                        range(0, prev_layer.width, self.pool_size)):
                    prev_a_window = prev_a[t, m:m + self.pool_size, n:n +
                                           self.pool_size]
                    assert prev_a_window.shape == (self.pool_size,
                                                   self.pool_size)
                    # upsampling: the unit which was the max at the forward propagation
                    # receives all the error at backward propagation (the other units receive zero)
                    max_unit_index = np.unravel_index(prev_a_window.argmax(),
                                                      prev_a_window.shape)
                    prev_delta_window = f.zeros_like(prev_a_window)
                    prev_delta_window[max_unit_index] = delta[t, i, j]
                    prev_delta[r, m:m + self.pool_size, n:n +
                               self.pool_size] = prev_delta_window

        return der_w, der_b, prev_delta
