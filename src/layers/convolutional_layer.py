import numpy as np

from utils import functions as f
from utils import utils as u
from logger import Logger

from layers.interface_layer import Layer

from utils.fixed_point import FixedPoint
from ctypes import cdll
from ctypes import c_int8, c_int
from numpy.ctypeslib import ndpointer
from ctypes import *


class ConvolutionalLayer(Layer):
    def __init__(self, depth, kernel_size, init_func, act_func, dropout=False):
        super().__init__()
        self.log = Logger.get_logger(__name__)
        self.log.info('ConvolutionalLayer init')
        self.depth = depth
        self.kernel_size = kernel_size
        self.init_func = init_func
        self.act_func = act_func
        self.der_act_func = getattr(f, "der_%s" % act_func.__name__)
        self.dropout = dropout
        self.dropout_rate = 0.9
        self.fixedConverter = FixedPoint()
        self.lib = cdll.LoadLibrary(
            '/home/yakh149a/Downloads/MNIST-cnn-master/src/utils/libclib1.so')

    def get_weights(self, convert_to_float):
        if convert_to_float:
            return self.fixedConverter.convert_fixed_to_float(self.w)
        return self.w

    def get_biases(self, convert_to_float):
        if convert_to_float:
            return self.fixedConverter.convert_fixed_to_float(self.b)
        return self.b

    def config_lib(self, w, prev_a):
        import ctypes
        self._mul = self.lib.matrix_multiply
        self._mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
        self._mul.restype = ctypes.c_int

    def connect_to(self, prev_layer):
        self.stride_length = 1
        self.height = (
            (prev_layer.height - self.kernel_size) // self.stride_length) + 1
        self.width = (
            (prev_layer.width - self.kernel_size) // self.stride_length) + 1
        self.n_out = self.depth * self.height * self.width

        self.w = self.init_func(
            (self.depth, prev_layer.depth, self.kernel_size, self.kernel_size),
            prev_layer.n_out, self.n_out)
        self.b = f.zero_custom((self.depth, 1))
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
            feature maps, where each feature map is a 2d matrix
        """
        # print("w.shape " + str(self.w.shape))
        # print("depth " + str(self.depth) + " prev_layer.depth " +
        #       str(prev_layer.depth) + " self.kernel_size " +
        #       str(self.kernel_size) + " self.kernel_size " +
        # str(self.kernel_size))
        assert self.w.shape == (self.depth, prev_layer.depth, self.kernel_size,
                                self.kernel_size)
        assert self.b.shape == (self.depth, 1)
        assert prev_layer.a.ndim == 3
        prev_a = prev_layer.a

        if (self.dropout):
            dropout_matrix = np.random.rand(
                prev_a.shape[0], prev_a.shape[1]) < self.dropout_rate
            prev_a = np.multiply(prev_a, dropout_matrix)
            prev_a = prev_a / self.dropout_rate

        self.config_lib(self.w, prev_a)
        filters_c_out = self.w.shape[0]
        filters_c_in = self.w.shape[1]
        filters_h = self.w.shape[2]
        filters_w = self.w.shape[3]
        image_c = prev_a.shape[0]
        assert image_c == filters_c_in
        image_h = prev_a.shape[1]
        image_w = prev_a.shape[2]

        stride = 1
        new_h = ((image_h - filters_h) // stride) + 1
        new_w = ((image_w - filters_w) // stride) + 1
        range_h = range(0, image_h - filters_h + 1, self.stride_length)
        range_w = range(0, image_w - filters_w + 1, self.stride_length)
        self.z = np.zeros((filters_c_out, new_h, new_w), dtype=np.int16)
        for r in range(filters_c_out):
            for t in range(image_c):
                filter = self.w[r, t]
                filter_ravel = filter.ravel()
                ctypes_filter_ravel = filter_ravel.ctypes.data
                for i, m in enumerate(range_h):
                    for j, n in enumerate(range_w):
                        prev_a_window = prev_a[t, m:m + filters_h, n:n +
                                               filters_w]
                        c_correl = 0
                        if inference == True:
                            # self.log.info("type of weight - %s",
                            #               type(self.z[0][0][0]))
                            c_correl += self._mul(
                                prev_a_window.ravel().ctypes.data,
                                ctypes_filter_ravel, len(filter_ravel))
                        else:
                            numpy_correl = np.correlate(
                                prev_a_window.ravel(),
                                filter_ravel,
                                mode="valid")
                        # self.log.info(
                        #     "type of prev_a - %s , type of filter_ravel - %s",
                        #     type(prev_a_window.ravel()[0]),
                        #     type(filter_ravel[0]))
                        numpy_correl = np.correlate(
                            prev_a_window.ravel(), filter_ravel, mode="valid")
                        # self.log.info("c_correl - %s , numpp_correl - %s",
                        #               c_correl, numpy_correl)
                        self.z[r, i, j] += c_correl

        for r in range(self.depth):
            self.z[r] += self.b[r]

        self.a = np.vectorize(self.act_func)(self.z)
        assert self.a.shape == self.z.shape

    def get_prev_a_window(self, prev_a, t, v, h):
        return prev_a[t, v:v + self.height - self.kernel_size +
                      1:self.stride_length, h:h + self.width -
                      self.kernel_size + 1:self.stride_length]

    def delta_window(self, delta, r, v, h):
        return delta[r, v:v + self.height - self.kernel_size +
                     1:self.stride_length, h:h + self.width -
                     self.kernel_size + 1:self.stride_length]

    def backpropagate(self, prev_layer, delta):
        """
        Backpropagate the error through the layer

        :param prev_layer: the previous layer of the network. The activations of the previous layer are a list of
            feature maps, where each feature map is a 2d matrix
        :param delta:
        """
        assert delta.shape[0] == self.depth

        prev_a = prev_layer.a

        if (self.dropout):
            dropout_matrix = np.random.rand(
                prev_a.shape[0], prev_a.shape[1]) < self.dropout_rate
            prev_a = np.multiply(prev_a, dropout_matrix)
            prev_a = prev_a / self.dropout_rate
        der_w = np.empty_like(self.w)
        for r in range(self.depth):
            for t in range(prev_layer.depth):
                for h in range(self.kernel_size):
                    for v in range(self.kernel_size):
                        prev_a_window = self.get_prev_a_window(prev_a, t, v, h)
                        delta_window = self.delta_window(delta, r, v, h)
                        assert prev_a_window.shape == delta_window.shape
                        der_w[r, t, h, v] = np.sum(
                            prev_a_window * delta_window)

        der_b = np.empty((self.depth, 1))
        for r in range(self.depth):
            der_b[r] = np.sum(delta[r])

        prev_delta = f.zeros_like(prev_a)
        range_h = range(0, prev_layer.height - self.kernel_size + 1,
                        self.stride_length)
        range_w = range(0, prev_layer.width - self.kernel_size + 1,
                        self.stride_length)
        for r in range(self.depth):
            for t in range(prev_layer.depth):
                kernel = self.w[r, t]
                for i, m in enumerate(range_h):
                    for j, n in enumerate(range_w):
                        prev_delta[t, m:m + self.kernel_size, n:n +
                                   self.kernel_size] += kernel * delta[r, i, j]
        prev_delta *= prev_layer.der_act_func(prev_layer.z)

        return der_w, der_b, prev_delta
