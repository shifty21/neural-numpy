import numpy as np

from layers.interface_layer import Layer
from utils import functions as f
from logger import Logger
from utils.fixed_point import FixedPoint


class Flatten(Layer):
    def __init__(self, height, width, depth):
        super().__init__()
        self.log = Logger.get_logger(__name__)
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
        self.w = np.empty((0))
        self.b = np.empty((0))
        self.n_out = prev_layer.depth * prev_layer.height * prev_layer.width

    def feedforward(self, prev_layer, inference_custom=False):
        self.shape = prev_layer.a
        self.a = prev_layer.a.flatten()

    def backpropagate(self, prev_layer, delta):
        return 0, 0, delta

    def summary(self):
        return (self.n_out, 1)
