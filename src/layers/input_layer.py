import numpy as np

from utils import functions as f
from utils import utils as u
from logger import Logger
from layers.interface_layer import Layer


class InputLayer(Layer):
    def __init__(self, height, width, depth=1):
        super().__init__()
        self.depth = depth
        self.height = height
        self.width = width
        self.n_out = self.depth * self.height * self.width
        self.der_act_func = lambda x: x

    def connect_to(self, prev_layer):
        raise AssertionError

    def feedforward(self, prev_layer):
        raise AssertionError

    def backpropagate(self, prev_layer, delta):
        raise AssertionError

    def get_weights(self, convert_to_float):
        return []

    def get_biases(self, convert_to_float):
        return []

    def summary(self):
        return (self.depth, self.height, self.width)
