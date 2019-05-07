import numpy as np

import functions as f
import utils as u
from logger import Logger
from layers.interface_layer import Layer


class InputLayer(Layer):
    def __init__(self, height, width):
        super().__init__()
        self.depth = 1
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