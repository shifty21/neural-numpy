import abc

import numpy as np

from utils import functions as f
from utils import utils as u
from logger import Logger

class Layer(metaclass=abc.ABCMeta):
    def __init__(self):
        self.depth = None
        self.height = None
        self.width = None
        self.n_out = None
        self.w = None
        self.b = None
        self.vw = None
        self.vb = None
        self.mtw = None
        self.vtw = None
        self.mtb = None
        self.vtb = None
        self.t = 0
        self.m0w = None
        self.m0b = None
        self.u0w = 0
        self.u0b = 0

    @abc.abstractmethod
    def connect_to(self, prev_layer):
        raise AssertionError

    @abc.abstractmethod
    def feedforward(self, prev_layer):
        raise AssertionError

    @abc.abstractmethod
    def backpropagate(self, prev_layer, delta):
        raise AssertionError

    @abc.abstractmethod
    def get_weights(self,convert_to_float):
        raise AssertionError

    @abc.abstractmethod
    def get_biases(self,convert_to_float):
        return AssertionError

