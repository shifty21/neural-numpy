import numpy as np

import layers as l
import utils as u
import time
from logger import Logger
from regularization import L1Regularization
from regularization import L2Regularization

class NeuralNetwork():


    def __init__(self, layers, loss_func, regularization):
        self.log = Logger.get_logger(__name__)
        self.log.info('NeuralNetwork Initialized')
        assert len(layers) > 0

        assert isinstance(layers[0], l.InputLayer)
        self.input_layer = layers[0]

        assert isinstance(layers[-1], l.FullyConnectedLayer)
        self.output_layer = layers[-1]

        self.layers = [(prev_layer, layer) for prev_layer, layer in zip(layers[:-1], layers[1:])]

        self.loss_func = loss_func
        self.regularization = regularization
        if (regularization == None):
            self.log.info('No Regularization provided')
        else :
            self.log.info("Regularization :- " + regularization)

        for prev_layer, layer in self.layers:
            layer.connect_to(prev_layer)

        self.log.info('NeuralNetwork Initialized')


    def feedforward(self, x):
        self.input_layer.z = x
        self.input_layer.a = x
        for prev_layer, layer in self.layers:
            layer.feedforward(prev_layer)


    def backpropagate(self, batch, optimizer):
        sum_der_w = {layer: np.zeros_like(layer.w) for _, layer in self.layers}
        sum_der_b = {layer: np.zeros_like(layer.b) for _, layer in self.layers}

        for x, y in batch:
            self.feedforward(x)

            # propagate the error backward
            loss = self.loss_func(self.output_layer.a, y)
            if (self.regularization == "l1"):
                self.log.debug("Applying L1 regularization")
                loss = L1Regularization.apply_regularization(loss,self.output_layer.w)
            elif (self.regularization == "l2"):
                self.log.debug("Applying L2 regularization")
                loss = L2Regularization.apply_regularization(loss,self.output_layer.w)

            delta = loss * self.output_layer.der_act_func(self.output_layer.z, y)
            for prev_layer, layer in reversed(self.layers):
                der_w, der_b, prev_delta = layer.backpropagate(prev_layer, delta)
                sum_der_w[layer] += der_w
                sum_der_b[layer] += der_b
                delta = prev_delta
        # update weights and biases to find the local minia
        optimizer.apply(self.layers, sum_der_w, sum_der_b, len(batch))
