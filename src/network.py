import numpy as np

import layers as l
import utils as u


class NeuralNetwork():
    def __init__(self, layers, loss_func):
        assert len(layers) > 0

        assert isinstance(layers[0], l.InputLayer)
        self.input_layer = layers[0]

        assert isinstance(layers[-1], l.FullyConnectedLayer)
        self.output_layer = layers[-1]

        self.layers = [(prev_layer, layer) for prev_layer, layer in zip(layers[:-1], layers[1:])]

        self.loss_func = loss_func

        for prev_layer, layer in self.layers:
            layer.connect_to(prev_layer)

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
            delta = loss * self.output_layer.der_act_func(self.output_layer.z, y)
            for prev_layer, layer in reversed(self.layers):
                der_w, der_b, prev_delta = layer.backpropagate(prev_layer, delta)
                sum_der_w[layer] += der_w
                sum_der_b[layer] += der_b
                delta = prev_delta
        # update weights and biases to find the local minia
        optimizer.apply(self.layers, sum_der_w, sum_der_b, len(batch))


def train(net, optimizer, num_epochs, batch_size, trn_set, vld_set=None):
    assert isinstance(net, NeuralNetwork)
    assert num_epochs > 0
    assert batch_size > 0

    trn_x, trn_y = trn_set
    inputs = [(x, y) for x, y in zip(trn_x, trn_y)]

    for i in range(num_epochs):
        np.random.shuffle(inputs)

        # divide input observations into batches
        batches = [inputs[j:j+batch_size] for j in range(0, len(inputs), batch_size)]
        inputs_done = 0
        for j, batch in enumerate(batches):
            net.backpropagate(batch, optimizer)
            inputs_done += len(batch)
            u.print("Epoch %02d %s [%d/%d]" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)

        if vld_set:
            # test the net at the end of each epoch
            u.print("Epoch %02d %s [%d/%d] > Testing..." % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)
            accuracy = test(net, vld_set)
            u.print("Epoch %02d %s [%d/%d] > Validation accuracy: %0.2f%%" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs), accuracy*100), override=True)
        u.print()

def test(net, tst_set):
    assert isinstance(net, NeuralNetwork)

    tst_x, tst_y = tst_set
    tests = [(x, y) for x, y in zip(tst_x, tst_y)]

    accuracy = 0
    for x, y in tests:
        net.feedforward(x)
        if np.argmax(net.output_layer.a) == np.argmax(y):
            accuracy += 1
    accuracy /= len(tests)
    return accuracy
