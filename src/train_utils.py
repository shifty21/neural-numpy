
import numpy as np
import utils as u
from logger import Logger
from network import NeuralNetwork
from utils.fixed_point import FixedPoint

class Train:
    def __init__ (self):
        self.log = Logger.get_logger(__name__)

    def train(self,net, optimizer, num_epochs, batch_size, trn_set, tst_set, vld_set=None):

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
                u.print("Epoch %02d %s [%d/%d] > Testing \n" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)
                accuracy = self.test(net, vld_set)
                u.print("Epoch %02d %s [%d/%d] > Validation \n" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)
            u.print()
            u.print("Testing network...", bcolor=u.bcolors.BOLD)
            accuracy = self.test(net, tst_set)
            u.print("Test accuracy: %0.2f%%" % (accuracy*100))
        # self.save(net, "np_weights.npz")

    def test(self, net, tst_set):
        assert isinstance(net, NeuralNetwork)

        tst_x, tst_y = tst_set
        tests = [(x, y) for x, y in zip(tst_x, tst_y)]
        accuracy = 0
        inputs_len = len(tests)
        inputs_done =0
        for x, y in tests:
            inputs_done+=1
            net.feedforward(x)
            u.print("%s [%d/%d] > Testing..." % (u.bar(inputs_done, inputs_len), inputs_done, inputs_len), override=True)
            if np.argmax(net.output_layer.a) == np.argmax(y):
                accuracy += 1
        accuracy /= len(tests)

        return accuracy


    def test_inference(self, net, tst_set):
        assert isinstance(net, NeuralNetwork)

        tst_x, tst_y = tst_set
        tests = [(x, y) for x, y in zip(tst_x, tst_y)]
        inputs_len = len(tests)
        inputs_done =0
        accuracy = 0
        # np.set_printoptions(suppress=True)
        converter = FixedPoint()
        for x, y in tests:
            inputs_done+=1
            net.feedforward(x,True)
            u.print("%s [%d/%d] > Testing... >> accuracy  --- %f" % (u.bar(inputs_done, inputs_len), inputs_done, inputs_len, accuracy/inputs_done*100), override=True)
            if np.argmax(net.output_layer.a) == np.argmax(y):
                accuracy += 1
        accuracy /= inputs_len

        # self.save(net, "retrain_weights.npz", True)
        return accuracy

    def save(self, net, filename, convert_to_float=False):
        with open(filename, "wb") as f:
            w = list()
            b = list()
            last_layer = 0
            debug = False
            for prev_layer, layer in net.layers:
                weights = prev_layer.get_weights(convert_to_float)
                biases = prev_layer.get_biases(convert_to_float)
                if len(weights) > 0:
                    self.log.debug("printing weights for layer: " + str(prev_layer) + " weights shape " + str(weights.shape))
                    w.append(weights)
                if len(biases) > 0:
                    self.log.debug("printing biases for layer: " + str(prev_layer) + " biases shape " + str(biases.shape))
                    b.append(biases)
                last_layer = layer
            weights = last_layer.get_weights(convert_to_float)
            biases = last_layer.get_biases(convert_to_float)
            if len(weights) > 0:
                self.log.debug("printing weights for last layer: " + str(layer) + " weights shape " + str(weights.shape))
                w.append(weights)
            if len(biases) > 0:
                self.log.debug("printing biases for last layer: " + str(layer) + " biases shape " + str(biases.shape))
                b.append(biases)
            np.savez_compressed(f, w=w,b=b)
            self.log.debug("length of weights is " + str(len(w)) + " length of biases is " + str(len(b)))
