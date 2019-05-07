
import numpy as np
import utils as u
from logger import Logger


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
                u.print("Epoch %02d %s [%d/%d] > Testing..." % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs)), override=True)
                accuracy = self.test(net, vld_set)
                u.print("Epoch %02d %s [%d/%d] > Validation accuracy: %0.2f%%" % (i+1, u.bar(inputs_done, len(inputs)), inputs_done, len(inputs), accuracy*100), override=True)
                self.log.debug("Validation accuracy: %0.2f%%" % (accuracy*100))
            u.print()
            u.print("Testing network...", bcolor=u.bcolors.BOLD)
            accuracy = self.test(net, tst_set)
            u.print("Test accuracy: %0.2f%%" % (accuracy*100))
            self.log.debug("Test accuracy: %0.2f%%" % (accuracy*100))


    def test(self, net, tst_set):
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
