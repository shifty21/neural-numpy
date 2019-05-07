from logger import Logger
import numpy as np


class L2Regularization:

    def apply_regularization(loss, weights):
        # print ("shape of weight matrix: " + str(weights.shape))
        return loss + 0.9/(2*5000)*np.linalg.norm(weights)
