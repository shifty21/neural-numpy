from logger import Logger
import numpy as np


class L1Regularization:

    def apply_regularization(loss, weights):
        # print ("shape of weight matrix: " + str(weights.shape))
        return loss + 0.9/(2*50000)*np.abs(weights).sum()
        
