from logger import Logger
import numpy as np

class Dropout(object):
    def __init__ (self, size_a1, size_a2,dropout_rate):
        self.dropout_array = np.random.rand(size_a1, size_a2) < dropout_rate
