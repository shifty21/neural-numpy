from .l2 import L2Regularization
from .l1 import L1Regularization
from logger import Logger
from datetime import datetime


class Regularize:

    def __init__(self,regularization):
        self.log = Logger.get_logger(__name__)
        self.regularizer_function = None
        if (regularization == None):
            self.log.info('No Regularization provided')
        else :
            if (regularization!="l1" and regularization != "l2"):
                self.log.error("Only possible values are l1 and l2")
                sys.exit(-1)
            if (regularization == "l1"):
                self.regularizer_function = L1Regularization.apply_regularization
            elif (regularization == "l2"):
                self.regularizer_function = L2Regularization.apply_regularization
            self.log.info("Initialized regularizer with " + str(regularization))

    def apply (self, loss, output_layer):
        if (self.regularizer_function == None): return loss
        return self.regularizer_function(loss,output_layer.w)
