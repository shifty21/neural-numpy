from .l2 import L2Regularization
from .l1 import L1Regularization
from logger import Logger
from datetime import datetime


class Regularize:

    def __init__(self,regularization):
        if (regularization == None):
            self.log.info('No Regularization provided')
        else :
            if (regularization!="l1" and regularization != "l2"):
                self.log.error("Only possible values are l1 and l2")
                sys.exit(-1)
            self.regularization = regularization
        self.log = Logger.get_logger(__name__)
        self.log.info("Initialized regularizer with " + regularization)

    def apply (self, loss, output_layer):
        if (self.regularization == "l1"):
            self.log.debug("Applying L1 regularization")
            return L1Regularization.apply_regularization(loss,output_layer.w)
        elif (self.regularization == "l2"):
            self.log.debug("Applying L2 regularization")
            return L2Regularization.apply_regularization(loss,output_layer.w)
