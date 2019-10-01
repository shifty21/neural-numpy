from .input_layer import InputLayer
from .convolutional_layer import ConvolutionalLayer
from .fully_connected_layer import FullyConnectedLayer
from .max_pooling_layer import MaxPoolingLayer
from .flatten import Flatten

# from os.path import dirname, basename, isfile, join
# import glob
# modules = glob.glob(join(dirname(__file__), "*.py"))
# __all__ = [ basename(f)[:-3] for f in modules if isfile(f) and not f.endswith('__init__.py')]

#needs to be class names not file names
__all__ = [
    "InputLayer", "ConvolutionalLayer", "FullyConnectedLayer",
    "MaxPoolingLayer", "Flatten"
]
