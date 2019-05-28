import numpy as np
from rig.type_casts import NumpyFixToFloatConverter, NumpyFloatToFixConverter


class FixedPoint:

    def __init__():
        self.float_to_fixed = NumpyFloatToFixConverter(signed=True, n_bits=8, n_frac=5)
        self.fixed_to_float = NumpyFixToFloatConverter(5)


    def convert_fixed_to_float(self,array):
        return self.float_to_fixed(array)


    def convert_fixed_to_float(self,array):
        return self.fixed_to_float(array)


    def right_shift(self,array):
        return np.right_shift(array,5)
