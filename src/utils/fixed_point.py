import numpy as np
from rig.type_casts import NumpyFixToFloatConverter, NumpyFloatToFixConverter


class FixedPoint:

    def __init__(self):
        self.float_to_fixed = NumpyFloatToFixConverter(signed=True, n_bits=8, n_frac=6)
        self.fixed_to_float = NumpyFixToFloatConverter(12)
        self.float64_to_fixed = NumpyFloatToFixConverter(signed=True,n_bits=8,n_frac=3)


    def convert_float_to_fixed(self,array):
        return self.float_to_fixed(array)


    def convert_fixed_to_float(self,array):
        return self.fixed_to_float(array)

    def convert_float4_to_fixed(self,array):
        return self.float64_to_fixed(array)

    def right_shift(self,array):
        return np.right_shift(array,5)
