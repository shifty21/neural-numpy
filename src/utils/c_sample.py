# test.py
import numpy as np
from numpy.ctypeslib import ndpointer
import ctypes
import warnings
warnings.filterwarnings('error')
_doublepp = ndpointer(dtype=np.int, ndim=2, flags='C_CONTIGUOUS')

_dll = ctypes.CDLL(
    '/home/yakh149a/Downloads/MNIST-cnn-master/src/utils/libclib1.so')

_mul = _dll.matrix_multiply
_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_mul.restype = ctypes.c_int

_accurate_dll = ctypes.CDLL(
    '/home/yakh149a/Downloads/MNIST-cnn-master/src/cpp_multiplier/behavioral.so'
)
_custom_mul = _accurate_dll.matrix_multiply
_custom_mul.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
_custom_mul.restype = ctypes.c_int

# _print = _dll.foobar
# _print.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
# _print.restype = None


def foobar(x, y):
    # print("address value of first element " + str(x.ctypes.data))
    print("strides " + str(x.strides) + " item size " + str(x.itemsize))
    # _print(y.ctypes.data, ctypes.c_int(len(y)), ctypes.c_int(len(y)))
    z = _mul(x.ctypes.data, y.ctypes.data, ctypes.c_int(len(x)))
    print("result in python = " + str(z))


def mul(x, y):
    result = np.int32(0)
    try:
        result = x * y
    except RuntimeWarning:
        print("overflow in x == " + str(x) + " y == " + str(y))
        return np.int32(x) * np.int32(y)
    return result


def custom_multiplier(x, y, l):
    return _custom_mul(x, y, l)


if __name__ == '__main__':
    x = np.array([-7, 13, 2], dtype=np.int8)
    y = np.array([13, -10, -13], dtype=np.int8)
    # foobar(x, y)
    result = custom_multiplier(x.ravel().ctypes.data,
                               y.ravel().ctypes.data, len(y))
    print("python result  - " + str(result))
    print("numpy --- " + str(np.correlate(x.ravel(), y.ravel(), mode="valid")))
