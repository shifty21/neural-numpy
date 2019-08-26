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
_custom_mul.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
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
    # temp_zip = 0
    # x = np.array([
    #     0, -5, -6, 2, 0, 0, 2, -7, 0, -2, 5, 0, 0, -1, -1, 3, 2, 0, -6, 4, -4,
    #     6, 3, -5, 2, -2, -2, -1, 4, -4, -3, -2, 0, 3, -4, 1, -5, 7, 0, 0, 4, 1,
    #     0, 0, 0, 1, -5, 1, 0, -1, 5, -6, 1, -3, 3, -1, 0, 3, 4, 0, -4, -5, 0,
    #     -1, 3, 2, 1, -1, 4, -3, -4, -3, -1, -2, -7, 3, -1, -3, -5, -9, 5, -2,
    #     4, -2, 4, 2, 4, 0, 0, 6, 0, -3, 8, -4, 2, -2, -2, 1, -2, 0, 0, 2, 1,
    #     -4, -1, 0, 5, 4, -1, 3, 1, 0, 0, 1, -2, 1, 1, 0, -1, -4, -3, -7, 0, 2,
    #     -5, -5, -3, 0, 0, 0, 4, 2, -5, -6, -1, -4, 3, 1, 9, 5, -3, 2, -4, 2,
    #     -6, -4, -7, 2, -1, 2, -2, 0, 0, 5, 2, 4, -1, -2, 2, 0, 0, -1, 0, -3,
    #     -4, -5, 2, -5, 5, -7, 8, -7, 0, -5, 0, -6, -6, -5, 0, -2, -1, 6, -1,
    #     -1, 0, 1, -3, -3, -5, -1, -3, -3, 2, 1, -3, -2, -5, -3, -3, 0, 2, 2,
    #     -1, -1, 3, -5, -2, -4, 0, -2, -6, 0, -2, 0, -4, -2, -4, -2, 1, -9, -9,
    #     0, 0, 10, 0, 0, -3, 0, 5, -3, 0, 0, -4, 0, -1, 0, -5, 0, 0, -8, -6, 0,
    #     1, -3, 2, -2, 0, -1, 0, -6, -3, 7, -6, 0, 2, 6
    # ],
    #              dtype=np.int8)
    # y = np.array([
    #     16, 0, 0, 0, 0, 0, 16, 16, 0, 0, 0, 0, 0, 16, 16, 0, 0, 0, 0, 0, 0, 0,
    #     0, 15, 0, 16, 0, 0, 16, 0, 0, 0, 0, 0, 0, 16, 0, 0, 16, 0, 0, 0, 0, 0,
    #     0, 16, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 16, 0, 16, 0, 0, 0, 0, 0, 0,
    #     16, 0, 16, 0, 16, 0, 0, 0, 16, 16, 16, 0, 0, 16, 16, 0, 0, 0, 16, 0, 0,
    #     0, 0, 0, 16, 0, 0, 0, 0, 16, 0, 0, 16, 15, 0, 16, 16, 0, 16, 0, 0, 16,
    #     16, 0, 0, 0, 16, 0, 16, 16, 0, 0, 0, 0, 15, 0, 16, 0, 16, 0, 0, 0, 16,
    #     16, 8, 16, 0, 0, 16, 16, 0, 0, 16, 0, 16, 0, 16, 0, 0, 0, 0, 16, 0, 0,
    #     0, 15, 0, 0, 0, 0, 16, 16, 0, 16, 0, 0, 0, 16, 0, 15, 0, 0, 16, 0, 16,
    #     16, 15, 16, 16, 0, 0, 0, 0, 11, 0, 0, 0, 0, 16, 16, 0, 0, 15, 0, 16, 0,
    #     0, 0, 16, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 16, 16, 0, 16, 0, 0, 0, 16, 0,
    #     16, 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 16, 0, 0, 0, 0, 0, 16, 0, 0, 16,
    #     0, 0, 0, 0, 0, 16, 0, 0, 0, 16, 16, 0, 0, 0, 16, 0, 16, 0, 0, 0, 0
    # ],
    #              dtype=np.int8)
    # z = [1, 3, 4, 5]
    # print(type(z))
    # print(np.asanyarray(z).flags['C_CONTIGUOUS'])
    # print(x.flags['C_CONTIGUOUS'])
    # print(x.flags['C_CONTIGUOUS'])
    # counter = 0
    # for i, j in zip(x, y):
    #     try:
    #         if (i != 0 and j != 0):
    #             temp_zip += i * j
    #     except RuntimeWarning:
    #         print("overflow encountered ")
    #         temp_zip += np.int16(i) * np.int16(j)
    #     counter = counter + 1
    #     print("counter=" + str(counter) + " x=" + str(i) + " y=" + str(j) +
    #           " prod=" + str(np.int16(i) * np.int16(j)) + " result=" +
    #           str(temp_zip))
