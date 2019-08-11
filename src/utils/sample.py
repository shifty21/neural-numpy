import ctypes
import numpy as np
from ctypes import *
from numpy import ctypeslib as ct
lib = cdll.LoadLibrary(
    "/home/yakh149a/Downloads/MNIST-cnn-master/src/utils/go_multiplier.so")


class GoString(Structure):
    _fields_ = [("p", c_char_p), ("n", c_longlong)]


class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_void_p)), ("len", c_longlong),
                ("cap", c_longlong)]


class GoSlice2(Structure):
    _fields_ = [("data", POINTER(GoSlice)), ("len", c_longlong),
                ("cap", c_longlong)]


def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


temp = GoSlice((c_void_p * 5)(74, 4, 122, 9, 12), 5, 5)
# arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr1 = np.array([[1, 2, 3], [4, 5, 6]])
arr2 = np.array([[1], [4], [1]])

mfunc = wrap_function(lib, "Sort", GoSlice, [GoSlice])

v1 = arr1[0].ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
v11 = arr1[1].ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
v2 = arr2[0].ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
v21 = arr2[1].ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
v22 = arr2[2].ctypes.data_as(ctypes.POINTER(ctypes.c_void_p))
xv1 = arr1.ctypes.data_as((ctypes.POINTER(ctypes.c_void_p)))
xv2 = arr2.ctypes.data_as((ctypes.POINTER(ctypes.c_void_p)))
print(xv1)
ar1 = GoSlice(v1, 3, 3)
ar11 = GoSlice(v11, 3, 3)
ar2 = GoSlice(v2, 1, 1)
ar21 = GoSlice(v21, 1, 1)
ar22 = GoSlice(v22, 1, 1)
t = GoSlice2((GoSlice * 2)(ar1, ar11), 2, 2)
t2 = GoSlice2((GoSlice * 3)(ar2, ar21, ar22), 3, 3)
cxv1 = (cast(xv1, POINTER(c_void_p)))
print(cxv1)
x = GoSlice((c_void_p * 6)(cxv1), 6, 6)
y = GoSlice((c_void_p * 3)(xv2), 3, 3)
lib.multiply_matrix.argtypes = [GoSlice, GoSlice]
lib.multiply_matrix.restype = GoSlice
r = lib.multiply_matrix(t, t2)
print(r.data[0].cap)
