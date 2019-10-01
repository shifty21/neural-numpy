import multiprocessing
from multiprocessing import Manager
import numpy as np

from Behavioral.mult_8x8 import mult_8x8_approx
from Behavioral.mult_16x16 import mult_16x16_approx
from logger import Logger

from ctypes import cdll
from ctypes import *
import ctypes


class CustomMultiplier:

    total_overflow = 0
    non_overflow = 0
    log = None

    @classmethod
    def __init__(self, sigmoid_arr):
        self.log = Logger.get_logger(__name__)
        self.total_overflow = 0
        self.sigmoid_arr = sigmoid_arr
        # self.pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # self.log.info("Loaded library")

    @staticmethod
    def get_total_overflow():
        CustomMultiplier.total_overflow

    @staticmethod
    def get_non_overflow():
        CustomMultiplier.non_overflow

    @staticmethod
    def set_total_overflow():
        CustomMultiplier.total_overflow += 1

    @staticmethod
    def set_non_overflow():
        CustomMultiplier.non_overflow += 1

    @staticmethod
    def close():
        print("total overflow " + str(CustomMultiplier.total_overflow))
        print("non overflows " + str(CustomMultiplier.non_overflow))

    def get_data_matrix_list(data_matrix):
        dc = []
        for j in range(len(data_matrix[0])):
            for k in range(len(data_matrix)):
                dc.append(data_matrix[k][j])
        # print(dc)
        return dc

    @staticmethod
    def matrix_multiplication(weight_matrix, data_matrix, multiplier, log):
        result = np.zeros((weight_matrix.shape[0], data_matrix.shape[1]),
                          dtype=np.int16)
        dc = CustomMultiplier.get_data_matrix_list(data_matrix)
        for i in range(len(weight_matrix)):
            temp_zip = 0
            # for x, y in zip(np.asarray(dc), weight_matrix[i]):
            #     temp_zip += CustomMultiplier.mul(x, y)
            # print(
            #     "strides length in custom multiplier dc =%d and weight matrix =%d",
            #     np.asarray(dc).strides[0], weight_matrix[i].strides[0])
            # print("dc = %s, weight matrix %s \n", np.asarray(dc),
            #       weight_matrix[i])
            c_temp = multiplier(
                np.asarray(dc).ctypes.data, weight_matrix[i].ctypes.data,
                len(dc))
            # np.asarray(dc).strides[0], weight_matrix[i].strides[0])

            # if c_temp != temp_zip:
            # log.debug("value of python multiplier == " + str(temp_zip) +
            # " value of c multiplier == " + str((c_temp)))
            # log.debug(c_temp)
            result[i][0] = c_temp
            # result[i][0] = temp_zip
            # print("value inserted in array " + str(result[i][0]) +
            #       " and c_temp = " + str(c_temp))
        return (result)

    @staticmethod
    def multiply_row_column_parallely(lst, dc, weight_matrix_row, i):
        temp_zip = 0
        for x, y in zip(dc, weight_matrix_row):
            temp_zip += CustomMultiplier.mul(x, y)
            lst[i] = [temp_zip]

    def sigmoid_activation_lut(self, arr):
        result = np.zeros((arr.shape), dtype=np.float64)
        rol_len = len(arr[0])
        with np.errstate(over="ignore"):
            for i in range(len(arr)):
                for j in range(rol_len):
                    result[i][j] = self.sigmoid_arr[arr[i][j]]
        return result

    def sigmoid_activation_lut_conv(self, arr):
        result = np.zeros((arr.shape), dtype=np.float64)
        for d in range(arr.shape[0]):
            for h in range(arr.shape[1]):
                for w in range(arr.shape[2]):
                    result[d][h][w] = self.sigmoid_arr[arr[d][h][w]]
        return result

    def lookup_lut(list_row):
        pass

    @staticmethod
    def mul(x, y):
        result = np.int16(0)
        try:
            result = x * y
        except RuntimeWarning:
            return np.int16(x) * np.int16(y)
        return result
