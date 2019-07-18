import numpy as np
from logger import Logger
from Behavioral.mult_8x8 import mult_8x8_approx
from Behavioral.mult_16x16 import mult_16x16_approx


class CustomMultiplier:

    total_overflow = 0
    non_overflow = 0
    log = None

    @classmethod
    def __init__(self, sigmoid_arr):
        self.log = Logger.get_logger(__name__)
        self.total_overflow = 0
        self.sigmoid_arr = sigmoid_arr

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

    @staticmethod
    def matrix_multiplication(weight_matrix, data_matrix):
        result = np.zeros((weight_matrix.shape[0], data_matrix.shape[1]),
                          dtype=np.int16)
        for i in range(len(weight_matrix)):
            for j in range(len(data_matrix[0])):
                dc = []
                for k in range(len(data_matrix)):
                    dc.append(data_matrix[k][j])
                temp_zip = 0
                for x, y in zip(dc, weight_matrix[i]):
                    temp_zip += CustomMultiplier.mul(x, y)
                    result[i][j] = temp_zip
        return (result)

    def sigmoid_activation_lut(self, arr):
        result = np.zeros((arr.shape), dtype=np.float64)
        # self.log.info("shape of arr %s", str((arr)))
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                with np.errstate(over="ignore"):
                    result[i][j] = self.sigmoid_arr[arr[i][j]]
                    # result[i][j] = 1.0 / 1.0 + np.exp(-arr[i][j])
                    # self.log.info(
                    #     "index %d value is %.10f expected value %.10f",
                    # arr[i][j], result[i][j],
                    # 1.0 / 1.0 + np.exp(-arr[i][j]))
        # self.log.info("result of lut %s ", str(result))
        return result

    @staticmethod
    def mul(x, y):
        result = np.int8(0)
        try:
            # result = mult_8x8_approx(x, y, "acc")
            # result = mult_16x16_approx(x, y, "acc")
            result = x * y
            # print("type of result " + str(type(result)))
        except RuntimeWarning:
            # print ("runtime warning")
            return np.int16(x) * np.int16(y)
        # print("type of o/p %s ", type(result))
        return result
