import numpy as np
from logger import Logger



class CustomMultiplier:

    total_overflow = 0
    non_overflow = 0
    def __int__():
        CustomMultiplier.total_overflow=0

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
        print ("total overflow " + str(CustomMultiplier.total_overflow))
        print ("non overflows " + str(CustomMultiplier.non_overflow))


    @staticmethod
    def matrix_multiplication(weight_matrix, data_matrix):
        result = np.zeros((weight_matrix.shape [0], data_matrix.shape[1]),dtype=np.int16)
        for i in range(len(weight_matrix)):
            for j in range(len(data_matrix[0])):
                dc = []
                for k in range(len(data_matrix)):
                    dc.append(data_matrix[k][j])
                    temp_zip = 0
                for x,y in zip(dc,weight_matrix[i]):
                    temp_zip+= CustomMultiplier.mul(x,y)
                    result[i][j] = temp_zip
        return (result)

    @staticmethod
    def mul(x,y):
        result = np.int8(0)
        try:
            result = x*y
            # CustomMultiplier.set_non_overflow()
        except RuntimeWarning:
            # CustomMultiplier.set_total_overflow()
            return np.int16(x)*np.int16(y)
        return result
        # return np.int16(x)*np.int16(y)

