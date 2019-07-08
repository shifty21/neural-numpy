import numpy as np
from logger import Logger
import sys

from utils.fixed_point import FixedPoint

### weights initializations ####################################################
class CustomDataType:

    data_type = None
    def __init__(self,dtype):
        self.log = Logger.get_logger(__name__)
        if (dtype == 32):
            self.data_type = np.float32
        elif (dtype == 64):
            self.data_type = np.float64
        else:
            self.log.error("Only permited type are float32 and float64 please enter 32 or 64")
            sys.exit(-1)
    def get_data_type(self):
        return self.data_type


float_type = np.float64
fixed = FixedPoint()
log = Logger.get_logger("functions")

def custom_multiply(array1,array2):
    result = np.multiply(array1,array2)
    result = fixed.right_shift(result)
    return result

def InitCustomDataType(dtype):
    dt = CustomDataType(dtype)
    float_type = dt.get_data_type()

def matrix_multiplication(weight_matrix, data_matrix):
    result = np.zeros((weight_matrix.shape [0], data_matrix.shape[1]),dtype=np.int16)
    # result = np.zeros((weight_matrix.shape[0], data_matrix.shape[1]))
    # print("type of weight matrix " + str(type(weight_matrix[0][0]))  + " type of data matrix " + str(type(data_matrix[0][0])))
    for i in range(len(weight_matrix)):
        for j in range(len(data_matrix[0])):
            dc = []
            for k in range(len(data_matrix)):
                dc.append(data_matrix[k][j])
            # temp = np.dot(dc,weigh t_matrix[i])
            overflow = 0
            temp_zip = 0
            for x,y in zip(dc,weight_matrix[i]):
                # if (x*y>32767):
                    # overflow = overflow + 1
                #     temp_zip+=32767
                # else:
                temp_zip+= mul(x,y)

            # print ("no of overflow"  + str(overflow) + " value of temp_zip " +str(temp_zip) )
            # print("type of temp -- " + str(type(temp_zip)) + " value of j " + str(j))
            # result[i][j] = (temp_zip)
            result[i][j] = temp_zip
    # print ("type of multiplication result " + str(type(result[0][0])))
    # max_value = np.amax(result)
    # result = np.interp(result, (result.min(), result.max()), (-128, +127))
    # print ("value of result after interpolation -- " + str(result))
    # result = np.int8(result)
    return (result)

def mul(x,y):

    return np.int16(x)*np.int16(y)

def glorot_uniform(shape, num_neurons_in, num_neurons_out):
    scale = np.sqrt(6. / (num_neurons_in + num_neurons_out))
    return np.random.uniform(low=-scale, high=scale, size=shape)

def zero(shape, *args):
    return np.zeros(shape,dtype=float_type)
    # return np.zeros(shape,dtype=get_data_type())

def array(*args,**kwargs):
    kwargs.setdefault("dtype",float_type)
    return np.array(*args, **kwargs)

def zeros_like(arr):
    return np.zeros_like(arr)

def zero_custom(shape):
    return np.zeros(shape,dtype=float_type)

def glorot_uniform_initializer(shape,num_neurons_in,num_neurons_out):
    scale = np.sqrt(6. / (num_neurons_in + num_neurons_out))
    return np.random.uniform(low=-scale, high=scale, size=shape).astype(float_type)


### activations ################################################################

def sigmoid(x):
    # with  np.errstate(over="ignore"):
    return 1.0 / (1.0 + np.exp(-x))


def der_sigmoid(x, y=None):
    s = sigmoid(x)
    return s * (1 - s)


def softmax(x):
    e = np.exp(x - np.max(x))
    return e / np.sum(e)

def der_softmax(x, y=None):
    s = softmax(x)
    if y is not None:
        k = s[np.where(y == 1)]
        a = - k * s
        a[np.where(y == 1)] = k * (1 - k)
        return a
    return s * (1 - s)

def relu(x):
    x[x<0] = 0
    return x

def der_relu(x, y = None):
    s = relu(x)
    return s*(1-s)


### objectives #################################################################

def quadratic(a, y):
    return a-y


def log_likelihood(a, y):
    return -1.0 / a[np.where(y == 1)]


def categorical_crossentropy(a, y):
    a = a.flatten() / np.sum(a)
    i = np.where(y.flatten() == 1)
    return np.log(a)[i]
