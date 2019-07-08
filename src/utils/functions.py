import numpy as np
from logger import Logger
import sys
from utils.fixed_point import FixedPoint
from train_utils import Train
import warnings
warnings.filterwarnings("error")

class CustomDataType:

    data_type = None
    def __init__(self,dtype=64):
        self.log = Logger.get_logger(__name__)
        if (dtype == 32):
            self.data_type = np.float32
        elif (dtype == 64):
            self.data_type = np.float64
        else:
            self.log.error("Only permited type are float32 and float64 please enter 32 or 64")
            sys.exit(-1)
        self.total_overflow = 0
    def get_data_type(self):
        return self.data_type

    def get_overflows(self):
        return self.total_overflow

    def set_overflows(self):
        self.total_overflow+=1
        print ("total overflows" + str(self.total_overflow))


float_type = np.float64
fixed = FixedPoint()
log = Logger.get_logger("functions")
train = Train()
dt = None


def InitCustomDataType(dtype):
    dt = CustomDataType(dtype)
    float_type = dt.get_data_type()

def matrix_multiplication(weight_matrix, data_matrix):
    result = np.zeros((weight_matrix.shape [0], data_matrix.shape[1]),dtype=np.int16)
    for i in range(len(weight_matrix)):
        for j in range(len(data_matrix[0])):
            dc = []
            for k in range(len(data_matrix)):
                dc.append(data_matrix[k][j])
            temp_zip = 0
            for x,y in zip(dc,weight_matrix[i]):
                temp_zip+= mul(x,y)
            result[i][j] = temp_zip
    return (result)

def mul(x,y):
    result = np.int8(0)
    try:
        result = x*y
    except RuntimeWarning:
        dt.set_overflow()
        return np.int16(x)*np.int16(y)
    return result

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
    with  np.errstate(over="ignore"):
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
    x[x>0] = 1
    return x

### objectives #################################################################

def quadratic(a, y):
    return a-y


def log_likelihood(a, y):
    return -1.0 / a[np.where(y == 1)]


def categorical_crossentropy(a, y):
    a = a.flatten() / np.sum(a)
    i = np.where(y.flatten() == 1)
    return np.log(a)[i]
