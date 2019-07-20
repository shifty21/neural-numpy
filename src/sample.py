import numpy as np
from timeit import timeit
from multiprocessing import Pool, Process, Manager, Value
import multiprocessing


def sigmoid(x):
    with np.errstate(over="ignore"):
        dict[x] = (1.0 / 1.0 + np.exp(-x))


if __name__ == '__main__':

    manager = Manager()
    dict = manager.dict()
    min = np.iinfo(np.int16).min
    max = np.iinfo(np.int16).max
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    pool.map(sigmoid, range(min, max))
    print(type(dict))
