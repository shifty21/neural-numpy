import numpy as np
from timeit import timeit
from multiprocessing import Pool


def mmul(matrix):
    for i in range(100):
        matrix = matrix * matrix
    return matrix

if __name__ == '__main__':
    matrices = []
    for i in range(4):
        matrices.append(np.random.random_integers(100, size=(1000, 1000)))

    pool = Pool(8)
    import time
    millis = int(round(time.time() * 1000))
    map(mmul, matrices)
    end = int(round(time.time() * 1000))
    print (end- millis)
    pool.map(mmul, matrices)
    e = int(round(time.time() * 1000))
    print (e - end)
