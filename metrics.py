import numpy as np
import math


def mse(y1: np.array, y2: np.array) -> float:
    """
    Get the mean squared error
    """
    if y1.shape != y2.shape:
        raise Exception("Invalid shapes")
    elif y1.shape[0] == 0:
        return 0
    else:
        nb_row = y1.shape[0]
        nb_col = y1.shape[1]
        res = 0
        for i in range(nb_row):
            for j in range(nb_col):
                res += (y1[i, j] - y2[i, j]) ** 2
        return res / (nb_row * nb_col)


def rmse(y1, y2):
    """
    Get the root mean squared error
    """
    return math.sqrt(mse(y1, y2))
