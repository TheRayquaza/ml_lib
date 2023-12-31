import numpy as np
import math


def mae(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Get the mean absolute error
    """
    return np.mean(np.abs(y1 - y2))


def mse(y1: np.ndarray, y2: np.ndarray) -> float:
    """
    Get the mean squared error
    """
    return np.mean(np.square(y1 - y2))


def rmse(y1, y2):
    """
    Get the root mean squared error
    """
    return math.sqrt(mse(y1, y2))
