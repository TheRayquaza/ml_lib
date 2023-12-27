import random
import numpy as np
from math import log10


def generate_linear_dataset(N: int, noise=True, slope=None):
    """
    Generate a linear dataset.

    Parameters
    ----------
    N : int
        The number of samples in the dataset.
    noise : bool, optional
        Whether to add random noise to the dataset (default is True).
    slope : int, optional
        The slope of the linear function. If not provided, a random slope will be generated.

    Returns
    -------
    tuple of np.ndarray
        X : np.ndarray
            The input features.
        y : np.ndarray
            The target values.
    """
    if not slope:
        a = random.randint(int(-20 * log10(N)), int(20 * log10(N)))
    else:
        a = slope
    b = random.randint(int(-20 * log10(N)), int(20 * log10(N)))
    X = np.random.rand(N, 1)
    if noise:
        y = b + a * X + np.random.randn(N, 1)
    else:
        y = b + a * X
    return X, y


def generate_polynomial_dataset(N: int, degree=2, noise=True):
    """
    Generate a polynomial dataset.

    Parameters
    ----------
    N : int
        The number of samples in the dataset.
    degree : int, optional
        The degree of the polynomial (default is 2).
    noise : bool, optional
        Whether to add random noise to the dataset (default is True).

    Returns
    -------
    tuple of np.ndarray
        X : np.ndarray
            The input features.
        y : np.ndarray
            The target values.
    """
    L = [
        random.randint(int(-20 * log10(N)), int(20 * log10(N)))
        for _ in range(degree + 1)
    ]
    X = np.random.rand(N, 1).astype(np.float64)
    y = np.zeros((N, 1), dtype=np.float64)
    for i in range(degree + 1):
        y += L[i] * X**i
    if noise:
        y += np.random.randn(N, 1).astype(np.float64)
    return X, y
