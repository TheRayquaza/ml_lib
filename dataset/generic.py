import random
import numpy as np
from math import log10

def generate_linear_dataset(N: int, noise=True):
    a = random.randint(int(-20*log10(N)), int(20*log10(N)))
    b = random.randint(int(-20*log10(N)), int(20*log10(N)))
    X = np.random.rand(N, 1)
    if noise:
        y = b + a * X + np.random.randn(N, 1)
    else:
        y = b + a * X
    return X, y

def generate_polynomial_dataset(N: int, degree=2, noise=True):
    L = [random.randint(int(-20*log10(N)), int(20*log10(N))) for _ in range(degree + 1)]
    X = np.random.rand(N, 1).astype(np.float64)
    y = np.zeros((N, 1), dtype=np.float64)
    for i in range(degree + 1):
        y += L[i] * X ** i
    if noise:
        y += np.random.randn(N, 1).astype(np.float64)
    return X, y