from classes.transformer import Transformer
from itertools import combinations_with_replacement
import numpy as np


class PolynomialFeatures(Transformer):
    def __init__(self, degree=2):
        self.degree = degree

    def __generate_monomials(self, A):
        res = []
        for degree in range(self.degree + 1):
            for combination in combinations_with_replacement(A, degree):
                res.append(np.prod(combination))
        return np.array(res)

    def transform(self, X: np.array) -> np.array:
        res = []
        for i in range(X.shape[0]):
            res.append(self.__generate_monomials(X[i]))
        return np.array(res)

    def fit(self, X: np.array):
        pass

    def fit_transform(self, X: np.array) -> np.array:
        return self.transform(X)
