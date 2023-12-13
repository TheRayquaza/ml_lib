from itertools import combinations_with_replacement
import numpy as np
import math

class PolynomialFeatures:

    def __init__(self, degree=2):
        self.degree = degree
    
    def __generate_monomials(self, A):
        res = []
        for degree in range(self.degree + 1):
            for combination in combinations_with_replacement(A, degree):
                res.append(np.prod(combination))
        return np.array(res)

    def transform(self, X:np.array) -> np.array:
        res = []
        for i in range(X.shape[0]):
            res.append(self.__generate_monomials(X[i]))
        return np.array(res)

class StandardScaler:

    def __init__(self):
        self.means = None
        self.stds = None
        self.fitted = False
    
    def __calculate_means(self, X):
        for i in range(X.shape[1]):
            self.means[i] = np.mean(X[i, :])
    
    def __calculate_stds(self, X):
        for i in range(X.shape[1]):
            self.stds[i] = math.sqrt(np.sum(X[i, :]) / (X.shape[0] - 1))

    def fit(self, X:np.array):
        self.means = np.zeros((X.shape[1], 1))
        self.__calculate_means(X)
        self.stds = np.zeros((X.shape[1], 1))
        self.__calculate_stds(X)
        self.fitted = True

    def transform(self, X:np.array) -> np.array:
        if not self.fitted:
            raise Exception("StandardScaler: scaler not fitted")
        if self.stds.shape[0] != X.shape[1] or self.means.shape[0] != X.shape[1]:
            raise Exception("StandardScaler: invalid shapes")
        res = np.zeros_like(X)
        for j in range(X.shape[1]):
            res[:, j] = (X[:, j] - self.means[j]) / self.stds[j]
        return res
    
    def fit_transform(self, X:np.array) -> np.array:
        self.fit(X)
        return self.transform(X)

class Normalizer:

    def __init__(self):
        self.maxes = None
        self.mines = None
        self.fitted = False
    
    def __calculate_max_min(self, X):
        for i in range(X.shape[1]):
            self.maxes[i] = np.max(X[i, :])
            self.mines[i] = np.min(X[i, :])

    def fit(self, X:np.array):
        self.maxes = np.zeros((X.shape[1], 1))
        self.mines = np.zeros((X.shape[1], 1))
        self.__calculate_max_min(X)
        self.fitted = True

    def transform(self, X:np.array) -> np.array:
        if not self.fitted:
            raise Exception("Normalizer: scaler not fitted")
        if self.mines.shape[0] != X.shape[1] or self.maxes.shape[0] != X.shape[1]:
            raise Exception("Normalizer: invalid shapes")
        res = np.zeros(X.shape)
        for i in range(X.shape[1]):
            res[:, i] = (X[:, i] - self.mines[i]) / (self.maxes[i] - self.mines[i])
        return res
    
    def fit_transform(self, X:np.array) -> np.array:
        self.fit(X)
        return self.transform(X)