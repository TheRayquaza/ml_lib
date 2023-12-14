from classes.transformer import Transformer
from itertools import combinations_with_replacement
import numpy as np
import math


class StandardScaler(Transformer):
    def __init__(self):
        self.means = None
        self.stds = None
        self.fitted = False

    def __calculate_means(self, X):
        self.means = np.mean(X, axis=0)

    def __calculate_stds(self, X):
        self.stds = np.std(X, axis=0)

    def fit(self, X: np.array):
        self.means = np.zeros(X.shape[1])
        self.__calculate_means(X)
        self.stds = np.zeros(X.shape[1])
        self.__calculate_stds(X)
        self.fitted = True

    def transform(self, X: np.array) -> np.array:
        if not self.fitted:
            raise Exception("StandardScaler: not fitted")
        if self.stds.shape[0] != X.shape[1] or self.means.shape[0] != X.shape[1]:
            raise Exception("StandardScaler: invalid shapes")
        res = (X - self.means) / self.stds
        return res

    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform(X)


class Normalizer(Transformer):
    def __init__(self):
        self.maxes = None
        self.mines = None
        self.fitted = False

    def __calculate_max_min(self, X):
        self.maxes = np.max(X, axis=0)
        self.mines = np.min(X, axis=0)

    def fit(self, X: np.array):
        self.maxes = np.zeros(X.shape[1])
        self.mines = np.zeros(X.shape[1])
        self.__calculate_max_min(X)
        self.fitted = True

    def transform(self, X: np.array) -> np.array:
        if not self.fitted:
            raise Exception("Normalizer: not fitted")
        if self.mines.shape[0] != X.shape[1] or self.maxes.shape[0] != X.shape[1]:
            raise Exception("Normalizer: invalid shapes")
        res = (X - self.mines) / (self.maxes - self.mines)
        return res

    def fit_transform(self, X: np.array) -> np.array:
        self.fit(X)
        return self.transform(X)
