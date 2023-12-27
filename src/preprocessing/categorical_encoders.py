from numpy.core.multiarray import array as array
from classes.transformer import Transformer
import numpy as np


class OneHotEncoder(Transformer):
    def __init__(self):
        self.__categories = None
        self.__values = None
        self.X = None

    @property
    def infrequent_categories_(self):
        return self.__categories

    @property
    def n_features_in(self):
        return self.__categories.shape[0]

    def fit(self, X: np.array, y=None):
        if X.shape[1] != 1:
            raise NotImplementedError("OneHotEncoder: 1 feature support")
        self.__categories = np.unique(X)
        self.__values = np.zeros(
            shape=(self.__categories.shape[0], self.__categories.shape[0])
        )
        for i in range(self.__categories.shape[0]):
            self.__values[i][self.__categories.shape[0] - i - 1] = 1
        return self

    def transform(self, X: np.array, y=None):
        if X.shape[1] != 1:
            raise NotImplementedError("OneHotEncoder: 1 feature support")
        result = np.zeros(shape=(X.shape[0], self.__categories.shape[0]))
        for i in range(X.shape[0]):
            index_cat = np.where(X[i] == self.__categories)
            result[i, :] = self.__values[index_cat, :]
        return result

    def fit_transform(self, X: np.array, y=None) -> np.array:
        self.fit(X, y)
        return self.transform(X, y)
