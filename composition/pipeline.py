from classes.model import Model
from classes.transformer import Transformer
import numpy as np


class Pipeline(Model):
    def __init__(self, steps=[]):
        self.steps = steps
        self.__model = None
        if len(steps) != 0:
            self.__model = self.steps[-1][-1]

    def __str__(self) -> str:
        step_names = [name for name, _ in self.steps]
        return " -> ".join(step_names)

    @property
    def weights(self):
        return self.__model.weights

    @property
    def model(self):
        return self.__model

    def fit(self, X: np.array, y: np.array):
        for _, element in self.steps:
            if issubclass(type(element), Model):
                element.fit(X, y)
            elif issubclass(type(element), Transformer):
                X = element.fit_transform(X)
            else:
                raise Exception("Pipeline: invalid argument", element)

    def predict(self, X: np.array) -> np.array:
        for _, element in self.steps:
            if issubclass(type(element), Model):
                X = element.predict(X)
            elif issubclass(type(element), Transformer):
                X = element.transform(X)
            else:
                raise Exception("Pipeline: invalid argument", element)
        return X
