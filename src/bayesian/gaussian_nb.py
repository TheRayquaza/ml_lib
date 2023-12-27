from classes.model import Model
import numpy as np


class GaussianNB(Model):
    def __init__(self):
        pass

    def __str__(self):
        return "GaussianNB"

    def fit(self, X: np.array, y: np.array):
        return self

    def predict(self, X: np.array):
        pass
