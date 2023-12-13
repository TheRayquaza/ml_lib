from abc import ABC, abstractmethod
import numpy as np

class Model:
    @abstractmethod
    def fit(self, X:np.array, y:np.array) -> np.array:
        pass

    @abstractmethod
    def predict(self, X:np.array, y:np.array) -> np.array:
        pass
