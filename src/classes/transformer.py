from abc import ABC, abstractmethod
import numpy as np


class Transformer(ABC):
    def __str__(self) -> str:
        return "Transformer"

    @abstractmethod
    def fit(self, X: np.array, y: np.array) -> np.array:
        pass

    @abstractmethod
    def transform(self, X: np.array, y: np.array) -> np.array:
        pass

    @abstractmethod
    def fit_transform(self, X: np.array, y: np.array) -> np.array:
        pass
