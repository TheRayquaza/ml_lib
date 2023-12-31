import numpy as np


class Model:
    def __init__(self, random_state=None, name="Model"):
        self.name = name
        np.random.seed(random_state)
        self._fitted = False

    def __str__(self):
        attributes = ", ".join([f"{key}={value}" for key, value in self.__dict__.items() if not isinstance(value, np.ndarray) and not key.startswith("_") and not key == "name"])
        output = f"{self.name}({attributes})" if attributes else f"{self.name}()"
        if len(output) > 60:
            return output[:53] + "..."
        else:
            return output

    def __repr__(self) -> str:
        return self.__str__()
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        self._fitted = True
        self.samples, self.features = X.shape
        self.X = X
        self.y = y
        if self.samples != self.y.shape[0]:
            raise ValueError(f"{self.name}: invalid shape having {self.samples} samples for {y.shape[0]} target values")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise ValueError(f"{self.name}: model not fitted")
        elif X.shape[1] != self.features:
            raise ValueError(f"{self.name}: invalid number of features (having {X.shape[1]} while expecting {self.features})")
        return None

    def fit_predict(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        self.fit(X, y)
        return self.predict(X)
