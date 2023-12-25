from classes.model import Model
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class KNeighborsRegressor(Model):
    def __init__(self, k=5, n_jobs=None):
        if k < 0:
            raise ValueError("KNeighborsRegressor: k should be non-negative")
        self.k = k
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.fitted = False

    def fit(self, X: np.array, y: np.array):
        self.X = np.array(X)
        self.y = np.array(y)
        self.classes = np.unique(self.y)
        self.fitted = True
        return self

    def __quick_ascending_insert(self, x: list, v: float) -> int:
        i = len(x)
        x.append(v)
        while i > 0 and x[i - 1] > v:
            x[i], x[i - 1] = x[i - 1], x[i]
            i -= 1
        return i

    def __euclidean_distance(self, x1: np.array, x2: np.array) -> float:
        return np.linalg.norm(x1 - x2)

    def __make_prediction(self, X: np.array):
        distances, values = [], []
        for i in range(self.X.shape[0]):
            j = self.__quick_ascending_insert(
                distances, self.__euclidean_distance(self.X[i], X)
            )
            values.insert(j, self.y[i, 0])
        best_values = values[: min(self.k, len(values))]
        return np.mean(np.array(best_values))

    def predict(self, X: np.array):
        X = np.array(X)
        if not self.fitted:
            raise Exception("KNeighborsRegressor: not fitted")
        result = np.zeros((X.shape[0], 1))
        if not self.n_jobs:
            for i in range(X.shape[0]):
                result[i] = self.__make_prediction(X[i])
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self.__make_prediction, X[i]): i for i in range(X.shape[0])
            }
            for future in as_completed(future_to_pred):
                result[future_to_pred[future]] = future.result()
        return result
