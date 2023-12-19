from classes.model import Model
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class KNeighborsClassifier(Model):
    def __init__(self, k=5, n_jobs=-1):
        if k < 0:
            raise ValueError("KNeighborsClassifier: k should be non-negative")
        self.k = k
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.fitted = False

    def fit(self, X: np.array, y: np.array):
        self.X = X
        self.y = y
        self.classes = np.unique(y)
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
        distances, classes = [], []
        for i in range(self.X.shape[0]):
            j = self.__quick_ascending_insert(
                distances, self.__euclidean_distance(self.X[i], X)
            )
            classes.insert(j, self.y[i, 0])
        best_classes = classes[: min(self.k, len(classes))]
        values, count = np.unique(best_classes, return_counts=True)
        return values[np.argmax(count)]

    def predict(self, X: np.array):
        if not self.fitted:
            raise Exception("KNeighborsClassifier: not fitted")
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
