from classes.model import Model
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class BaggingClassifier(Model):
    def __init__(
        self,
        estimator,
        n_estimators=10,
        n_jobs=None,
        random_state=None,
    ):
        random.seed(random_state)
        self.estimator = estimator
        self.estimators = [estimator] * n_estimators
        self.n_estimators = n_estimators
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.__fitted = False
        if not self.estimator:
            raise ValueError(
                f"BaggingClassifier: the estimator {self.estimator} is invalid"
            )
        if self.n_estimators <= 0:
            raise ValueError(
                f"BaggingClassifier: Unable to create {self.n_estimators} estimators"
            )

    def __str__(self):
        return "BaggingClassifier"

    def __bagging(self) -> list:
        bag = []
        for model in self.estimators:
            size = random.randint(0, self.X.shape[0])
            indexes = np.random.permutation(size)
            bag.append((model, self.X[indexes], self.y[indexes]))
        return bag

    def fit(self, X: np.array, y: np.array):
        self.__fitted = True
        self.X = X
        self.y = y
        L = self.__bagging()
        if not self.n_jobs:
            for model, X, y in L:
                model.fit(X, y)
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {pool.submit(model.fit, X, y): model for model, X, y in L}
            for future in as_completed(future_to_pred):
                if future.result() == None:
                    raise Exception("BaggingClassifier: Something went wrong")
        return self

    def __make_prediction(self, model: Model, X: np.array) -> list:
        result = []
        if not self.n_jobs:
            for model in self.estimators:
                result.append(model.predict(X))
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(model.predict, X): model for model in self.estimators
            }
            for future in as_completed(future_to_pred):
                result.append(future.result())
        return result

    def predict(self, X: np.array) -> np.array:
        if not self.__fitted:
            raise Exception("BaggingClassifier: not fitted")
        result = np.zeros((X.shape[0], 1))
        if not self.n_jobs:
            for i in range(X.shape[0]):
                sub = []
                for model in self.estimators:
                    sub.append(self.__make_prediction(model, X[i : i + 1]))
                uniques, counts = np.unique(np.array(sub), return_counts=True)
                result[i] = uniques[np.argmax(counts)]
        else:
            for i in range(X.shape[0]):
                pool = ThreadPoolExecutor(max_workers=self.n_jobs)
                future_to_pred = {
                    pool.submit(self.__make_prediction, model, X[i : i + 1]): model
                    for model in self.estimators
                }
                sub = []
                for future in as_completed(future_to_pred):
                    sub.append(future.result())
                uniques, counts = np.unique(np.array(sub), return_counts=True)
                result[i] = uniques[np.argmax(counts)]
        return result
