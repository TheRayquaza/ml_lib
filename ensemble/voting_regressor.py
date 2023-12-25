from classes.model import Model
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class VotingRegressor(Model):
    def __init__(
        self,
        estimators,
        n_jobs=None,
        bootstrap=True,
        random_state=None,
    ):
        random.seed(random_state)
        self.estimators = estimators
        self.n_estimators = len(self.estimators)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.__fitted = False
        self.bootstrap = bootstrap
        if self.n_estimators <= 0:
            raise ValueError(
                f"VotingRegressor: Unable to create {self.n_estimators} estimators"
            )

    def __str__(self):
        return "VotingRegressor"

    def fit(self, X: np.array, y: np.array):
        self.__fitted = True
        self.X = X
        self.y = y
        if not self.n_jobs:
            for name, model in self.estimators:
                model.fit(X, y)
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(model.fit, X, y): model for name, model in self.estimators
            }
            for future in as_completed(future_to_pred):
                if future.result() == None:
                    raise Exception("VotingRegressor: Something went wrong")
        return self

    def __make_prediction(self, model: Model, X: np.array) -> list:
        result = []
        if not self.n_jobs:
            for name, model in self.estimators:
                result.append(model.predict(X))
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(model.predict, X): model for name, model in self.estimators
            }
            for future in as_completed(future_to_pred):
                result.append(future.result())
        return result

    def predict(self, X: np.array) -> np.array:
        if not self.__fitted:
            raise Exception("VotingRegressor: not fitted")
        result = np.zeros((X.shape[0], 1))
        if not self.n_jobs:
            for i in range(X.shape[0]):
                sub = []
                for model in self.estimators:
                    sub.append(self.__make_prediction(model, X[i : i + 1]))
                result[i] = np.mean(np.array(sub))
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
                result[i] = np.mean(np.array(sub))
        return result
