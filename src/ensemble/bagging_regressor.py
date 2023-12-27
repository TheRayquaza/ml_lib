from classes.model import Model
import numpy as np
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class BaggingRegressor(Model):
    def __init__(self, estimator, n_estimators=10, n_jobs=None, random_state=None):
        """
        Initialize the BaggingRegressor.

        Parameters
        ----------
        estimator : Model
            The base estimator from which the bagging ensemble is built.
        n_estimators : int, optional
            The number of base estimators in the ensemble (default is 10).
        n_jobs : int, optional
            The number of jobs to run in parallel (default is None). -1 means using all processors.
        random_state : int, optional
            Random state for reproducibility (default is None).
        """
        random.seed(random_state)
        self.estimator = estimator
        self.estimators = [estimator] * n_estimators  # Duplicate the estimator
        self.n_estimators = n_estimators
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self._fitted = False

        # Validate estimator and number of estimators
        if not self.estimator:
            raise ValueError(
                f"BaggingRegressor: the estimator {self.estimator} is invalid"
            )
        if self.n_estimators <= 0:
            raise ValueError(
                f"BaggingRegressor: Unable to create {self.n_estimators} estimators"
            )

    def __str__(self):
        return "BaggingRegressor"

    def _bagging(self) -> list:
        """
        Internal method to create the bagging samples.

        Returns
        -------
        list
            A list of tuples containing (model, sampled_X, sampled_y) for each estimator.
        """
        bag = []
        for model in self.estimators:
            size = random.randint(0, self.X.shape[0])
            indexes = np.random.permutation(size)
            bag.append((model, self.X[indexes], self.y[indexes]))
        return bag

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the bagging regressor.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        """
        self._fitted = True
        self.X = X
        self.y = y
        L = self._bagging()

        # Fit models either sequentially or in parallel
        if not self.n_jobs:
            for model, X, y in L:
                model.fit(X, y)
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {pool.submit(model.fit, X, y): model for model, X, y in L}
            for future in as_completed(future_to_pred):
                if future.result() == None:
                    raise Exception("BaggingRegressor: Something went wrong")
        return self

    def _make_prediction(self, model: Model, X: np.ndarray) -> list:
        """
        Internal method to make predictions using each base estimator.

        Parameters
        ----------
        model : Model
            The base estimator to use for predictions.
        X : np.ndarray
            The input data for prediction.

        Returns
        -------
        list
            The predictions made by the model.
        """
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the BaggingRegressor.

        Parameters
        ----------
        X : np.ndarray
            Input data for making predictions.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        if not self._fitted:
            raise Exception("BaggingRegressor: not fitted")

        result = np.zeros((X.shape[0], 1))

        # Make predictions for each instance in X
        if not self.n_jobs:
            for i in range(X.shape[0]):
                sub = []
                for model in self.estimators:
                    sub.append(self._make_prediction(model, X[i : i + 1]))
                result[i] = np.mean(np.array(sub))
        else:
            for i in range(X.shape[0]):
                pool = ThreadPoolExecutor(max_workers=self.n_jobs)
                future_to_pred = {
                    pool.submit(self._make_prediction, model, X[i : i + 1]): model
                    for model in self.estimators
                }
                sub = []
                for future in as_completed(future_to_pred):
                    sub.append(future.result())
                result[i] = np.mean(np.array(sub))
        return result
