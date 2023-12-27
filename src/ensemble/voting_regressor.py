from classes.model import Model
import numpy as np
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
        """
        Initialize the VotingRegressor.

        Parameters
        ----------
        estimators : list
            A list of tuples containing (name, model) where name is a string identifier and model is an instance of Model.
        n_jobs : int, optional
            The number of jobs to run in parallel during fitting and prediction (default is None).
        bootstrap : bool, optional
            Whether to use bootstrap samples for training each estimator (default is True).
        random_state : int, optional
            Random state for reproducibility (default is None).
        """
        np.random.seed(random_state)
        self.estimators = estimators
        self.n_estimators = len(self.estimators)
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self._fitted = False
        self.bootstrap = bootstrap

        # Validate the number of estimators
        if self.n_estimators <= 0:
            raise ValueError(
                f"VotingRegressor: Unable to create {self.n_estimators} estimators"
            )

    def __str__(self):
        return "VotingRegressor"

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the VotingRegressor.

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

        # Fit models either sequentially or in parallel
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

    def _make_prediction(self, model: Model, X: np.ndarray) -> list:
        """
        Internal method to make predictions using each estimator.

        Parameters
        ----------
        model : Model
            The estimator to use for predictions.
        X : np.ndarray
            The input data for prediction.

        Returns
        -------
        list
            The predictions made by the model.
        """
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the VotingRegressor.

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
            raise Exception("VotingRegressor: not fitted")
        result = np.zeros((X.shape[0], 1))
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
