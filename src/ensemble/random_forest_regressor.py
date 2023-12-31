from classes.model import Model
from tree.decision_tree_regressor import DecisionTreeRegressor
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class RandomForestRegressor(Model):
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        method="mse",
        n_jobs=None,
        bootstrap=True,
        random_state=None,
        name="RandomForestRegressor"
    ):
        """
        Initialize the RandomForestRegressor.

        Parameters
        ----------
        n_estimators : int, optional
            The number of decision tree estimators in the ensemble (default is 10).
        max_depth : int, optional
            The maximum depth of each decision tree (default is None, meaning unlimited depth).
        method : str, optional
            The method used to split nodes in each decision tree, either "mse" or "mae" (default is "mse").
        n_jobs : int, optional
            The number of jobs to run in parallel during fitting and prediction (default is None).
        bootstrap : bool, optional
            Whether to use bootstrap samples for training each decision tree (default is True).
        random_state : int, optional
            Random state for reproducibility (default is None).
        name : str, optional
            The name given to the model.
        """
        super().__init__(random_state=random_state, name=name)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.method = method
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.bootstrap = bootstrap

        # Validate the number of estimators
        if n_estimators <= 0:
            raise ValueError(
                f"RandomForestRegressor: Unable to create {n_estimators} estimators"
            )

        # Validate the maximum depth
        self.max_depth = max_depth
        if max_depth and max_depth <= 0:
            raise ValueError(
                f"RandomForestRegressor: Unable to use max_depth {max_depth}"
            )

        # Initialize decision tree estimators
        self.estimators = [
            DecisionTreeRegressor(max_depth=max_depth, method=method, n_jobs=n_jobs)
        ] * n_estimators

    def _bagging(self):
        """
        Internal method to create the bagging samples.

        Returns
        -------
        list
            A list of tuples containing (model, sampled_X, sampled_y) for each estimator.
        """
        L = []
        if not self.bootstrap:
            for model in self.estimators:
                L.append((model, self.X, self.y))
        else:
            for model in self.estimators:
                indexes = np.random.randint(0, self.X.shape[0], self.X.shape[0])
                L.append((model, self.X[indexes], self.y[indexes]))
        return L

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the RandomForestRegressor.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.
        """
        super().fit(X, y)
        L = self._bagging()

        # Fit models either sequentially or in parallel
        if not self.n_jobs:
            for model, X, y in L:
                model.fit(X, y)
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(model.fit, X, y): model for model, X, y in self._bagging()
            }
            for future in as_completed(future_to_pred):
                if future.result() == None:
                    raise Exception("RandomForestRegressor: Something went wrong")
        return self

    def _make_prediction(self, model: DecisionTreeRegressor, X: np.ndarray):
        """
        Internal method to make predictions using each decision tree estimator.

        Parameters
        ----------
        model : DecisionTreeRegressor
            The decision tree estimator to use for predictions.
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
        Predict using the RandomForestRegressor.

        Parameters
        ----------
        X : np.ndarray
            Input data for making predictions.

        Returns
        -------
        np.ndarray
            Predicted values.
        """
        super().predict(X)
        result = np.zeros((X.shape[0]))
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
