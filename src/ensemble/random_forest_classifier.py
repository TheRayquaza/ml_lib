from classes.model import Model
from tree.decision_tree_classifier import DecisionTreeClassifier
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class RandomForestClassifier(Model):
    def __init__(
        self,
        n_estimators=10,
        max_depth=None,
        method="gini",
        n_jobs=None,
        bootstrap=True,
        random_state=None,
    ):
        """
        Initialize the RandomForestClassifier.

        Parameters
        ----------
        n_estimators : int, optional
            The number of decision tree estimators in the ensemble (default is 10).
        max_depth : int, optional
            The maximum depth of each decision tree (default is None, meaning unlimited depth).
        method : str, optional
            The method used to split nodes in each decision tree, either "gini" or "entropy" (default is "gini").
        n_jobs : int, optional
            The number of jobs to run in parallel during fitting and prediction (default is None).
        bootstrap : bool, optional
            Whether to use bootstrap samples for training each decision tree (default is True).
        random_state : int, optional
            Random state for reproducibility (default is None).
        """
        np.random.seed(random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.method = method
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self.bootstrap = bootstrap

        # Validate the number of estimators
        if n_estimators <= 0:
            raise ValueError(
                f"RandomForestClassifier: Unable to create {n_estimators} estimators"
            )

        # Validate the maximum depth
        self.max_depth = max_depth
        if max_depth and max_depth <= 0:
            raise ValueError(
                f"RandomForestClassifier: Unable to use max_depth {max_depth}"
            )

        # Initialize decision tree estimators
        self.estimators = [
            DecisionTreeClassifier(max_depth=max_depth, method=method, n_jobs=n_jobs)
        ] * n_estimators

        self._fitted = False

    def __str__(self):
        return "RandomForestClassifier"

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
                size = np.random.randint(0, self.X.shape[0])
                indexes = np.random.permutation(size)
                L.append((model, self.X[indexes], self.y[indexes]))
        return L

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the RandomForestClassifier.

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
            future_to_pred = {
                pool.submit(model.fit, X, y): model for model, X, y in self._bagging()
            }
            for future in as_completed(future_to_pred):
                if future.result() == None:
                    raise Exception("RandomForestClassifier: Something went wrong")
        return self

    def _make_prediction(self, model: DecisionTreeClassifier, X: np.ndarray):
        """
        Internal method to make predictions using each decision tree estimator.

        Parameters
        ----------
        model : DecisionTreeClassifier
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
        Predict using the RandomForestClassifier.

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
            raise Exception("RandomForestClassifier: not fitted")
        result = np.zeros((X.shape[0], 1))
        if not self.n_jobs:
            for i in range(X.shape[0]):
                sub = []
                for model in self.estimators:
                    sub.append(self._make_prediction(model, X[i : i + 1]))
                uniques, counts = np.unique(np.array(sub), return_counts=True)
                result[i] = uniques[np.argmax(counts)]
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
                uniques, counts = np.unique(np.array(sub), return_counts=True)
                result[i] = uniques[np.argmax(counts)]
        return result