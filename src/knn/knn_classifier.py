from classes.model import Model
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


def _default_distance_method(X1: np.ndarray, X2: np.ndarray) -> float:
    return float(np.linalg.norm(X1 - X2))


class KNeighborsClassifier(Model):
    def __init__(self, k=5, distance_method=_default_distance_method, n_jobs=-1, name="KNeighborsClassifier"):
        """
        Initialize the KNeighborsClassifier.

        Parameters
        ----------
        k : int, optional
            Number of neighbors to consider for classification (default is 5).
        distance_method : callable, optional
            Method to compute the distance between points (default is np.linalg.norm).
        n_jobs : int, optional
            The number of jobs to run in parallel during prediction (default is -1, using all available CPUs).
        name : str, optional
            The name given to the model
        """
        super().__init__(random_state=None, name=name)
        if k <= 0:
            raise ValueError("KNeighborsClassifier: k should be greater than 0")
        self.k = k
        self.distance_method = distance_method
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs


    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the KNeighborsClassifier.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.

        Returns
        -------
        self : KNeighborsClassifier
            Returns the instance itself.
        """
        super().fit(X, y)
        self.classes = np.unique(y)
        return self

    def _quick_ascending_insert(self, x: list, v: float) -> int:
        """
        Helper function for inserting a value into a sorted list in ascending order.

        Parameters
        ----------
        x : list
            Sorted list.
        v : float
            Value to insert.

        Returns
        -------
        int
            Index where the value was inserted.
        """
        i = len(x)
        x.append(v)
        while i > 0 and x[i - 1] > v:
            x[i], x[i - 1] = x[i - 1], x[i]
            i -= 1
        return i

    def _make_prediction(self, X: np.ndarray):
        """
        Make a prediction for a single data point.

        Parameters
        ----------
        X : np.ndarray
            Data point for prediction.

        Returns
        -------
        int
            Predicted class for the data point.
        """
        distances, classes = [], []
        for i in range(self.samples):
            j = self._quick_ascending_insert(
                distances, self.distance_method(self.X[i], X)
            )
            classes.insert(j, self.y[i])
        best_classes = classes[: min(self.k, len(classes))]
        values, count = np.unique(best_classes, return_counts=True)
        return values[np.argmax(count)]

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the KNeighborsClassifier.

        Parameters
        ----------
        X : np.ndarray
            Input data for making predictions.

        Returns
        -------
        np.ndarray
            Predicted class labels.
        """
        super().predict(X)
        
        samples = X.shape[0]
        result = np.zeros((samples))
        
        if not self.n_jobs:
            for i in range(samples):
                result[i] = self._make_prediction(X[i])
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self._make_prediction, X[i]): i for i in range(samples)
            }
            for future in as_completed(future_to_pred):
                result[future_to_pred[future]] = future.result()
        return result
