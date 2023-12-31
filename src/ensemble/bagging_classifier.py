from classes.model import Model
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class BaggingClassifier(Model):
    def __init__(
        self,
        estimator,
        n_estimators=10,
        n_jobs=None,
        random_state=None,
        name="BaggingClassifier"
    ):
        """
        Initialize the BaggingClassifier.

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
        name: str, optional
            The name given to the model
        """
        super().__init__(random_state=random_state, name=name)
        self.estimator = estimator
        self.estimators = [estimator] * n_estimators  # Duplicate the estimator
        self.n_estimators = n_estimators
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        # Validate estimator and number of estimators
        if not self.estimator:
            raise ValueError(
                f"BaggingClassifier: the estimator {self.estimator} is invalid"
            )
        if self.n_estimators <= 0:
            raise ValueError(
                f"BaggingClassifier: Unable to create {self.n_estimators} estimators"
            )

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
            indexes = np.random.randint(0, self.samples, self.samples)
            bag.append((model, self.X[indexes], self.y[indexes]))
        return bag

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the bagging classifier.

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
            future_to_pred = {pool.submit(model.fit, X, y): model for model, X, y in L}
            for future in as_completed(future_to_pred):
                if future.result() == None:
                    raise Exception("BaggingClassifier: Something went wrong")
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
        Predict using the BaggingClassifier.

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

        samples = X.shape[0]
        result = np.zeros((X.shape[0]))

        if not self.n_jobs:
            for i in range(samples):
                sub = []
                for model in self.estimators:
                    sub.append(self._make_prediction(model, X[i : i + 1]))
                uniques, counts = np.unique(np.array(sub), return_counts=True)
                result[i] = uniques[np.argmax(counts)]
        else:
            for i in range(samples):
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
