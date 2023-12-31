from classes.model import Model
from classes.transformer import Transformer
import numpy as np


class Pipeline(Model):
    def __init__(self, steps=[], name="Pipeline", random_state=None):
        """
        Initialize the Pipeline.

        Parameters
        ----------
        steps : list
            List of (name, transform) tuples.
        """
        super().__init__(random_state=random_state, name=name)
        self.steps = steps
        self._model = None
        if len(steps) != 0:
            self._model = self.steps[-1][-1]

    @property
    def model(self):
        """
        Get the final model in the pipeline.

        Returns
        -------
        Model
            The final model in the pipeline.
        """
        return self._model

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit all the transforms one by one and then fit the final estimator.

        Parameters
        ----------
        X : np.ndarray
            Training data.
        y : np.ndarray
            Target values.

        Raises
        ------
        Exception
            If any element in the pipeline is neither a Model nor a Transformer.
        """
        super().fit(X, y)
        for _, element in self.steps:
            if issubclass(type(element), Model):
                element.fit(X, y)
            elif issubclass(type(element), Transformer):
                X = element.fit_transform(X)
            else:
                raise Exception(f"Pipeline: invalid argument {element}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Apply transforms to the data, and predict with the final estimator.

        Parameters
        ----------
        X : np.ndarray
            Data to predict on.

        Returns
        -------
        np.ndarray
            Predicted values.

        Raises
        ------
        Exception
            If any element in the pipeline is neither a Model nor a Transformer.
        """
        super().predict(X)
        for _, element in self.steps:
            if issubclass(type(element), Model):
                X = element.predict(X)
            elif issubclass(type(element), Transformer):
                X = element.transform(X)
            else:
                raise Exception(f"Pipeline: invalid argument {element}")
        return X
