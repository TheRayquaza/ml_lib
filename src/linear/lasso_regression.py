from classes.model import Model
import numpy as np
from metrics.regression_metrics import mse


class LassoRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-2,
        alpha=1e-1,
        method="default",
        batch_size=32,
        random_state=None,
        verbose=False,
    ):
        """
        Initialize the LassoRegression model.

        Parameters
        ----------
        learning_rate (float, optional): The learning rate for gradient descent.
        decay (float, optional): The decay rate for learning rate scheduling.
        alpha (float, optional): The regularization strength.
        method (str, optional): The optimization method ("default", "stochastic" or "mini-batch").
        random_state (int, optional): Seed for the random number generator.
        verbose (bool): Whether the verbose mode should be activated
        """
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.decay = decay
        self.alpha = alpha
        self.method = method
        self.batch_size = batch_size
        self.verbose = verbose
        self.initial_learning_reate = learning_rate
        self._fitted = False
        if not method in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"LassoRegression: Unknown method {method}")

    def __str__(self) -> str:
        return "LassoRegression"

    def _compute_mini_batch_gradient(self) -> np.ndarray:
        """
        Compute the mini batch gradient.

        Returns
        -------
            np.ndarray:
                The mini batch gradient
        """
        indices = np.arange(self.X.shape[0])
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = self.X[indices], self.y[indices]
        mini_batches = [
            (
                X_shuffled[i : i + self.batch_size],
                y_shuffled[i : i + self.batch_size],
            )
            for i in range(0, self.X.shape[0], self.batch_size)
        ]
        gradients = np.zeros((self.X.shape[1], 1))
        for X_batch, y_batch in mini_batches:
            gradients += 2 * np.dot(
                X_batch.T, np.dot(X_batch, self.weights - y_batch)
            ) + 2 * self.alpha * np.sign(self.weights)
        return gradients / len(mini_batches)

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """
        Compute the stochastic gradient for each sample.

        Returns
        -------
            np.ndarray:
                The stochastic gradient
        """
        gradients = np.zeros((self.X.shape[1], 1))
        for m in range(self.X.shape[0]):
            gradients += 2 * np.dot(
                self.X[m : m + 1].T,
                np.dot(self.X[m : m + 1], self.weights) - self.y[m : m + 1],
            ) + 2 * self.alpha * np.sign(self.weights)
        return gradients / self.X.shape[0]

    def _compute_gradient(self) -> np.ndarray:
        """
        Compute the gradient for the entire dataset.

        Returns
        -------
            np.ndarray:
                The full gradient
        """
        return 2 * np.dot(
            self.X.T, np.dot(self.X, self.weights) - self.y
        ) + 2 * self.alpha * np.sign(self.weights)

    def _train(self, epoch: int):
        """
        Train the model for one epoch.

        Parameters
        ----------
        epoch : int
            The current epoch number.
        """
        if not self._fitted:
            raise Exception("LassoRegression: not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                f"LassoRegression: Invalid shapes having {self.X.shape} and {self.y.shape}"
            )
        else:
            last_gradients = self.gradients.copy()
            if self.method == "default":
                self.gradients = self._compute_gradient()
            elif self.method == "stochastic":
                self.gradients = self._compute_stochastic_gradient()
            else:
                self.gradients = self._compute_mini_batch_gradient()
            if abs(np.sum(last_gradients - self.gradients)) <= self.tol:
                self._finished = True
            self.weights -= self.learning_rate * self.gradients
            self.learning_rate = self.initial_learning_reate / (1 + self.decay * epoch)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, tol=1e-4):
        """
        Fit the LassoRegression model to the data.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        epochs : int
            Number of training epochs.
        tol : float
            Tolerance for convergence.

        Returns
        -------
        self
            The instance itself
        """
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.epochs = epochs
        self.gradients = np.zeros((X.shape[1], 1))
        self.tol = tol
        self._finished = False
        self._fitted = True
        if epochs < 0:
            raise ValueError(f"LassoRegression: invalid number of epochs {epochs}")
        for epoch in range(epochs):
            if self._finished:
                break
            self._train(epoch)
            if self.verbose:
                print(f"MSE epoch {epoch}: {mse(y, self.predict(X))}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the LassoRegression model.

        Parameters
        ----------
        X : np.ndarray
            Input data for predictions.

        Returns
        -------
        np.ndarray:
            The dot product between the input and weights
        """
        if not self._fitted:
            raise Exception(f"LassoRegression: not fitted with data")
        if X.shape[1] != self.features:
            raise ValueError(
                f"LassoRegression: shape should be {self.features} and not {X.shape[1]}"
            )
        return np.dot(X, self.weights)
