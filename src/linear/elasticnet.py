from classes.model import Model
import numpy as np
from metrics.regression_metrics import mse


class ElasticNet(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-2,
        alpha=1e-1,
        rho=1,
        method="default",
        batch_size=32,
        random_state=None,
        verbose=False
    ):
        """
        Initialize the ElasticNet model.

        Parameters
        ----------
        learning_rate : float, optional
            The learning rate for gradient descent (default is 1e-4).
        decay : float, optional
            The decay rate for learning rate scheduling (default is 1e-2).
        alpha : float, optional
            The regularization strength (default is 0.1).
        rho : float, optional
            The elastic net mixing parameter (default is 1).
        method : str, optional
            The optimization method ("default", "stochastic" or "mini-batch") (default is "default").
        batch_size: int, optional
            The batch size, will be used if the method is "mini-batch" (default is 32)
        random_state : int, optional
            Seed for random number generation (default is None).
        verbose: bool, optional
            Whether the verbose should be activated
        """
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.decay = decay
        self.alpha = alpha
        self.rho = rho
        self.method = method
        self.verbose = verbose
        self.initial_learning_reate = learning_rate
        self.batch_size = batch_size
        self._fitted = False
        if not method in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"ElasticNet: Unknown method {method}")
        if self.batch_size <= 0 and method == "mini-batch":
            raise ValueError(f"ElasticNet: invalid batch size {batch_size}")

    def __str__(self) -> str:
        return "ElasticNet"

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """
        Compute the stochastic gradient for each sample and take the average

        Returns
        -------
            np.ndarray:
                The stochastic gradient
        """
        res = np.zeros((self.X.shape[1], 1))
        for m in range(self.X.shape[0]):
            res += (
                2
                * np.dot(self.X[m : m + 1].T,
                    np.dot(self.X[m : m + 1], self.weights) - self.y[m : m + 1]
                )
                + 2 * self.alpha * np.sign(self.weights)
                + self.alpha * self.rho * self.weights
            )
        return res / self.X.shape[0]

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
        res = np.zeros((self.X.shape[1], 1))
        for X_batch, y_batch in mini_batches:
            res += (
                2 * np.dot(X_batch.T, np.dot(X_batch, self.weights) - y_batch)
                + 2 * self.alpha * np.sign(self.weights)
                + self.alpha * self.rho * self.weights
            )
        return res / self.X.shape[0]

    def _compute_gradient(self) -> np.ndarray:
        """
        Compute the gradient for the entire dataset.

        Returns
        -------
            np.ndarray:
                The full gradient
        """
        return (
            2 * np.dot(self.X.T, np.dot(self.X, self.weights) - self.y)
            + 2 * self.alpha * self.rho * np.sign(self.weights)
            + self.alpha * self.rho * self.weights
        )

    def _train(self, epoch: int):
        """
        Train the ElasticNet model for one epoch.

        Parameters
        ----------
        epoch : int
            Current epoch number.
        """
        if not self._fitted:
            raise Exception("ElasticNet: not fitted")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                f"ElasticNet: invalid shapes {self.X.shape} and {self.y.shape}"
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
        Fit the ElasticNet model.

        Parameters
        ----------
        X : np.ndarray
            Input features.
        y : np.ndarray
            Target values.
        epochs : int, optional
            Number of training epochs (default is 100).
        tol : float, optional
            Tolerance to declare convergence (default is 1e-4).

        Returns
        -------
        self : ElasticNet
            Returns the instance itself.
        """
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.epochs = epochs
        self.gradients = np.zeros((X.shape[1], 1))
        self.tol = tol
        self._fitted = True
        self._finished = False
        for epoch in range(epochs):
            if self._finished:
                self._finished = False
                break
            self._train(epoch)
            if self.verbose:
                print(f"MSE epoch {epoch}: {mse(y, self.predict(X))}")
        return self

    def predict(self, X: np.ndarray):
        """
        Predict using the ElasticNet model.

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
            raise Exception("ElasticNet: not fitted")
        if X.shape[1] != self.features:
            raise Exception(
                f"ElasticNet: shape should be {self.features} and not {X.shape[1]}"
            )
        else:
            return np.dot(X, self.weights)
