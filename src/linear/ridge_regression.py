from classes.model import Model
import numpy as np
from metrics.regression_metrics import mse


class RidgeRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-2,
        alpha=0.1,
        method="default",
        batch_size=32,
        random_state=None,
        verbose=False,
    ):
        """
        Initializes the Ridge Regression model with specified parameters.

        Parameters
        ----------
            learning_rate (float, optional): The learning rate for gradient descent.
            decay (float, optional): Used to reduce the learning rate.
            alpha (float, optional): The regularization strength.
            method (str, optional): The training method.
            batch_size (int, optional): The size of oen batch in the case of mini-batch
            random_state (int, optional): The seed for random number generator.
            verbose (bool, optional): Whether the verbose should activated
        """
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.decay = decay
        self.method = method
        self.alpha = alpha
        self.batch_size = batch_size
        self.verbose = verbose
        self.initial_learning_rate = learning_rate
        self._fitted = False
        if method not in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"RidgeRegression: Unknown method {method}")

    def __str__(self) -> str:
        """String representation of the RidgeRegression class."""
        return "RidgeRegression"

    def _compute_mini_batch_gradient(self) -> np.ndarray:
        """Computes the gradient using mini-batch gradient descent."""
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
            gradients += (
                2 * X_batch.T.dot(X_batch.dot(self.weights) - y_batch)
                + 2 * self.alpha * self.weights
            )
        return gradients / len(mini_batches)

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """Computes the gradient using stochastic gradient descent."""
        gradients = np.zeros((self.X.shape[1], 1))
        for m in range(self.X.shape[0]):
            gradients += (
                2
                * np.dot(
                    self.X[m : m + 1].T,
                    np.dot(self.X[m : m + 1], self.weights) - self.y[m : m + 1],
                )
                + 2 * self.alpha * self.weights
            )
        return gradients / self.X.shape[0]

    def _compute_gradient(self) -> np.ndarray:
        """Computes the gradient for the entire dataset (batch gradient descent)."""
        return (
            2 * np.dot(self.X.T, np.dot(self.X, self.weights) - self.y)
            + 2 * self.alpha * self.weights
        )

    def _train(self, epoch: int):
        """
        Trains the model for one epoch using the specified optimization method.

        Parameters
        ----------
            epoch (int): The current epoch number.
        """
        if not self._fitted:
            raise ValueError("RidgeRegression: not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise ValueError(
                f"RidgeRegression: Invalid shapes having {self.X.shape} and {self.y.shape}"
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
            self.weights -= self.learning_rate / (epoch + 1) * self.gradients
            self.learning_rate = self.initial_learning_rate / (1 + self.decay * epoch)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, tol=1e-4):
        """
        Fits the Ridge Regression model to the data.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            epochs (int): The number of epochs for training.
            tol (float): The tolerance for the stop criterion.
        """
        self.X = X
        self.y = y
        self.id = np.identity(X.shape[1])
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.gradients = np.zeros((self.X.shape[1], 1))
        self.epochs = epochs
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

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for given input features.

        Parameters
        ----------
            X (np.ndarray): The input features.

        Returns:
            np.ndarray: Predicted target values.
        """
        if not self._fitted:
            raise ValueError("RidgeRegression: not fitted")
        if X.shape[1] != self.features:
            raise ValueError(
                "RidgeRegression: Shape should be", self.features, "and not", X.shape[1]
            )
        else:
            return np.dot(X, self.weights)
