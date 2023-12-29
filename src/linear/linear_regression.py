from classes.model import Model
import numpy as np
from metrics.regression_metrics import mse


class LinearRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-2,
        method="default",
        batch_size=32,
        random_state=None,
        verbose=False,
        n_weights=None,
    ):
        """
        Initializes the Linear Regression model with specified parameters.

        Parameters
        ----------
            learning_rate (float): The initial learning rate for gradient descent.
            decay (float): The rate of decay for the learning rate.
            method (str): The method for optimization.
            random_state (int, optional): The seed for random number generator.
            verbose (bool, optional): Whether the verbse mode should be actiavted.
            n_weights (int, optional): If given, give a spefici shape to the weight
        """
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.decay = decay
        self.method = method
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_weights = n_weights
        self.initial_learning_reate = learning_rate
        self._fitted = False
        if not method in ["default", "stochastic", "analytic", "mini-batch"]:
            raise Exception("LinearRegression: Unknown method " + method)

    def __str__(self) -> str:
        return "LinearRegression"

    def _compute_mini_batch_gradient(self) -> np.ndarray:
        """
        Computes the gradient using mini batch

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
            gradients += 2 * np.dot(X_batch.T, np.dot(X_batch, self.weights) - y_batch)
        return gradients / len(mini_batches)

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """
        Computes the gradient using stochastic gradient descent.

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
            )
        return gradients / self.X.shape[0]

    def _compute_gradient(self) -> np.ndarray:
        """
        Computes the gradient for the entire dataset (batch gradient descent).

        Returns
        -------
            np.ndarray:
                The Full gradient
        """
        return 2 * np.dot(self.X.T, np.dot(self.X, self.weights) - self.y)

    def _train(self, epoch: int):
        """Train for a given epoch

        Parameters
        ----------
            epoch (int): epoch number
        """
        if not self._fitted:
            raise Exception("LinearRegression: not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                f"LinearRegression: invalid shapes having {self.X.shape} and {self.y.shape}"
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

    def _train_analytic(self):
        """Method for findind the weights using pseudo inverse"""
        if not self._fitted:
            raise Exception("LinearRegression: not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                f"LinearRegression: Invalid shapes {self.X.shape} and {self.y.shape}"
            )
        else:
            self.weights = np.linalg.pinv(self.X).dot(self.y)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, tol=1e-4):
        """
        Fits the Linear Regression model to the data.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            epochs (int): The number of epochs for training.
            tol (float): The tolerance for the stop criterion.

        Returns
        -------
            self: Returns an instance of self.
        """
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.bias = 0
        self.epochs = epochs
        self.tol = tol
        self.gradients = np.zeros((X.shape[1], 1))
        self._finished = False
        self._fitted = True
        if self.method == "analytic":
            self._train_analytic()
        else:
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

        Returns
        -------
            np.ndarray: Predicted target values.
        """
        if not self._fitted:
            raise ValueError("LinearRegression: not fitted with data")
        elif X.shape[1] != self.features:
            raise Exception(
                f"LinearRegression: shape should {self.features} and not {X.shape[1]}"
            )
        else:
            return np.dot(X, self.weights)
