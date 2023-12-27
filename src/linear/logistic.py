from classes.model import Model
import numpy as np
from scipy.special import expit


class LogisticRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-3,
        method="default",
        threshold=0.5,
        random_state=None,
    ):
        """
        Initializes the Logistic Regression model with specified parameters.

        Parameters
        ----------
            learning_rate (float): The initial learning rate for gradient descent.
            decay (float): The rate of decay for the learning rate.
            method (str): The method for training.
            threshold (float): Threshold for converting predicted probabilities to class labels.
            random_state (int, optional): The seed for random number generator.
        """
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.initial_learning_reate = learning_rate
        self.decay = decay
        self.method = method
        self.threshold = threshold
        self._fitted = False
        if not method in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"LogisticRegression: Unknown method {method}")

    def __str__(self) -> str:
        return "LogisticRegression"

    def _compute_mini_batch_gradient(self) -> np.ndarray:
        """Computes the gradient using mini batch

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
        res = np.zeros((self.X.shape[0], 1))
        for X_batch, y_batch in mini_batches:
            res += 2 * X_batch.T.dot(X_batch.dot(self.weights) - y_batch)
        return res / self.X.shape[0]

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """
        Computes the gradient using stochastic gradient descent.

        Returns
        -------
            np.ndarray:
                The stochastic gradient
        """
        res = np.zeros((self.X.shape[0], 1))
        for m in range(self.X.shape[0]):
            res += 2 * self.X[m : m + 1].T.dot(
                self.X[m : m + 1].dot(self.weights) - self.y[m : m + 1]
            )
        return res / self.X.shape[0]

    def _compute_gradient(self) -> np.ndarray:
        """
        Computes the gradient for the entire dataset (batch gradient descent).

        Returns
        -------
            np.ndarray:
                The full gradient
        """
        return 2 * self.X.T.dot(self.X.dot(self.weights) - self.y)

    def _train(self, epoch: int):
        """
        Trains the model for one epoch using the specified optimization method.

        Parameters
        ----------
            epoch (int): The current epoch number.
        """
        if self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise ValueError(
                f"LogisticRegression: Invalid shapes X={self.X.shape} while y={self.y.shape}"
            )
        else:
            last_gradients = self.gradients.copy()
            if self.method == "default":
                self.gradients = self._compute_gradient()
            elif self.method == "stochastic":
                self.gradients = self._compute_stochastic_gradient()
            else:
                self.gradients = self._compute_mini_batch_gradient()
            if abs(np.sum(last_gradients) - np.sum(self.gradients)) <= self.tol:
                self._finished = True
            self.weights -= self.learning_rate * self.gradients
            self.learning_rate = self.initial_learning_reate / (1 + self.decay * epoch)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, tol=1e-4):
        """
        Fits the Logistic Regression model to the data.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target binary labels.
            epochs (int): The number of epochs for training.
            tol (float): The tolerance for the stop criterion.

        Returns
        -------
            LogisticRegression: Returns an instance of self.
        """
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.fitted = True
        self.epochs = epochs
        self.tol = tol
        self.gradients = np.zeros((X.shape[1], 1))
        self._finished = False
        self._fitted = True
        for epoch in range(epochs):
            if self._finished:
                self._finished = False
                break
            self._train(epoch)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the class labels for given input features.

        Parameters
        ----------
            X (np.ndarray): The input features.

        Returns
        -------
            np.ndarray: Predicted class labels.
        """
        if not self._fitted:
            raise Exception("LogisticRegression: model not fitted")
        if X.shape[1] != self.features:
            raise Exception(
                f"LogisticRegression: shape should be {self.features} and not {X.shape[1]}"
            )
        return expit(np.dot(X, self.weights)) > self.threshold
