from classes.model import Model
import numpy as np


class LassoRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-2,
        alpha=0.1,
        method="default",
        random_state=None,
    ):
        """
        Initialize the LassoRegression model.

        Parameters
        ----------
        learning_rate : float
            The learning rate for gradient descent.
        decay : float
            The decay rate for learning rate scheduling.
        alpha : float
            The regularization strength.
        method : str
            The optimization method ("default", "stochastic" or "mini-batch").
        random_state : int or None
            Seed for the random number generator.
        """
        np.random.seed(random_state)
        self.learning_rate = learning_rate
        self.decay = decay
        self.alpha = alpha
        self.method = method
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
        res = np.zeros((self.X.shape[0], 1))
        for X_batch, y_batch in mini_batches:
            res += 2 * X_batch.T.dot(
                X_batch.dot(self.weights) - y_batch
            ) + 2 * self.alpha * np.sign(self.weights)
        return res / self.X.shape[0]

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """
        Compute the stochastic gradient for each sample.

        Returns
        -------
            np.ndarray:
                The stochastic gradient
        """
        res = np.zeros((self.X.shape[0], 1))
        for m in range(self.X.shape[0]):
            res += 2 * self.X[m : m + 1].T.dot(
                self.X[m : m + 1].dot(self.weights) - self.y[m : m + 1]
            ) + 2 * self.alpha * np.sign(self.weights)
        return res / self.X.shape[0]

    def _compute_gradient(self) -> np.ndarray:
        """
        Compute the gradient for the entire dataset.

        Returns
        -------
            np.ndarray:
                The full gradient
        """
        return 2 * self.X.T.dot(
            self.X.dot(self.weights) - self.y
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
            if abs(np.sum(last_gradients) - np.sum(self.gradients)) <= self.tol:
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
        if epochs < 0:
            raise ValueError(f"LassoRegression: invalid number of epochs {epochs}")
        self._fitted = True
        for epoch in range(epochs):
            if self._finished:
                break
            self._train(epoch)
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
