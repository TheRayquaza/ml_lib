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
        verbose=False,
        name="ElasticNet"
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
        verbose : bool, optional
            Whether the verbose should be activated
        name : str, optional
            The name given to the model
        """
        super().__init__(random_state=random_state, name=name)

        self.learning_rate = learning_rate
        self.decay = decay
        self.alpha = alpha
        self.rho = rho
        self.method = method
        self.verbose = verbose
        self.initial_learning_reate = learning_rate
        self.batch_size = batch_size

        if not method in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"ElasticNet: Unknown method {method}")
        if self.batch_size <= 0 and method == "mini-batch":
            raise ValueError(f"ElasticNet: invalid batch size {batch_size}")
        if self.learning_rate <= 0:
            raise ValueError(f"ElasticNet: invalid learning rate {learning_rate}")
        if self.decay <= 0:
            raise ValueError(f"ElasticNet: invalid decay {decay}")
        if self.rho <= 0:
            raise ValueError(f"ElasticNet: invalid mxiing parameter {rho}")

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """
        Compute the stochastic gradient for each sample and take the average

        Returns
        -------
            np.ndarray:
                The stochastic gradient
        """
        res = np.zeros((self.features))
        for m in range(self.samples):
            res += (
                2
                * np.dot(self.X[m : m + 1].T,
                    np.dot(self.X[m : m + 1], self.weights) - self.y[m : m + 1]
                )
                + 2 * self.alpha * np.sign(self.weights)
                + self.alpha * self.rho * self.weights
            )
        return res / self.samples

    def _compute_mini_batch_gradient(self) -> np.ndarray:
        """
        Compute the mini batch gradient.

        Returns
        -------
            np.ndarray:
                The mini batch gradient
        """
        indices = np.arange(self.samples)
        np.random.shuffle(indices)
        X_shuffled, y_shuffled = self.X[indices], self.y[indices]
        mini_batches = [
            (
                X_shuffled[i : i + self.batch_size],
                y_shuffled[i : i + self.batch_size],
            )
            for i in range(0, self.samples, self.batch_size)
        ]
        res = np.zeros((self.features))
        for X_batch, y_batch in mini_batches:
            res += (
                2 * np.dot(X_batch.T, np.dot(X_batch, self.weights) - y_batch)
                + 2 * self.alpha * np.sign(self.weights)
                + self.alpha * self.rho * self.weights
            )
        return res / self.samples

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
        last_gradients = self.gradients.copy()
        if self.method == "default":
            self.gradients = self._compute_gradient()
        elif self.method == "stochastic":
            self.gradients = self._compute_stochastic_gradient()
        else:
            self.gradients = self._compute_mini_batch_gradient()
        if float(abs(np.sum(last_gradients - self.gradients))) <= self.tol:
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
        super().fit(X, y)
        self.weights = np.random.randn(self.features)
        self.epochs = epochs
        self.gradients = np.zeros((self.features))
        self.tol = tol
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
        super().predict(X)
        return np.dot(X, self.weights)
