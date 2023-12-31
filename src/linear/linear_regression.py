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
        name="LinearRegression"
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
            name (str, optional): The name given to the model
        """
        super().__init__(random_state=random_state, name=name)

        self.learning_rate = learning_rate
        self.decay = decay
        self.method = method
        self.batch_size = batch_size
        self.verbose = verbose
        self.n_weights = n_weights
        self.initial_learning_rate = learning_rate

        if not method in ["default", "stochastic", "mini-batch"]:
            raise Exception("LinearRegression: Unknown method " + method)
        if self.batch_size <= 0 and method == "mini-batch":
            raise ValueError(f"LinearRegression: invalid batch size {batch_size}")
        if self.decay <= 0:
            raise ValueError(f"LinearRegression: invalid decay {decay}")
        if self.learning_rate <= 0:
            raise ValueError(f"LinearRegression: invalid learning rate {learning_rate}")

    def _compute_mini_batch_gradient(self) -> np.ndarray:
        """
        Computes the gradient using mini batch

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
        gradients = np.zeros((self.features))
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
        gradients = np.zeros((self.features))
        for m in range(self.samples):
            gradients += 2 * np.dot(
                self.X[m : m + 1].T,
                np.dot(self.X[m : m + 1], self.weights) - self.y[m : m + 1],
            )
        return gradients / self.samples

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
        self.learning_rate = self.initial_learning_rate / (1 + self.decay * epoch)

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
        super().fit(X, y)
        self.weights = np.random.randn(self.features)
        self.bias = 0
        self.epochs = epochs
        self.tol = tol
        self.gradients = np.zeros((self.featues))
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

        Returns
        -------
            np.ndarray: Predicted target values.
        """
        super().predict(X)
        return np.dot(X, self.weights)
