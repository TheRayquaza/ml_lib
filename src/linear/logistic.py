from classes.model import Model
import numpy as np
from scipy.special import expit
from metrics.regression_metrics import mse

class LogisticRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-3,
        method="default",
        threshold=0.5,
        batch_size=32,
        random_state=None,
        verbose=False,
        name="LogisticRegression"
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
            verbose (bool, optional): Whether the verbose mode should be activated.
            name (str, optional): The name given to the model
        """
        super().__init__(random_state=random_state, name=name)

        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.decay = decay
        self.method = method
        self.threshold = threshold
        self.batch_size = batch_size
        self.verbose = verbose

        if not method in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"LogisticRegression: Unknown method {method}")
        if self.batch_size <= 0 and method == "mini-batch":
            raise ValueError(f"LogisticRegression: invalid batch size {batch_size}")
        if self.decay <= 0:
            raise ValueError(f"LogisticRegression: invalid decay {decay}")
        if self.learning_rate <= 0:
            raise ValueError(f"LogisticRegression: invalid learning rate {learning_rate}")
        if self.threshold < 0 or self.threshold > 1:
            raise ValueError(f"LogisticRegression: invalid threshold {threshold}")

    def _compute_mini_batch_gradient(self) -> np.ndarray:
        """Computes the gradient using mini batch

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
            res += 2 * np.dot(X_batch.T, np.dot(X_batch, self.weights) - y_batch)
        return res / len(mini_batches)

    def _compute_stochastic_gradient(self) -> np.ndarray:
        """
        Computes the gradient using stochastic gradient descent.

        Returns
        -------
            np.ndarray:
                The stochastic gradient
        """
        res = np.zeros((self.features))
        for m in range(self.samples):
            res += 2 * np.dot(self.X[m : m + 1].T,
                np.dot(self.X[m : m + 1], self.weights) - self.y[m : m + 1]
            )
        return res / self.samples

    def _compute_gradient(self) -> np.ndarray:
        """
        Computes the gradient for the entire dataset (batch gradient descent).

        Returns
        -------
            np.ndarray:
                The full gradient
        """
        return 2 * np.dot(self.X.T, np.dot(self.X, self.weights) - self.y)

    def _train(self, epoch: int):
        """
        Trains the model for one epoch using the specified optimization method.

        Parameters
        ----------
            epoch (int): The current epoch number.
        """
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
        self.learning_rate = self.initial_learning_rate / (1 + self.decay * epoch)

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
        super().fit(X, y)
        self.weights = np.random.randn(self.features)
        self.epochs = epochs
        self.tol = tol
        self.gradients = np.zeros((self.features))
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
        Predicts the class labels for given input features.

        Parameters
        ----------
            X (np.ndarray): The input features.

        Returns
        -------
            np.ndarray: Predicted class labels.
        """
        super().predict(X)
        return expit(np.dot(X, self.weights)) > self.threshold
