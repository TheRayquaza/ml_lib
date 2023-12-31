from classes.model import Model
import numpy as np
from scipy.special import softmax


class SoftmaxRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-3,
        method="default",
        threshold=0.5,
        batch_size=32,
        random_state=None,
        name="SoftmaxRegression"
    ):
        super().__init__(random_state=random_state, name=name)
        self.learning_rate = learning_rate
        self.initial_learning_reate = learning_rate
        self.decay = decay
        self.method = method
        self.threshold = threshold
        self.batch_size = batch_size

        if not method in ["default", "stochastic", "mini-batch"]:
            raise Exception(f"SoftmaxRegression: Unknown method {method}")

    def __compute_stochastic_gradient(self) -> np.ndarray:
        res = np.zeros((self.samples))
        for m in range(self.samples):
            res += 2 * self.X[m : m + 1].T.dot(
                self.X[m : m + 1].dot(self.weights) - self.y[m : m + 1]
            )
        return res / self.samples

    def __compute_gradient(self) -> np.ndarray:
        return 2 * self.X.T.dot(self.X.dot(self.weights) - self.y)

    def __train(self, epoch):
        last_gradients = self.gradients
        if self.method == "default":
            self.__compute_gradient()
        elif self.method == "stochastic":
            self.__compute_stochastic_gradient()
        if float(abs(np.sum(last_gradients - self.gradients))) <= self.tol:
            self._finished = True
        self.weights -= self.learning_rate * self.gradients
        self.learning_rate = self.initial_learning_reate / (1 + self.decay * epoch)

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, tol=1e-4):
        super().fit(X, y)
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.fitted = True
        self.epochs = epochs
        self.tol = tol
        self.gradients = np.zeros((X.shape[1], 1))
        self._finished = False
        for epoch in range(epochs):
            if self._finished:
                self._finished = False
                break
            self.__train(epoch)
        return self

    def predict(self, X: np.ndarray):
        super().predict(X)
        return np.argmax(softmax(np.dot(X, self.weights)))
