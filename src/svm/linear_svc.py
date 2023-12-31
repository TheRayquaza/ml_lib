from classes.model import Model
import numpy as np
from metrics.classification_metrics import accuracy_score

class LinearSVC(Model):
    def __init__(
        self, learning_rate=1e-3, decay=1e-4, regularization=1e-2, method="default", verbose=False, random_state=None, name="LinearSVC"
    ):
        super().__init__(random_state=random_state, name=name)

        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.decay = decay
        self.regularization = regularization
        self.method = method
        self.verbose = verbose

        if learning_rate <= 0:
            raise ValueError(f"LinearSVC: learning rate negative {learning_rate}")
        if decay <= 0:
            raise ValueError(f"LinearSVC: decay negative {decay}")
        if regularization <= 0:
            raise ValueError(f"LienarSVC: regularization negative {regularization}")
        if not method in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"LinearSVC: unknown method {method}")
        if method == "mini-batch" or method == "stochastic":
            raise NotImplementedError("LinearSVC: not implemented yet")

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, tol=1e-3):
        super().fit(X, y)

        self.weights = np.random.normal(0, 0.01, self.features)
        self.bias = 0
        self.gradients = np.zeros((self.features))
        self.gradient_bias = 0
        self.epochs = epochs
        self.tol = tol
        self._finished = False
        self.last_accuracy = None
        self.y_ = np.where(y <= 0, -1, 1)

        for epoch in range(epochs):
            self._train(epoch)
            if self.verbose:
                print(f"Accuracy on epoch {epoch}: {accuracy_score(self.y_, self.predict(X))}")
        return self

    def predict(self, X: np.array):
        super().predict(X)
        return np.sign(np.dot(X, self.weights) - self.bias)

    def _compute_gradient(self):
        self.gradients = np.zeros(self.weights.shape)
        self.gradient_bias = 0

        for i in range(self.samples):
            condition = self.y_[i] * (np.dot(self.X[i], self.weights) - self.bias) >= 1
            if not condition:
                self.gradients += -np.dot(self.X[i], self.y_[i])
                self.gradient_bias += -self.y_[i]
            self.gradients += 2 * self.regularization * self.weights
        self.gradients /= self.samples
        self.bias /= self.samples

    def _train(self, epoch):
        if self.method == "default":
            self._compute_gradient()
        self.weights -= self.learning_rate * self.gradients
        self.bias -= self.learning_rate * self.gradient_bias
        self.learning_rate = self.initial_learning_rate / (1 + self.decay * epoch)
