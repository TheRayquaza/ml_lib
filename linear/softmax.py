from classes.model import Model
import numpy as np
import random
import math
from scipy.special import softmax


class SoftmaxRegression(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-3,
        method="default",
        threshold=0.5,
        random_state=None,
    ):
        random.seed(random_state)
        self.fitted = False
        self.learning_rate = learning_rate
        self.initial_learning_reate = learning_rate
        self.decay = decay
        self.method = method
        self.threshold = threshold
        if not method in ["default", "stochastic", "analytic"]:
            raise Exception("LogisticRegression: Unknown method " + method)

    def __str__(self) -> str:
        return "LogisticRegression"

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, tol=1e-4):
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.fitted = True
        self.epochs = epochs
        self.tol = tol
        self.gradients = np.zeros((X.shape[1], 1))
        self.__finished = False
        if self.method == "analytic":
            self.__train_analytic()
        else:
            for epoch in range(epochs):
                if self.__finished:
                    self.__finished = False
                    break
                self.__train(epoch)

    def predict(self, X: np.array):
        if X.shape[1] != self.features:
            raise Exception("Shape should be", self.features, "and not", X.shape[1])
        else:
            return np.argmax(softmax(np.dot(X, self.weights)))

    def __compute_stochastic_gradient(self):
        for m in range(self.X.shape[0]):
            rand_index = random.randint(0, self.X.shape[0] - 1)
            self.gradients = 2 * self.X[rand_index : rand_index + 1].T.dot(
                self.X[rand_index : rand_index + 1].dot(self.weights)
                - self.y[rand_index : rand_index + 1]
            )

    def __compute_gradient(self):
        self.gradients = 2 * self.X.T.dot(self.X.dot(self.weights) - self.y)

    def __train(self, epoch):
        if not self.fitted:
            raise Exception("LogisticRegression: not fitted")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                "LogisticRegression: Invalid shapes X=",
                self.X.shape,
                "while y=",
                self.y.shape,
            )
        else:
            last_gradients = self.gradients
            if self.method == "default":
                self.__compute_gradient()
            elif self.method == "stochastic":
                self.__compute_stochastic_gradient()
            if abs(np.sum(last_gradients) - np.sum(self.gradients)) <= self.tol:
                self.__finished = True
            self.weights -= self.learning_rate * self.gradients
            self.learning_rate = self.initial_learning_reate / (1 + self.decay * epoch)
