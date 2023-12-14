from classes.model import Model
import numpy as np
import random


class Ridge(Model):
    def __init__(self, learning_rate=0.01, alpha=0.1, method="default"):
        self.fitted = False
        self.learning_rate = learning_rate
        self.method = method
        self.alpha = alpha
        if not method in ["default", "stochastic", "analytic"]:
            raise Exception("Ridge: Unknown method", method)

    def __str__(self) -> str:
        return "Ridge"

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=100, tol=1e-4):
        self.X = X
        self.y = y
        self.id = np.identity(X.shape[1])
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
            raise Exception(
                "Ridge: Shape should be", self.features, "and not", X.shape[1]
            )
        else:
            return np.dot(X, self.weights)

    def __compute_stochastic_gradient(self):
        for m in range(self.X.shape[0]):
            rand_index = random.randint(0, self.X.shape[0] - 1)
            self.gradients = (
                2
                * self.X[rand_index : rand_index + 1].T.dot(
                    self.X[rand_index : rand_index + 1].dot(self.weights)
                    - self.y[rand_index : rand_index + 1]
                )
                + 2 * self.alpha * self.weights
            )

    def __compute_gradient(self):
        self.gradients = (
            2 * self.X.T.dot(self.X.dot(self.weights) - self.y)
            + 2 * self.alpha * self.weights
        )

    def __train(self, epoch):
        if not self.fitted:
            raise Exception("Ridge: Model not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                "Ridge: Invalid shapes Xshape",
                self.X.shape,
                "while yshape",
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
            self.weights -= self.learning_rate / (epoch + 1) * self.gradients

    def __train_analytic(self):
        if not self.fitted:
            raise Exception("Ridge: Model not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                "Ridge: Invalid shapes X", self.X.shape, "while y", self.y.shape
            )
        else:
            self.weights = np.linalg.inv(
                self.X.T.dot(self.X) + self.alpha * self.id
            ).dot(self.X.T.dot(self.y))
