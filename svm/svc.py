from classes.model import Model
import numpy as np
import random


class SVC(Model):
    def __init__(
        self,
        learning_rate=1e-4,
        decay=1e-3,
        regularization=1e-3,
        kernel="linear",
        degree=2,
        gamma=1,
        random_state=None,
    ):
        random.seed(random_state)
        self.fitted = False
        self.learning_rate = learning_rate
        self.initial_learning_reate = learning_rate
        self.decay = decay
        self.regularization = regularization
        self.kernel = kernel
        if not self.kernel in ["linear", "poly", "rbf", "sigmoid"]:
            raise ValueError("SVC: Unknown kernel ", kernel)
        self.kernel_function = None
        if self.kernel == "linear":
            self.kernel_function = self.__linear_kernel
        elif self.kernel == "poly":
            self.kernel_function = self.__polynomial_kernel
            self.gamma = gamma
            self.degree = degree
        elif self.kernel == "rbf":
            self.kernel_function = self.__rbf_kernel
            self.gamma = gamma
        elif self.kernel == "sigmoid":
            self.kernel_function = self.__sigmoid_kernel
            self.gamma = gamma

    def __linear_kernel(self, X1: np.array, X2: np.array):
        return np.dot(X1, X2)

    def __polynomial_kernel(self, X1: np.array, X2: np.array):
        return (self.gamma * np.dot(X1, X2)) ** self.degree

    def __rbf_kernel(self, X1: np.array, X2: np.array):
        return np.exp(-self.gamma * np.sqrt(X1 - X2))  # TODO

    def __sigmoid_kernel(self, X1: np.array, X2: np.array):
        return np.tanh(self.gamma * np.dot(X1, X2))

    def __str__(self) -> str:
        return "SVC"

    def fit(self, X: np.ndarray, y: np.ndarray, epochs=1000, tol=1e-3):
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.fitted = True
        self.epochs = epochs
        self.tol = tol
        self.gradients = np.zeros((X.shape[1], 1))
        self.__finished = False
        for epoch in range(epochs):
            if self.__finished:
                self.__finished = False
                break
            self.__train(epoch)

    def predict(self, X: np.array):
        if X.shape[1] != self.features:
            raise Exception("Shape should be", self.features, "and not", X.shape[1])
        else:
            result = 0
            for i in range(X.shape[1]):
                result += self.weights[i] * self.kernel_function(self.X[:, i], self.X)
            return result.reshape(-1, 1)

    def __compute_gradient(self):
        self.gradients = np.zeros(self.weights.shape)
        dot_prod = self.predict(self.X) * self.y
        self.gradients += 2 * self.regularization * self.weights
        for i in range(dot_prod.shape[0]):
            if dot_prod[i] < 1:
                self.gradients -= np.dot(self.X[i : i + 1, :].T, self.y[i]).reshape(
                    -1, 1
                )
        self.weights -= self.learning_rate * self.gradients

    def __train(self, epoch):
        if not self.fitted:
            raise Exception("SVC: not fitted")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception(
                "SVC: Invalid shapes X=",
                self.X.shape,
                "while y=",
                self.y.shape,
            )
        else:
            last_gradients = self.gradients
            self.__compute_gradient()
            if abs(np.sum(last_gradients) - np.sum(self.gradients)) <= self.tol:
                self.__finished = True
            self.weights -= self.learning_rate * (self.weights * self.gradients)
            self.learning_rate = self.initial_learning_reate / (1 + self.decay * epoch)
