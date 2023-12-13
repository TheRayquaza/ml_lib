from model import Model
import numpy as np
import random

class LinearModel(Model):
    def __init__(self, learning_rate=0.01, epochs=1000, method="default", tol=0.001):
        self.fitted = False
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.method = method
        if not method in ["default", "stochastic", "analytic"]:
            raise Exception("LinearModel: Unknown method", method)
        self.tol = tol

    def __str__(self) -> str:
        return "LinearModel"

    def fit(self, X:np.ndarray, y:np.ndarray, learning_rate=0.01, epochs=100):
        self.X = X
        self.y = y
        self.features = X.shape[1]
        self.weights = np.random.randn(self.X.shape[1], 1)
        self.fitted = True
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.gradients = np.zeros((X.shape[1], 1))
        self.__finished = False
        if self.method == "analytic":
            self.__train_analytic()
        else:
            for epoch in range(epochs):
                if self.__finished:
                    break
                self.__train(epoch)

    def predict(self, X:np.array):
        if (X.shape[1] != self.features):
            raise Exception("Shape should be", self.features, "and not", X.shape[1])
        else:
            return np.dot(X, self.weights)
    
    def __compute_stochastic_gradient(self):
        for m in range(self.X.shape[0]):
            rand_index = random.randint(0, self.X.shape[0] - 1)
            self.gradients = 2 * self.X[rand_index:rand_index+1].T.dot(self.X[rand_index:rand_index+1].dot(self.weights) - self.y[rand_index:rand_index+1])

    def __compute_gradient(self):
        self.gradients = 2 * self.X.T.dot(self.X.dot(self.weights) - self.y)

    def __train(self, epoch):
        if not self.fitted:
            raise Exception("Linear model not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception("Invalid shapes Xshape", self.X.shape, "while yshape", self.y.shape)
        else:
            last_gradients = self.gradients
            if self.method == "default":
                self.__compute_gradient()
            elif self.method == "stochastic":
                self.__compute_stochastic_gradient()
            if abs(np.sum(last_gradients) - np.sum(self.gradients)) <= self.tol :
                self.__finished = True
            self.weights -= self.learning_rate / (epoch + 1) * self.gradients
    
    def __train_analytic(self):
        if not self.fitted:
            raise Exception("LinearModel: Model not fitted with data")
        elif self.y.shape[0] != self.X.shape[0] or self.y.shape[1] != 1:
            raise Exception("LinearModel: Invalid shapes Xshape", self.X.shape, "while yshape", self.y.shape)
        else:
            self.weights = np.linalg.pinv(self.X).dot(self.y)