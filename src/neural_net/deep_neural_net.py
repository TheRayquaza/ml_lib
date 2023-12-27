from classes.model import Model
import numpy as np
import random


def sigmoid(z: np.array) -> np.array:
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_prime(z: np.array) -> np.array:
    return sigmoid(z) * (1 - sigmoid(z))


def relu(z: np.array) -> np.array:
    return np.maximum(0, z)


def relu_prime(z: np.array) -> np.array:
    z[z < 0] = 0
    z[z > 0] = 1
    return z


def tanh(z: np.array) -> np.array:
    return np.tanh(z)


def tanh_prime(z: np.array) -> np.array:
    return 1 - np.power(tanh(z), 2)


def id(z: np.array) -> np.array:
    return z


def id_prime(z: np.array) -> np.array:
    return 1


def softmax(z):
    exp_z = np.exp(z - np.max(z, axis=-1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


class DeepNeuralNetwork(Model):
    def __init__(
        self,
        layers,
        n_features,
        method="default",
        task="regression",
        epochs=100,
        batch_size=10,
        decay=1e-2,
        eta=1e-2,
        threshold=1e-1,
        activation=sigmoid,
        activation_prime=sigmoid_prime,
        last_activation=None,
        random_state=None,
    ):
        random.seed(random_state)
        self.n_layers = len(layers)
        self.eta = eta
        self.decay = decay
        self.current_eta = eta
        self.activation = activation
        self.activation_prime = activation_prime
        self.last_activation = last_activation
        self.threshold = threshold
        self.method = method
        self.task = task
        self.epochs = epochs
        self.last_errors = None
        self.batch_size = batch_size
        self.n_features = n_features
        self.n_layers = len(layers)
        self.layers = layers
        self.__fitted = False
        if self.layers[0] != self.n_features:
            raise ValueError(
                "DeepNeuralNetwork: input layer should have the amount of neurons as features"
            )
        if self.n_layers < 2:
            raise ValueError(f"DeepNeuralNetwork: len(layers) < 2 (= {self.n_layers})")
        if not method in ["default", "stochastic", "mini-batch"]:
            raise ValueError(f"DeepNeuralNetwork: method {method} is unknown")
        if not task in ["classifcation", "regression"]:
            raise ValueError(f"DeepNeuralNetwork: task {task} is unknown")
        if method == "mini-batch" and batch_size <= 0:
            raise ValueError(f"DeepNeuralNetwork: batch_size negative or zero")

    def __str__(self):
        return "DeepNeuralNetwork"

    def __initialize_layers(self, layers):
        self.weights = [None] * (self.n_layers - 1)
        self.dweights = [None] * (self.n_layers - 1)
        self.bias = [None] * (self.n_layers - 1)
        self.dbias = [None] * (self.n_layers - 1)
        self.z = [None] * (self.n_layers - 1)
        self.x = [None] * (self.n_layers - 1)
        for i in range(self.n_layers - 1):
            if layers[i] <= 0:
                raise ValueError(
                    f"DeepNeuralNetwork: unable to create neural layer with {layers[i]} neurons"
                )
            self.weights[i] = np.random.randn(layers[i], layers[i + 1]) * np.sqrt(
                2.0 / layers[i]
            )
            self.dweights[i] = np.zeros(shape=(layers[i], layers[i + 1]))
            self.bias[i] = np.random.randn(1, layers[i + 1]) * np.sqrt(2.0 / layers[i])
            self.dbias[i] = np.zeros(shape=(1, layers[i + 1]))

    def __feed_forward(self, X):
        result = X
        for i in range(self.n_layers - 1):
            self.x[i] = result
            self.z[i] = np.dot(result, self.weights[i]) + self.bias[i]
            if i == self.n_layers - 2:
                result = self.z[i]
            else:
                result = self.activation(self.z[i])
        if self.last_activation:
            if self.task == "classification":
                return np.argmax(self.last_activation(result))
            else:
                return self.last_activation(result)
        else:
            if self.task == "classifcation":
                return np.argmax(result)
            else:
                return result

    def predict(self, X):
        if not self.__fitted:
            raise Exception("DeepNeuralNetwork: not fitted yet")
        return self.__feed_forward(X)

    def __train(self, X, y):
        for epoch in range(self.epochs):
            self.current_eta = self.eta / (1 + self.decay * epoch)
            y_pred = self.__feed_forward(X)
            errors = y_pred - y
            mse = np.linalg.norm(errors)

            if mse < self.threshold:
                break
            if (
                self.last_errors is np.array
                and np.linalg.norm(errors - self.last_errors) < self.threshold
            ):
                break

            self.last_errors = errors

            for i in range(self.n_layers - 2, -1, -1):
                if i == self.n_layers - 2:
                    errors = errors * self.activation_prime(self.z[i])
                else:
                    errors = np.dot(
                        errors, self.weights[i + 1].T
                    ) * self.activation_prime(self.z[i])
                self.dweights[i] = np.dot(errors.T, self.x[i])
                self.dbias[i] = np.sum(errors, axis=0, keepdims=True)
                self.weights[i] -= self.current_eta * self.dweights[i].T
                self.bias[i] -= self.current_eta * self.dbias[i]

    def cost(self, y_pred, y):
        if self.task == "regression":
            return np.linalg.norm(y_pred - y) / 2.0
        else:
            y_error = np.zeros_like(y)
            for i in range(y_pred.shape[0]):
                y_error[i] = np.argmax(y_pred[i, :]) - y[i]
            return np.sum(y_error)

    def fit(self, X, y):
        self.X = X
        self.y = y
        self.__fitted = True
        self.__initialize_layers(self.layers)
        if self.method == "stochastic":
            for i in range(X.shape[0]):
                self.__train(X[i], y[i])
        elif self.method == "default":
            self.__train(X, y)
        else:
            indices = np.arange(X.shape[0])
            np.random.shuffle(indices)
            X_shuffled, y_shuffled = X[indices], y[indices]
            mini_batches = [
                (
                    X_shuffled[i : i + self.batch_size],
                    y_shuffled[i : i + self.batch_size],
                )
                for i in range(0, X.shape[0], self.batch_size)
            ]
            for X_batch, y_batch in mini_batches:
                self.__train(X_batch, y_batch)
