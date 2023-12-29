import numpy as np
import pytest
from linear.ridge_regression import RidgeRegression


@pytest.mark.parametrize(
    "learning_rate, alpha, method",
    [
        (0.01, 0.1, "default"),
        (0.005, 0.05, "mini-batch"),
        (0.02, 0.2, "stochastic"),
        (0.015, 0.15, "mini-batch"),
    ],
)
def test_initialization(learning_rate, alpha, method):
    model = RidgeRegression(learning_rate=learning_rate, alpha=alpha, method=method)
    assert model.learning_rate == learning_rate
    assert model.alpha == alpha
    assert model.method == method
    assert not model._fitted


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[0], [1]])),
        (np.array([[0.5, 1], [2, 3.5]]), np.array([[1], [0]])),
    ],
)
def test_fit(X, y):
    model = RidgeRegression()
    model.fit(X, y)
    assert model._fitted
    assert model.weights.shape == (X.shape[1], 1)


@pytest.mark.parametrize(
    "X, y, expected_shape",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[0], [1]]), (2, 1)),
        (np.array([[0.5, 1], [2, 3.5]]), np.array([[1], [0]]), (2, 1)),
    ],
)
def test_predict(X, y, expected_shape):
    model = RidgeRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == expected_shape


@pytest.mark.parametrize("method", ["unknown", "mini", None])
def test_invalid_method(method):
    with pytest.raises(ValueError):
        RidgeRegression(method=method)


def test_predict_unfitted():
    X = np.array([[1, 2], [3, 4]])
    model = RidgeRegression()
    with pytest.raises(ValueError):
        model.predict(X)
