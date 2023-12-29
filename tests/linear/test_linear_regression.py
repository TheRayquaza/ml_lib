import pytest
import numpy as np
from linear.linear_regression import LinearRegression


@pytest.mark.parametrize(
    "learning_rate, decay, method",
    [
        (1e-4, 1e-3, "default"),
        (1e-3, 1e-2, "stochastic"),
        (1e-5, 1e-1, "mini-batch"),
        (1e-2, 0, "analytic"),
    ],
)
def test_linear_regression_init(learning_rate, decay, method):
    model = LinearRegression(learning_rate=learning_rate, decay=decay, method=method)
    assert model.learning_rate == learning_rate
    assert model.decay == decay
    assert model.method == method


@pytest.mark.parametrize(
    "X, y, epochs, tol, method",
    [
        (np.random.randn(100, 10), np.random.randn(100, 1), 100, 1e-4, "default"),
        (np.random.randn(200, 5), np.random.randn(200, 1), 50, 1e-3, "stochastic"),
        (np.random.randn(150, 20), np.random.randn(150, 1), 200, 1e-2, "mini-batch"),
        (np.random.randn(50, 3), np.random.randn(50, 1), 10, 1e-5, "analytic"),
    ],
)
def test_linear_regression_fit(X, y, epochs, tol, method):
    model = LinearRegression(method=method)
    model.fit(X, y, epochs, tol)
    assert model._fitted


@pytest.mark.parametrize(
    "X",
    [
        np.random.randn(50, 10),
        np.random.randn(100, 5),
    ],
)
def test_linear_regression_predict(X):
    model = LinearRegression()
    y = np.random.randn(X.shape[0], 1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0], 1)
