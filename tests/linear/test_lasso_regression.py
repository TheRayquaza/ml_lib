import pytest
import numpy as np
from linear.lasso_regression import LassoRegression


@pytest.mark.parametrize(
    "learning_rate, decay, alpha, method, random_state",
    [
        (1e-4, 1e-2, 0.1, "default", None),
        (1e-3, 1e-3, 0.5, "stochastic", 42),
        (1e-5, 1e-1, 0, "mini-batch", 10),
        (1e-2, 0, 1, "default", 0),
    ],
)
def test_lasso_regression_init(learning_rate, decay, alpha, method, random_state):
    model = LassoRegression(
        learning_rate=learning_rate,
        decay=decay,
        alpha=alpha,
        method=method,
        random_state=random_state,
    )
    assert model.learning_rate == learning_rate
    assert model.decay == decay
    assert model.alpha == alpha
    assert model.method == method


@pytest.mark.parametrize(
    "X, y, epochs, tol",
    [
        (np.random.randn(100, 10), np.random.randn(100, 1), 100, 1e-4),
        (np.random.randn(200, 5), np.random.randn(200, 1), 50, 1e-3),
    ],
)
def test_lasso_regression_fit(X, y, epochs, tol):
    model = LassoRegression()
    model.fit(X, y, epochs, tol)
    assert model._fitted


@pytest.mark.parametrize(
    "X",
    [
        np.random.randn(50, 10),
        np.random.randn(100, 5),
    ],
)
def test_lasso_regression_predict(X):
    model = LassoRegression()
    y = np.random.randn(X.shape[0], 1)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0], 1)
