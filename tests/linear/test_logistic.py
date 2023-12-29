import numpy as np
import pytest
from linear.logistic import LogisticRegression


@pytest.mark.parametrize(
    "learning_rate, decay, method, threshold",
    [
        (0.01, 0.001, "default", 0.5),
        (0.005, 0.0005, "mini-batch", 0.7),
        (0.02, 0.002, "stochastic", 0.3),
        (0.015, 0.0015, "mini-batch", 0.6),
    ],
)
def test_initialization(learning_rate, decay, method, threshold):
    model = LogisticRegression(
        learning_rate=learning_rate, decay=decay, method=method, threshold=threshold
    )
    assert model.learning_rate == learning_rate
    assert model.decay == decay
    assert model.method == method
    assert model.threshold == threshold
    assert not model._fitted


@pytest.mark.parametrize(
    "X, y",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[0], [1]])),
        (np.array([[0.5, 1], [2, 3.5]]), np.array([[1], [0]])),
    ],
)
def test_fit(X, y):
    model = LogisticRegression()
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
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == expected_shape
    assert predictions.dtype == np.bool_


@pytest.mark.parametrize("method", ["invalid", None])
def test_invalid_method(method):
    with pytest.raises(ValueError):
        LogisticRegression(method=method)


def test_predict_unfitted():
    X = np.array([[1, 2], [3, 4]])
    model = LogisticRegression()
    with pytest.raises(Exception):
        model.predict(X)
