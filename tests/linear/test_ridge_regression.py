import numpy as np
import pytest
from linear.ridge_regression import RidgeRegression


def test_initialization():
    model = RidgeRegression(learning_rate=0.01, alpha=0.1, method="default")
    assert model.learning_rate == 0.01
    assert model.alpha == 0.1
    assert model.method == "default"
    assert not model.fitted


def test_fit():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])
    model = RidgeRegression()
    model.fit(X, y)
    assert model.fitted
    assert model.weights.shape == (2, 1)


def test_predict():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])
    model = RidgeRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (2, 1)


def test_invalid_method():
    with pytest.raises(ValueError):
        RidgeRegression(method="invalid_method")


def test_predict_unfitted():
    X = np.array([[1, 2], [3, 4]])
    model = RidgeRegression()
    with pytest.raises(ValueError):
        model.predict(X)
