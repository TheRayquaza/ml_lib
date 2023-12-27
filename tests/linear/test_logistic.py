import numpy as np
import pytest
from linear.logistic import LogisticRegression


def test_initialization():
    model = LogisticRegression(
        learning_rate=0.01, decay=0.001, method="default", threshold=0.5
    )
    assert model.learning_rate == 0.01
    assert model.decay == 0.001
    assert model.method == "default"
    assert model.threshold == 0.5
    assert not model._fitted


def test_fit():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])
    model = LogisticRegression()
    model.fit(X, y)
    assert model._fitted
    assert model.weights.shape == (2, 1)


def test_predict():
    X = np.array([[1, 2], [3, 4]])
    y = np.array([[0], [1]])
    model = LogisticRegression()
    model.fit(X, y)
    predictions = model.predict(X)
    assert len(predictions) == 2
    assert predictions.dtype == np.bool_


def test_invalid_method():
    with pytest.raises(ValueError):
        LogisticRegression(method="invalid_method")


def test_predict_unfitted():
    X = np.array([[1, 2], [3, 4]])
    model = LogisticRegression()
    with pytest.raises(Exception):
        model.predict(X)
