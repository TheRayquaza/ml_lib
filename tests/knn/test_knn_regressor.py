import pytest
import numpy as np
from dataset.generic import generate_linear_dataset
from knn.knn_regressor import KNeighborsRegressor
from dataset.generic import generate_linear_dataset


def test_kneighbors_regressor_initialization():
    # Test initialization with default parameters
    knr = KNeighborsRegressor()
    assert knr.k == 5
    assert knr.distance_method is not None
    assert knr.n_jobs is None

    # Test initialization with custom parameters
    custom_k = 10
    custom_n_jobs = 4
    knr = KNeighborsRegressor(k=custom_k, n_jobs=custom_n_jobs)
    assert knr.k == custom_k
    assert knr.n_jobs == custom_n_jobs

    # Test initialization with invalid k
    with pytest.raises(ValueError):
        KNeighborsRegressor(k=-1)


def test_kneighbors_regressor_fit():
    X, y = generate_linear_dataset(200)
    knr = KNeighborsRegressor()
    knr.fit(X, y)
    assert knr._fitted


def test_kneighbors_regressor_predict():
    X, y = generate_linear_dataset(200)
    knr = KNeighborsRegressor()
    knr.fit(X, y)
    predictions = knr.predict(X)
    assert predictions.shape == (X.shape[0], 1)


def test_kneighbors_regressor_predict_without_fit():
    knr = KNeighborsRegressor()
    X, _ = generate_linear_dataset(200)
    with pytest.raises(Exception, match="not fitted"):
        knr.predict(X)


def test_kneighbors_regressor_parallel_predict():
    X, y = generate_linear_dataset(200)
    knr = KNeighborsRegressor(n_jobs=2)
    knr.fit(X, y)
    predictions = knr.predict(X)
    assert predictions.shape == (X.shape[0], 1)


def test_kneighbors_regressor_distance_method():
    X, y = generate_linear_dataset(200)
    knr = KNeighborsRegressor(distance_method=lambda a, b: np.sum(np.abs(a - b)))
    knr.fit(X, y)
    predictions = knr.predict(X)
    assert predictions.shape == (X.shape[0], 1)
