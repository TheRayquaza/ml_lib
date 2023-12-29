import pytest
import numpy as np
from dataset.generic import generate_linear_dataset
from knn.knn_regressor import KNeighborsRegressor
from dataset.generic import generate_linear_dataset


import pytest
import numpy as np
from dataset.generic import generate_linear_dataset
from knn.knn_regressor import KNeighborsRegressor
from dataset.generic import generate_linear_dataset

@pytest.mark.parametrize("k, n_jobs, expected_shape", [
    (5, None, (200, 1)),
    (10, 4, (200, 1)),
    (3, None, (200, 1)),
    (5, 2, (200, 1)),
])
def test_kneighbors_regressor_initialization(k, n_jobs, expected_shape):
    knr = KNeighborsRegressor(k=k, n_jobs=n_jobs)
    assert knr.k == k
    assert knr.n_jobs == n_jobs

    if k < 0:
        with pytest.raises(ValueError):
            KNeighborsRegressor(k=k)
    else:
        knr = KNeighborsRegressor(k=k, n_jobs=n_jobs)
        assert knr.k == k
        assert knr.n_jobs == n_jobs

@pytest.mark.parametrize("n_jobs, expected_shape", [
    (None, (200, 1)),
    (2, (200, 1)),
])
def test_kneighbors_regressor_parallel_predict(n_jobs, expected_shape):
    X, y = generate_linear_dataset(200)
    knr = KNeighborsRegressor(n_jobs=n_jobs)
    knr.fit(X, y)
    predictions = knr.predict(X)
    assert predictions.shape == expected_shape

@pytest.mark.parametrize("distance_method, expected_shape", [
    (None, (200, 1)),
    (lambda a, b: np.sum(np.abs(a - b)), (200, 1)),
    # Add more cases as needed
])
def test_kneighbors_regressor_distance_method(distance_method, expected_shape):
    X, y = generate_linear_dataset(200)
    knr = KNeighborsRegressor(distance_method=distance_method)
    knr.fit(X, y)
    predictions = knr.predict(X)
    assert predictions.shape == expected_shape


def test_kneighbors_regressor_predict():
    X, y = generate_linear_dataset(200)
    knr = KNeighborsRegressor()
    knr.fit(X, y)
    predictions = knr.predict(X)
    assert predictions.shape == (X.shape[0], 1)


def test_kneighbors_regressor_predict_without_fit():
    knr = KNeighborsRegressor()
    X, _ = generate_linear_dataset(200)
    with pytest.raises(Exception):
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
