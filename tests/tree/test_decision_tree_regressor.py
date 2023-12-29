import numpy as np
import pytest
from metrics.regression_metrics import mse
from dataset.generic import generate_linear_dataset
from tree.decision_tree_regressor import DecisionTreeRegressor


@pytest.mark.parametrize("max_depth", [2, 3, None])
@pytest.mark.parametrize("method", ["mse", "mae", "rmse"])
@pytest.mark.parametrize("n_jobs", [1, 2, None])
@pytest.mark.parametrize("split", ["best", "random"])
def test_initialization(max_depth, method, n_jobs, split):
    model = DecisionTreeRegressor(
        max_depth=max_depth, method=method, n_jobs=n_jobs, split=split
    )
    assert model.max_depth == max_depth
    assert model.method == method
    assert model.n_jobs == n_jobs
    assert model.split == split


@pytest.mark.parametrize("invalid_method", ["invalid", "unknown"])
def test_invalid_method(invalid_method):
    with pytest.raises(ValueError):
        DecisionTreeRegressor(method=invalid_method)


@pytest.mark.parametrize("invalid_split", ["invalid", "unknown"])
def test_invalid_split(invalid_split):
    with pytest.raises(ValueError):
        DecisionTreeRegressor(split=invalid_split)


def test_fit():
    X, y = generate_linear_dataset(200)
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    assert model.root is not None


def test_predict_unfitted():
    model = DecisionTreeRegressor(max_depth=2)
    with pytest.raises(ValueError):
        model.predict(np.array([[1, 2]]))


def test_fit_predict():
    X, y = generate_linear_dataset(200)
    model = DecisionTreeRegressor(max_depth=2)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0], 1)
    assert mse(y, model.predict(X)) < 100 * np.max(X)
