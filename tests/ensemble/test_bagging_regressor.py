import numpy as np
import pytest
from dataset.generic import generate_linear_dataset
from tree.decision_tree_regressor import DecisionTreeRegressor
from linear.linear_regression import LinearRegression
from ensemble.bagging_regressor import BaggingRegressor


@pytest.mark.parametrize(
    "base_regressor, n_estimators, n_jobs",
    [
        (DecisionTreeRegressor(max_depth=3), 5, None),
        (LinearRegression(), 10, -1),
        (DecisionTreeRegressor(max_depth=5), 3, 2),
        (LinearRegression(), 7, None),
    ],
)
def test_bagging_regressor_init(base_regressor, n_estimators, n_jobs):
    bagging_regressor = BaggingRegressor(base_regressor, n_estimators, n_jobs=n_jobs)

    assert bagging_regressor.estimator == base_regressor
    assert bagging_regressor.n_estimators == n_estimators
    assert not bagging_regressor._fitted

    with pytest.raises(ValueError, match="invalid"):
        BaggingRegressor(None, n_estimators)

    with pytest.raises(ValueError, match="estimators"):
        BaggingRegressor(base_regressor, 0)


@pytest.mark.parametrize(
    "base_regressor", [DecisionTreeRegressor(), LinearRegression()]
)
def test_bagging_regressor_str(base_regressor):
    bagging_regressor = BaggingRegressor(base_regressor)
    assert str(bagging_regressor) == "BaggingRegressor"


@pytest.mark.parametrize(
    "base_regressor, n_estimators, n_jobs",
    [
        (DecisionTreeRegressor(max_depth=3), 5, None),
        (LinearRegression(), 10, -1),
        (DecisionTreeRegressor(max_depth=5), 3, 2),
        (LinearRegression(), 7, None),
    ],
)
def test_bagging_regressor_fit_predict(base_regressor, n_estimators, n_jobs):
    bagging_regressor = BaggingRegressor(base_regressor, n_estimators, n_jobs=n_jobs)

    X, y = generate_linear_dataset(200)

    bagging_regressor.fit(X, y)

    assert bagging_regressor._fitted

    predictions = bagging_regressor.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)


@pytest.mark.parametrize(
    "base_regressor", [DecisionTreeRegressor(), LinearRegression()]
)
def test_bagging_regressor_predict_without_fit(base_regressor):
    bagging_regressor = BaggingRegressor(base_regressor)
    X, _ = generate_linear_dataset(200)

    with pytest.raises(Exception, match="not fitted"):
        bagging_regressor.predict(X)


@pytest.mark.parametrize(
    "base_regressor, n_estimators, n_jobs",
    [
        (DecisionTreeRegressor(max_depth=3), 5, -1),
        (LinearRegression(), 10, 4),
        (DecisionTreeRegressor(max_depth=5), 3, 2),
        (LinearRegression(), 7, None),
    ],
)
def test_bagging_regressor_parallel_fit_predict(base_regressor, n_estimators, n_jobs):
    bagging_regressor = BaggingRegressor(base_regressor, n_estimators, n_jobs=n_jobs)

    X, y = generate_linear_dataset(200)

    bagging_regressor.fit(X, y)

    assert bagging_regressor._fitted

    predictions = bagging_regressor.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)
