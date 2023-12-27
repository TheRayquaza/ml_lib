import numpy as np
import pytest
from dataset.generic import generate_linear_dataset
from tree.decision_tree_regressor import DecisionTreeRegressor
from linear.linear_regression import LinearRegression
from ensemble.bagging_regressor import BaggingRegressor


@pytest.mark.parametrize(
    "base_regressor", [DecisionTreeRegressor(), LinearRegression()]
)
def test_bagging_regressor_init(base_regressor):
    n_estimators = 5
    bagging_regressor = BaggingRegressor(base_regressor, n_estimators)

    assert bagging_regressor.estimator == base_regressor
    assert bagging_regressor.n_estimators == n_estimators
    assert bagging_regressor.n_jobs is None
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
    "base_regressor", [DecisionTreeRegressor(), LinearRegression()]
)
def test_bagging_regressor_fit_predict(base_regressor):
    bagging_regressor = BaggingRegressor(base_regressor)

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
    "base_regressor", [DecisionTreeRegressor(), LinearRegression()]
)
def test_bagging_regressor_parallel_fit_predict(base_regressor):
    n_estimators = 3
    bagging_regressor = BaggingRegressor(base_regressor, n_estimators, n_jobs=-1)

    X, y = generate_linear_dataset(200)

    bagging_regressor.fit(X, y)

    assert bagging_regressor._fitted

    predictions = bagging_regressor.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)
