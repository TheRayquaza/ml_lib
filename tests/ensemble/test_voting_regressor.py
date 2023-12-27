import numpy as np
import pytest
from dataset.generic import generate_linear_dataset
from ensemble.voting_regressor import VotingRegressor
from tree.decision_tree_regressor import DecisionTreeRegressor
from linear.linear_regression import LinearRegression
from ensemble.random_forest_regressor import RandomForestRegressor


@pytest.mark.parametrize(
    "estimators",
    [
        [
            ("DecisionTree", DecisionTreeRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
        [
            ("RandomForest", RandomForestRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
    ],
)
def test_voting_regressor_init(estimators):
    n_jobs = -1
    voting_regressor = VotingRegressor(estimators, n_jobs=n_jobs)

    assert voting_regressor.estimators == estimators
    assert voting_regressor.n_estimators == len(estimators)
    assert not voting_regressor._fitted

    with pytest.raises(ValueError, match="estimators"):
        VotingRegressor([], n_jobs=n_jobs)


@pytest.mark.parametrize(
    "estimators",
    [
        [
            ("DecisionTree", DecisionTreeRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
        [
            ("RandomForest", RandomForestRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
    ],
)
def test_voting_regressor_str(estimators):
    voting_regressor = VotingRegressor(estimators)
    assert str(voting_regressor) == "VotingRegressor"


@pytest.mark.parametrize(
    "estimators",
    [
        [
            ("DecisionTree", DecisionTreeRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
        [
            ("RandomForest", RandomForestRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
    ],
)
def test_voting_regressor_fit_predict(estimators):
    voting_regressor = VotingRegressor(estimators)

    X, y = generate_linear_dataset(200)

    voting_regressor.fit(X, y)

    assert voting_regressor._fitted

    predictions = voting_regressor.predict(X)
    assert len(predictions) == len(X)


@pytest.mark.parametrize(
    "estimators",
    [
        [
            ("DecisionTree", DecisionTreeRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
        [
            ("RandomForest", RandomForestRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
    ],
)
def test_voting_regressor_predict_without_fit(estimators):
    voting_regressor = VotingRegressor(estimators)
    X, _ = generate_linear_dataset(200)

    with pytest.raises(Exception, match="not fitted"):
        voting_regressor.predict(X)


@pytest.mark.parametrize(
    "estimators",
    [
        [
            ("DecisionTree", DecisionTreeRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
        [
            ("RandomForest", RandomForestRegressor()),
            ("LinearRegression", LinearRegression()),
        ],
    ],
)
def test_voting_regressor_parallel_fit_predict(estimators):
    n_jobs = 3
    voting_regressor = VotingRegressor(estimators, n_jobs=n_jobs)

    X, y = generate_linear_dataset(200)

    voting_regressor.fit(X, y)

    assert voting_regressor._fitted

    predictions = voting_regressor.predict(X)
    assert len(predictions) == len(X)
