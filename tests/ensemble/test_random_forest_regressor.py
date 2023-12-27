import numpy as np
import pytest
from ensemble.random_forest_regressor import RandomForestRegressor
from dataset.generic import generate_linear_dataset


@pytest.mark.parametrize(
    "n_estimators, max_depth, method, n_jobs, bootstrap",
    [(5, None, "mse", None, True), (10, 3, "rmse", 2, False), (3, 5, "mae", 10, True)],
)
def test_random_forest_regressor_init(
    n_estimators, max_depth, method, n_jobs, bootstrap
):
    random_forest_regressor = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        method=method,
        n_jobs=n_jobs,
        bootstrap=bootstrap,
    )

    assert random_forest_regressor.n_estimators == n_estimators
    assert random_forest_regressor.max_depth == max_depth
    assert random_forest_regressor.method == method
    assert random_forest_regressor.n_jobs == n_jobs
    assert random_forest_regressor.bootstrap == bootstrap
    assert not random_forest_regressor._fitted

    with pytest.raises(ValueError, match="estimators"):
        RandomForestRegressor(n_estimators=0)

    with pytest.raises(ValueError, match="max_depth"):
        RandomForestRegressor(n_estimators=3, max_depth=-1)


def test_random_forest_regressor_str():
    random_forest_regressor = RandomForestRegressor()
    assert str(random_forest_regressor) == "RandomForestRegressor"


def test_random_forest_regressor_fit_predict():
    random_forest_regressor = RandomForestRegressor()
    X, y = generate_linear_dataset(200)

    random_forest_regressor.fit(X, y)

    assert random_forest_regressor._fitted

    predictions = random_forest_regressor.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)


def test_random_forest_regressor_predict_without_fit():
    random_forest_regressor = RandomForestRegressor()
    X, _ = generate_linear_dataset(200)

    with pytest.raises(Exception, match="not fitted"):
        random_forest_regressor.predict(X)


def test_random_forest_regressor_parallel_fit_predict():
    random_forest_regressor = RandomForestRegressor(n_estimators=3, n_jobs=-1)
    X, y = generate_linear_dataset(200)

    random_forest_regressor.fit(X, y)

    assert random_forest_regressor._fitted

    predictions = random_forest_regressor.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)
