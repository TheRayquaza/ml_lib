import pytest
import numpy as np
from ensemble.random_forest_classifier import RandomForestClassifier


@pytest.mark.parametrize(
    "n_estimators, max_depth, method",
    [
        (5, 5, "gini"),
        (10, 3, "entropy"),
        (3, 7, "gini"),
        (8, None, "entropy"),
    ],
)
def test_random_forest_classifier_instantiation(n_estimators, max_depth, method):
    clf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, method=method
    )
    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == n_estimators
    assert clf.max_depth == max_depth
    assert clf.method == method


@pytest.mark.parametrize(
    "X_train, y_train, X_test",
    [
        (
            np.array([[1, 2], [3, 4], [5, 6]]),
            np.array([0, 1, 0]),
            np.array([[2, 3], [4, 5]]),
        ),
        (
            np.array([[1, 2], [3, 4], [5, 6], [7, 8]]),
            np.array([0, 1, 0, 1]),
            np.array([[2, 3], [4, 5]]),
        ),
    ],
)
def test_random_forest_classifier_fit_predict(X_train, y_train, X_test):
    clf = RandomForestClassifier(n_estimators=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    assert predictions.shape == (len(X_test), 1)


@pytest.mark.parametrize(
    "n_estimators",
    [
        0,
        -1,
    ],
)
def test_random_forest_classifier_invalid_n_estimators(n_estimators):
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=n_estimators)


def test_random_forest_classifier_empty_dataset():
    clf = RandomForestClassifier(n_estimators=5)
    with pytest.raises(Exception):
        clf.predict(np.array([]), np.array([]))
