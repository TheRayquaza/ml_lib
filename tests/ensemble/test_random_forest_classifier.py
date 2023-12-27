import pytest
import numpy as np
from ensemble.random_forest_classifier import RandomForestClassifier


def test_instantiation():
    clf = RandomForestClassifier(n_estimators=10, max_depth=5, method="gini")
    assert isinstance(clf, RandomForestClassifier)
    assert clf.n_estimators == 10
    assert clf.max_depth == 5
    assert clf.method == "gini"


def test_fit():
    X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
    y = np.array([0, 1, 0, 1])
    clf = RandomForestClassifier(n_estimators=5)
    clf.fit(X, y)
    assert clf._fitted


def test_predict():
    X_train = np.array([[1, 2], [3, 4], [5, 6]])
    y_train = np.array([0, 1, 0])
    X_test = np.array([[2, 3], [4, 5]])

    clf = RandomForestClassifier(n_estimators=5)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)
    assert predictions.shape == (2, 1)


def test_invalid_n_estimators():
    with pytest.raises(ValueError):
        RandomForestClassifier(n_estimators=0)


def test_empty_dataset():
    clf = RandomForestClassifier(n_estimators=5)
    with pytest.raises(ValueError):
        clf.fit(np.array([]), np.array([]))
