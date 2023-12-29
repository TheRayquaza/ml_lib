import numpy as np
import pytest
from dataset.classification import generate_classification_dataset
from knn.knn_classifier import (
    KNeighborsClassifier,
)  # Adjust the import according to your project structure


@pytest.mark.parametrize("k, distance_method, n_jobs", [
    (3, np.linalg.norm, -1),
    (5, None, None),
    (5, np.linalg.norm, 2),
    (7, lambda a, b: np.sum(np.abs(a - b)), 4),
])
def test_kneighbors_classifier_init(k, distance_method, n_jobs):
    if k < 0:
        with pytest.raises(ValueError):
            KNeighborsClassifier(k=k)
    else:
        classifier = KNeighborsClassifier(k=k, distance_method=distance_method, n_jobs=n_jobs)
        assert classifier.k == k
        assert classifier.distance_method == distance_method
        if n_jobs != -1:
            assert classifier.n_jobs == n_jobs
        assert not classifier._fitted

@pytest.mark.parametrize("n_jobs", [None, 2])
def test_kneighbors_classifier_parallel_predict(n_jobs):
    X, y = generate_classification_dataset()
    classifier = KNeighborsClassifier(k=5, n_jobs=n_jobs)
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)

@pytest.mark.parametrize("distance_method", [None, lambda a, b: np.sum(np.abs(a - b))])
def test_kneighbors_classifier_custom_distance_method(distance_method):
    X, y = generate_classification_dataset()
    classifier = KNeighborsClassifier(k=5, distance_method=distance_method)
    classifier.fit(X, y)
    classifier.predict(X)  # Just to ensure no error is raised


def test_kneighbors_classifier_fit():
    X, y = generate_classification_dataset()
    classifier = KNeighborsClassifier(k=5)
    classifier.fit(X, y)
    assert classifier._fitted


def test_kneighbors_classifier_predict_without_fit():
    classifier = KNeighborsClassifier(k=5)
    X, _ = generate_classification_dataset()

    with pytest.raises(Exception, match="not fitted"):
        classifier.predict(X)


def test_kneighbors_classifier_parallel_predict():
    X, y = generate_classification_dataset()
    classifier = KNeighborsClassifier(k=5, n_jobs=2)
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)


def test_kneighbors_classifier_custom_distance_method():
    def manhattan_distance(a, b):
        return np.sum(np.abs(a - b))

    X, y = generate_classification_dataset()
    classifier = KNeighborsClassifier(k=5, distance_method=manhattan_distance)
    classifier.fit(X, y)
    classifier.predict(X)  # Just to ensure no error is raised
