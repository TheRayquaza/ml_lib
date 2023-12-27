import numpy as np
import pytest
from dataset.classification import generate_classification_dataset
from knn.knn_classifier import (
    KNeighborsClassifier,
)  # Adjust the import according to your project structure


def test_kneighbors_classifier_init():
    classifier = KNeighborsClassifier(k=3, distance_method=np.linalg.norm)
    assert classifier.k == 3
    assert classifier.distance_method == np.linalg.norm
    assert (
        classifier.n_jobs == -1 or classifier.n_jobs > 0
    )  # Depends on the system's CPU count
    assert not classifier._fitted

    with pytest.raises(ValueError):
        KNeighborsClassifier(k=-1)


def test_kneighbors_classifier_fit():
    X, y = generate_classification_dataset()
    classifier = KNeighborsClassifier(k=5)
    classifier.fit(X, y)
    assert classifier._fitted


def test_kneighbors_classifier_predict():
    X, y = generate_classification_dataset()
    classifier = KNeighborsClassifier(k=5)
    classifier.fit(X, y)
    predictions = classifier.predict(X)
    assert len(predictions) == len(X)


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
