import numpy as np
import pytest
from dataset.classification import generate_classification_dataset
from ensemble.bagging_classifier import BaggingClassifier
from tree.decision_tree_classifier import DecisionTreeClassifier
from linear.logistic import LogisticRegression


@pytest.mark.parametrize(
    "base_estimator, n_estimators, n_jobs",
    [
        (DecisionTreeClassifier(max_depth=3), 5, None),
        (LogisticRegression(), 10, -1),
        (DecisionTreeClassifier(max_depth=5), 3, 2),
        (LogisticRegression(), 7, None),
    ],
)
def test_bagging_classifier_init(base_estimator, n_estimators, n_jobs):
    bagging_classifier = BaggingClassifier(base_estimator, n_estimators, n_jobs=n_jobs)

    assert bagging_classifier.estimator == base_estimator
    assert bagging_classifier.n_estimators == n_estimators
    assert bagging_classifier.n_jobs == n_jobs
    assert not bagging_classifier._fitted

    with pytest.raises(ValueError, match="invalid"):
        BaggingClassifier(None, n_estimators)

    with pytest.raises(ValueError, match="estimators"):
        BaggingClassifier(base_estimator, 0)


@pytest.mark.parametrize(
    "base_estimator", [DecisionTreeClassifier(), LogisticRegression()]
)
def test_bagging_classifier_str(base_estimator):
    bagging_classifier = BaggingClassifier(base_estimator)
    assert str(bagging_classifier) == "BaggingClassifier"


@pytest.mark.parametrize(
    "base_estimator, n_estimators, n_jobs",
    [
        (DecisionTreeClassifier(max_depth=3), 5, None),
        (LogisticRegression(), 10, -1),
        (DecisionTreeClassifier(max_depth=5), 3, 2),
        (LogisticRegression(), 7, None),
    ],
)
def test_bagging_classifier_fit_predict(base_estimator, n_estimators, n_jobs):
    bagging_classifier = BaggingClassifier(base_estimator, n_estimators, n_jobs=n_jobs)

    X, y = generate_classification_dataset()

    bagging_classifier.fit(X, y)

    assert bagging_classifier._fitted

    predictions = bagging_classifier.predict(X)
    assert predictions.shape[0] == X.shape[0]
    assert all(isinstance(pred, np.ndarray) for pred in predictions)


@pytest.mark.parametrize(
    "base_estimator", [DecisionTreeClassifier(), LogisticRegression()]
)
def test_bagging_classifier_predict_without_fit(base_estimator):
    bagging_classifier = BaggingClassifier(base_estimator)
    X, _ = generate_classification_dataset()

    with pytest.raises(Exception, match="not fitted"):
        bagging_classifier.predict(X)


@pytest.mark.parametrize(
    "base_estimator, n_estimators, n_jobs",
    [
        (DecisionTreeClassifier(max_depth=3), 5, -1),
        (LogisticRegression(), 10, 4),
        (DecisionTreeClassifier(max_depth=5), 3, 2),
        (LogisticRegression(), 7, None),
    ],
)
def test_bagging_classifier_parallel_fit_predict(base_estimator, n_estimators, n_jobs):
    bagging_classifier = BaggingClassifier(base_estimator, n_estimators, n_jobs=n_jobs)

    X, y = generate_classification_dataset()

    bagging_classifier.fit(X, y)

    assert bagging_classifier._fitted

    predictions = bagging_classifier.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)
