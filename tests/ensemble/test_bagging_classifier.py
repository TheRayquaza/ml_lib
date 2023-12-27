import numpy as np
import pytest
from dataset.classification import generate_classification_dataset
from ensemble.bagging_classifier import BaggingClassifier
from tree.decision_tree_classifier import DecisionTreeClassifier
from linear.logistic import LogisticRegression


@pytest.mark.parametrize(
    "base_estimator", [DecisionTreeClassifier(), LogisticRegression()]
)
def test_bagging_classifier_init(base_estimator):
    n_estimators = 5
    bagging_classifier = BaggingClassifier(base_estimator, n_estimators)

    assert bagging_classifier.estimator == base_estimator
    assert bagging_classifier.n_estimators == n_estimators
    assert bagging_classifier.n_jobs is None
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
    "base_estimator", [DecisionTreeClassifier(), LogisticRegression()]
)
def test_bagging_classifier_fit_predict(base_estimator):
    bagging_classifier = BaggingClassifier(base_estimator)

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
    "base_estimator", [DecisionTreeClassifier(), LogisticRegression()]
)
def test_bagging_classifier_parallel_fit_predict(base_estimator):
    n_estimators = 3
    bagging_classifier = BaggingClassifier(base_estimator, n_estimators, n_jobs=-1)

    X, y = generate_classification_dataset()

    bagging_classifier.fit(X, y)

    assert bagging_classifier._fitted

    predictions = bagging_classifier.predict(X)
    assert len(predictions) == len(X)
    assert all(isinstance(pred, np.ndarray) for pred in predictions)
