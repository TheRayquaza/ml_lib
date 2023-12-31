import numpy as np
import pytest
from dataset.classification import generate_classification_dataset
from ensemble.voting_classifier import VotingClassifier
from tree.decision_tree_classifier import DecisionTreeClassifier
from linear.logistic import LogisticRegression


@pytest.mark.parametrize(
    "base_estimators",
    [
        [
            ("DecisionTreeClassifier", DecisionTreeClassifier(n_jobs=-1)),
            ("LogisticRegression", LogisticRegression()),
        ],
        [
            ("DecisionTreeClassifier", DecisionTreeClassifier(method="gini", n_jobs=-1)),
            ("DecisionTreeClassifier", DecisionTreeClassifier(method="entropy", n_jobs=-1)),
            ("LogisticRegression", LogisticRegression()),
        ],
        [
            ("LogisticRegression", LogisticRegression(method="stochastic")),
            ("LogisticRegression", LogisticRegression(method="default")),
        ],
    ],
)
def test_voting_classifier_init_fit_predict(base_estimators):
    voting_classifier = VotingClassifier(base_estimators)

    assert voting_classifier.n_estimators == len(base_estimators)
    assert voting_classifier.n_jobs is None
    assert voting_classifier.bootstrap is True
    assert not voting_classifier._fitted

    X, y = generate_classification_dataset()
    voting_classifier.fit(X, y)

    assert voting_classifier._fitted

    predictions = voting_classifier.predict(X)
    assert len(predictions) == len(X)


@pytest.mark.parametrize(
    "invalid_estimators",
    [
        [],
    ],
)
def test_voting_classifier_invalid_base_estimators(invalid_estimators):
    with pytest.raises(ValueError):
        VotingClassifier(invalid_estimators)


def test_voting_classifier_predict_without_fit():
    base_estimators = [("DecisionTree", DecisionTreeClassifier())]
    voting_classifier = VotingClassifier(base_estimators)
    X, _ = generate_classification_dataset()

    with pytest.raises(Exception, match="not fitted"):
        voting_classifier.predict(X)


@pytest.mark.parametrize(
    "base_estimators",
    [
        [
            ("DecisionTree1", DecisionTreeClassifier()),
            ("LogisticRegression", LogisticRegression()),
        ],
        [
            ("DecisionTree1", DecisionTreeClassifier()),
            ("DecisionTree2", DecisionTreeClassifier()),
            ("LogisticRegression", LogisticRegression()),
        ],
    ],
)
def test_voting_classifier_parallel_fit_predict(base_estimators):
    voting_classifier = VotingClassifier(base_estimators, n_jobs=-1)
    X, y = generate_classification_dataset()

    voting_classifier.fit(X, y)

    assert voting_classifier._fitted

    predictions = voting_classifier.predict(X)
    assert len(predictions) == len(X)
