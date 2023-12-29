import numpy as np
import pytest
from dataset.classification import generate_classification_dataset
from ensemble.voting_classifier import VotingClassifier
from tree.decision_tree_classifier import DecisionTreeClassifier
from linear.logistic import LogisticRegression


@pytest.mark.parametrize(
    "base_estimators, voting",
    [
        (
            [
                ("DecisionTree1", DecisionTreeClassifier()),
                ("LogisticRegression", LogisticRegression()),
            ],
            "hard",
        ),
        (
            [
                ("DecisionTree1", DecisionTreeClassifier()),
                ("DecisionTree2", DecisionTreeClassifier()),
                ("LogisticRegression", LogisticRegression()),
            ],
            "soft",
        ),
        (
            [
                ("LogisticRegression1", LogisticRegression()),
                ("LogisticRegression2", LogisticRegression()),
            ],
            "hard",
        ),
    ],
)
def test_voting_classifier_init_fit_predict(base_estimators, voting):
    voting_classifier = VotingClassifier(base_estimators, voting=voting)

    assert voting_classifier.n_estimators == len(base_estimators)
    assert voting_classifier.voting == voting
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
        None,
        [("DecisionTree", DecisionTreeClassifier()), (None, None)],
    ],
)
def test_voting_classifier_invalid_base_estimators(invalid_estimators):
    with pytest.raises(ValueError):
        VotingClassifier(invalid_estimators)


def test_voting_classifier_str():
    base_estimators = [("DecisionTree", DecisionTreeClassifier())]
    voting_classifier = VotingClassifier(base_estimators)
    assert str(voting_classifier) == "VotingClassifier"


def test_voting_classifier_predict_without_fit():
    base_estimators = [("DecisionTree", DecisionTreeClassifier())]
    voting_classifier = VotingClassifier(base_estimators)
    X, _ = generate_classification_dataset()

    with pytest.raises(Exception, match="not fitted"):
        voting_classifier.predict(X)


@pytest.mark.parametrize(
    "base_estimators, voting",
    [
        (
            [
                ("DecisionTree1", DecisionTreeClassifier()),
                ("LogisticRegression", LogisticRegression()),
            ],
            "hard",
        ),
        (
            [
                ("DecisionTree1", DecisionTreeClassifier()),
                ("DecisionTree2", DecisionTreeClassifier()),
                ("LogisticRegression", LogisticRegression()),
            ],
            "soft",
        ),
    ],
)
def test_voting_classifier_parallel_fit_predict(base_estimators, voting):
    voting_classifier = VotingClassifier(base_estimators, voting=voting, n_jobs=-1)
    X, y = generate_classification_dataset()

    voting_classifier.fit(X, y)

    assert voting_classifier._fitted

    predictions = voting_classifier.predict(X)
    assert len(predictions) == len(X)
