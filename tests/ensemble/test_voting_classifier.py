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
            ("DecisionTree", DecisionTreeClassifier()),
            ("LogisticRegression", LogisticRegression()),
        ],
        [
            ("DecisionTree", DecisionTreeClassifier()),
            ("LogisticRegression", LogisticRegression()),
        ],
        [
            ("LogisticRegression", LogisticRegression()),
            ("DecisionTree", DecisionTreeClassifier()),
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
    assert all(isinstance(pred, np.ndarray) for pred in predictions)


def test_voting_classifier_str():
    base_estimators = [("DecisionTree", DecisionTreeClassifier()), ("SVC", SVC())]
    voting_classifier = VotingClassifier(base_estimators)
    assert str(voting_classifier) == "VotingClassifier"


def test_voting_classifier_predict_without_fit():
    base_estimators = [("DecisionTree", DecisionTreeClassifier()), ("SVC", SVC())]
    voting_classifier = VotingClassifier(base_estimators)
    X, _ = generate_classification_dataset()

    with pytest.raises(Exception, match="not fitted"):
        voting_classifier.predict(X)
