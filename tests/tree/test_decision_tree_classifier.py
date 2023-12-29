import numpy as np
import pytest
from dataset.classification import generate_classification_dataset
from metrics.classification_metrics import accuracy_score
from tree.decision_tree_classifier import DecisionTreeClassifier

@pytest.mark.parametrize("max_depth, method, n_jobs, split", [
    (3, "gini", 2, "best"),
    (5, "entropy", None, "random"),
    (None, "gini", -1, "best"),
    # Add more combinations as needed
])
def test_initialization(max_depth, method, n_jobs, split):
    model = DecisionTreeClassifier(max_depth=max_depth, method=method, n_jobs=n_jobs, split=split)
    assert model.max_depth == max_depth
    assert model.method == method
    assert model.split == split
    assert not model._fitted

    if method not in ["gini", "entropy"]:
        with pytest.raises(ValueError):
            DecisionTreeClassifier(method=method)

    if split not in ["best", "random"]:
        with pytest.raises(ValueError):
            DecisionTreeClassifier(split=split)

@pytest.mark.parametrize("max_depth", [2, 3, 5, None])
def test_fit(max_depth):
    X, y = generate_classification_dataset()
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)
    assert model._fitted
    assert model.root is not None

@pytest.mark.parametrize("max_depth", [2, 3, 5, None])
def test_fit_predict(max_depth):
    X, y = generate_classification_dataset(200)
    model = DecisionTreeClassifier(max_depth=max_depth)
    model.fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == y.shape
    assert accuracy_score(y, model.predict(X)) > 0.9

def test_predict_unfitted():
    model = DecisionTreeClassifier(max_depth=2)
    with pytest.raises(ValueError):
        model.predict(np.array([[1, 2]]))
