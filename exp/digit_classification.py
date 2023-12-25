from sklearn.datasets import load_digits
from metrics.classification_metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from model_selection.splitter import train_test_split

from ensemble.random_forest_classifier import RandomForestClassifier
from ensemble.bagging_classifier import BaggingClassifier
from tree.decision_tree_classifier import DecisionTreeClassifier
from neural_net.deep_neural_net import DeepNeuralNetwork

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y.reshape(-1, 1), random_state=42
)

models = [
    DeepNeuralNetwork(
        n_features=X.shape[1], layers=[X.shape[1], 64, 9], task="classifcation"
    )
]

for model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print(model, ":")
    print(f"\tAccuracy: {accuracy_score(y_pred, y_test, percent=True)}%")
    print(f"\tPrecision: {precision_score(y_pred, y_test)}")
    print(f"\tRecall: {recall_score(y_pred, y_test)}")
    print(f"\tF1: {f1_score(y_pred, y_test)}")
    print("")
