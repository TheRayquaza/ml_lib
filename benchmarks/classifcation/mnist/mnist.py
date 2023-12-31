from sklearn.datasets import load_digits
from metrics.classification_metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from model_selection.splitter import train_test_split

from linear.logistic import LogisticRegression
from knn.knn_classifier import KNeighborsClassifier
from ensemble.random_forest_classifier import RandomForestClassifier
from ensemble.bagging_classifier import BaggingClassifier
from ensemble.voting_classifier import VotingClassifier
from tree.decision_tree_classifier import DecisionTreeClassifier

import pandas as pd
import numpy as np
import datetime

def _distance_manhattan(X1, X2):
    return np.abs(np.sum(X1 - X2))

X, y = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y.reshape(-1, 1), random_state=42
)

models = [
    KNeighborsClassifier(k=5, n_jobs=-1),
    KNeighborsClassifier(k=10, n_jobs=-1),
    KNeighborsClassifier(k=10, n_jobs=-1, distance_method=_distance_manhattan),
    DecisionTreeClassifier(max_depth=None, n_jobs=-1, method="gini"),
    DecisionTreeClassifier(max_depth=None, n_jobs=-1, method="entropy"),
    DecisionTreeClassifier(max_depth=10, n_jobs=-1, method="gini"),
    DecisionTreeClassifier(max_depth=10, n_jobs=-1, method="entropy"),
    RandomForestClassifier(max_depth=10, n_jobs=-1, method="gini"),
    RandomForestClassifier(max_depth=10, n_jobs=-1, method="entropy"),
    RandomForestClassifier(n_estimators=5, n_jobs=-1, max_depth=10),
    RandomForestClassifier(n_estimators=10, n_jobs=-1, max_depth=10),
    BaggingClassifier(estimator=DecisionTreeClassifier(), n_jobs=-1),
    VotingClassifier(estimators=[("DecisionTreeClassifier", DecisionTreeClassifier(max_depth=10)), ("KNeighborsClassifier", KNeighborsClassifier())], n_jobs=-1)
]

def run_benchmark():
    results = []
    for model in models:
        print(f"Benchmark with model {model}", end="")
        start_time = datetime.datetime.now()
        model.fit(X_train, y_train)
        train_time = datetime.datetime.now() - start_time
        start_time = datetime.datetime.now()
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='macro')
        prediction_time = datetime.datetime.now() - start_time
        recall = recall_score(y_test, y_pred, average='macro')
        precision = precision_score(y_test, y_pred, average='macro')
        accuracy = accuracy_score(y_test, y_pred)
        results.append((str(model), accuracy, precision, recall, f1, train_time.total_seconds(), prediction_time.total_seconds()))
        print(" [OK]")
    return results

if __name__ == "__main__":
    results = run_benchmark()
    df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)', 'Prediction Time (s)'])
    df.to_csv('result.csv', index=False)
    print(df)