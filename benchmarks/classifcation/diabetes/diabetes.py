from linear.logistic import LogisticRegression
from tree.decision_tree_classifier import DecisionTreeClassifier
from ensemble.random_forest_classifier import RandomForestClassifier
from ensemble.bagging_classifier import BaggingClassifier
from ensemble.voting_classifier import VotingClassifier
from knn.knn_classifier import KNeighborsClassifier

from model_selection.splitter import train_test_split
from preprocessing.scalers import StandardScaler
from metrics.classification_metrics import f1_score, accuracy_score, precision_score, recall_score

import pandas as pd
import datetime
import os

current_directory = os.path.dirname(__file__)

filename = "diabetes.csv"
full_path = os.path.join(current_directory, filename)

headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "Training Time (s)"]

df = pd.read_csv(full_path)
X, y = df.drop(columns=["Outcome"]), df["Outcome"]
y = y.to_numpy().reshape(-1, 1)
X = StandardScaler().fit_transform(X.to_numpy())
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


models = [
    LogisticRegression(learning_rate=1e-4),
    LogisticRegression(learning_rate=1e-2),
    KNeighborsClassifier(k=5, n_jobs=-1),
    KNeighborsClassifier(k=10, n_jobs=-1),
    KNeighborsClassifier(k=10, n_jobs=-1),
    DecisionTreeClassifier(max_depth=None, n_jobs=-1, method="gini"),
    DecisionTreeClassifier(max_depth=None, n_jobs=-1, method="entropy"),
    DecisionTreeClassifier(max_depth=10, n_jobs=-1, method="gini"),
    DecisionTreeClassifier(max_depth=10, n_jobs=-1, method="entropy"),
    RandomForestClassifier(max_depth=10, n_jobs=-1, method="gini"),
    RandomForestClassifier(max_depth=10, n_jobs=-1, method="entropy"),
    RandomForestClassifier(n_estimators=10, n_jobs=-1),
    RandomForestClassifier(n_estimators=20, n_jobs=-1),
    BaggingClassifier(estimator=DecisionTreeClassifier(), n_jobs=-1),
    VotingClassifier(estimators=[("DecisionTreeClassifier", DecisionTreeClassifier()), ("LogisticRegression", LogisticRegression())], n_jobs=-1)
]

def run_benchmark():
    results = []
    for model in models:
        print(f"Benchmark with model {model}", end="")
        start_time = datetime.datetime.now()
        model.fit(X_train, y_train)
        train_time = datetime.datetime.now() - start_time
        y_pred = model.predict(X_test)
        f1 = f1_score(y_test, y_pred, average="macro")
        recall = recall_score(y_test, y_pred, average="macro")
        precision = precision_score(y_test, y_pred, average="macro")
        accuracy = accuracy_score(y_test, y_pred)
        results.append((model, accuracy, precision, recall, f1, train_time))
        print(" [OK]")
    return results

if __name__ == "__main__":
    results = run_benchmark()
    df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Training Time (s)'])
    df.to_csv('result.csv', index=False)
    print(df)