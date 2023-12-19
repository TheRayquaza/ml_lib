import numpy as np
import matplotlib.pyplot as plt
from dataset.classification import generate_classification_dataset
from tree.decision_tree_classifier import DecisionTreeClassifier
from ensemble.random_forest_classifier import RandomForestClassifier

from composition.pipeline import Pipeline
from preprocessing.scalers import StandardScaler

from graphic.tree_graphivz import visualize_tree
from graphic.curve import (
    plot_2d_classification,
    plot_decision_boundary,
    plot_3d_classification,
)
from metrics.classification_metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
from model_selection import train_test_split

random_forest_2 = RandomForestClassifier(n_estimators=2, max_depth=5, n_jobs=-1)
random_forest_10 = RandomForestClassifier(n_estimators=10, max_depth=5, n_jobs=-1)
decision_tree = DecisionTreeClassifier(max_depth=10, n_jobs=-1)

pipelines = [decision_tree, random_forest_2, random_forest_10]

X, y = generate_classification_dataset(n_samples=100, n_features=2, n_classes=2)

fig, axes = plt.subplots(1 + 2 * len(pipelines))
plt.legend("upper right")

plot_2d_classification(X, y, fig=fig, ax=axes[0], title="Original Dataset")

for i in range(len(pipelines)):
    model = pipelines[i]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print(model, ":")
    print(f"\tAccuracy: {accuracy_score(y_pred, y_test, percent=True)}%")
    print(f"\tPrecision: {precision_score(y_pred, y_test)}")
    print(f"\tRecall: {recall_score(y_pred, y_test)}")
    print(f"\tF1: {f1_score(y_pred, y_test)}")
    plot_2d_classification(
        X, model.predict(X), ax=axes[2 * i + 1], title=f"Model N°{i + 1}"
    )
    plot_decision_boundary(X, y, model, ax=axes[2 * i + 2])
    print("")

visualize_tree(decision_tree.root)

plt.show()
