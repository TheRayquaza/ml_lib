import numpy as np
import matplotlib.pyplot as plt
from dataset.classification import generate_classification_dataset
from classification.logistic import LogisticRegression
from composition.pipeline import Pipeline
from preprocessing.scalers import StandardScaler, Normalizer
from preprocessing.features_creation import PolynomialFeatures
from graphic.curve import plot_2d_classification, plot_decision_boundary
from metrics.classification_metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
from model_selection import train_test_split

pipelines = [
    Pipeline(
        steps=[
            ("StandardScaler", StandardScaler()),
            ("LogisticRegression", LogisticRegression()),
        ]
    ),
]

X, y = generate_classification_dataset(n_samples=100, n_features=2, n_classes=2)

fig, axes = plt.subplots(1, 2 * len(pipelines) + 1, figsize=(15, 5))
plot_2d_classification(X, y, ax=axes[0], title="Original Dataset")

for i in range(len(pipelines)):
    model = pipelines[i]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model.fit(X_train, y_train)
    print(model, ":")
    print(f"\tAccuracy: {accuracy_score(model.predict(X_test), y_test, percent=True)}%")
    print(f"\tPrecision: {precision_score(model.predict(X_test), y_test)}")
    print(f"\tRecall: {recall_score(model.predict(X_test), y_test)}")
    print(f"\tF1: {f1_score(model.predict(X_test), y_test)}")
    plot_2d_classification(X, model.predict(X), ax=axes[i + 1], title=f"Model N°{i+1}")
    plot_decision_boundary(X, y, model, ax=axes[i + 2])
    print("")

plt.show()
