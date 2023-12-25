import numpy as np
import matplotlib.pyplot as plt
from dataset.classification import generate_classification_dataset

from neural_net.deep_neural_net import DeepNeuralNetwork, softmax

from graphic.curve import plot_2d_classification, plot_decision_boundary
from metrics.classification_metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
)
from model_selection.splitter import train_test_split

features = 2
classes = 3

dnn = DeepNeuralNetwork(
    layers=[features, 10, classes],
    n_features=features,
    method="mini-batch",
    batch_size=10,
    epochs=10,
    last_activation=softmax,
)
pipelines = [dnn]

X, y = generate_classification_dataset(
    n_samples=250, n_features=features, n_classes=classes
)

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

plt.show()
