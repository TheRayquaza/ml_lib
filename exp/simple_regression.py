import numpy as np
import matplotlib.pyplot as plt

from neural_net.deep_neural_net import (
    DeepNeuralNetwork,
    relu,
    relu_prime,
    sigmoid,
    sigmoid_prime,
    tanh,
    tanh_prime,
    id,
    id_prime,
)
from linear.lasso_regression import LassoRegression
from linear.ridge_regression import RidgeRegression
from knn.knn_regressor import KNeighborsRegressor

from preprocessing.scalers import StandardScaler, Normalizer
from preprocessing.features_creation import PolynomialFeatures

from composition.pipeline import Pipeline

from graphic.curve import plot_2d_curve
from dataset.generic import generate_linear_dataset, generate_polynomial_dataset
from metrics.regression_metrics import mse
from model_selection.splitter import train_test_split

layers = [1, 2, 1]
features = 1

pipelines = [
    Pipeline(
        steps=[("Scaler", StandardScaler()), ("LassoRegression", LassoRegression())]
    ),
    Pipeline(
        steps=[
            ("StandardScaler", StandardScaler()),
            ("KNN", KNeighborsRegressor(n_jobs=-1)),
        ]
    ),
]

linear_datasets = [
    generate_linear_dataset(250),
    generate_linear_dataset(250, slope=5.0),
]
poly_datasets = [
    generate_polynomial_dataset(250, degree=2),
]

datasets = [
    ("Linear Dataset 1", linear_datasets[0]),
    ("Linear Dataset 2", linear_datasets[1]),
    ("Polynomial Dataset 2", poly_datasets[0]),
]

for dataset_name, (X, y) in datasets:
    print(f"\nDataset: {dataset_name}\n{'=' * (len(dataset_name) + 10)}")

    fig, axes = plt.subplots(1, len(pipelines) + 1, figsize=(15, 10))
    axes[0].scatter(X, y, label="Original Data", color="black")
    axes[0].set_title("Original Data")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("y")
    axes[0].legend()

    for i, pipeline in enumerate(pipelines, start=1):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Training
        y_train_pred = pipeline.fit_predict(X_train, y_train)

        # Predictions
        y_test_pred = pipeline.predict(X_test)

        # Metrics
        mse_train = mse(y_train_pred, y_train)
        mse_test = mse(y_test_pred, y_test)

        print(f"{pipeline} (Dataset: {dataset_name}):")
        print(f"\tMSE on train: {mse_train:.4f}")
        print(f"\tMSE on test: {mse_test:.4f}")
        print("")

        # Plot the fitted curve
        plot_2d_curve(X, pipeline.predict(X), ax=axes[i], name=pipeline)

    plt.tight_layout()
    plt.show()
