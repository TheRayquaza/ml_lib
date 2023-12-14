import numpy as np
import matplotlib.pyplot as plt
from regression.elasticnet import ElasticNet
from regression.lasso import Lasso
from regression.linear import LinearModel
from regression.ridge import Ridge
from preprocessing.scalers import StandardScaler, Normalizer
from preprocessing.features_creation import PolynomialFeatures
from composition.pipeline import Pipeline
from graphic.curve import plot_2d_curve
from dataset.generic import generate_linear_dataset, generate_polynomial_dataset
from metrics import mse
from model_selection import train_test_split

pipelines = [
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("feature_creator", PolynomialFeatures(degree=2)),
            ("model", LinearModel()),
        ]
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("feature_creator", PolynomialFeatures(degree=2)),
            ("model", ElasticNet()),
        ]
    ),
    Pipeline(
        [
            ("scaler", Normalizer()),
            ("feature_creator", PolynomialFeatures(degree=2)),
            ("model", Lasso()),
        ]
    ),
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("feature_creator", PolynomialFeatures(degree=2)),
            ("model", Ridge()),
        ]
    ),
]

# Rest of the code remains the same


linear_datasets = [
    generate_linear_dataset(3000),
    generate_linear_dataset(3000, slope=5.0),
]
poly_datasets = [
    generate_polynomial_dataset(3000, degree=2),
    generate_polynomial_dataset(3000, degree=3),
    generate_polynomial_dataset(3000, degree=5)
]

datasets = [
    ("Linear Dataset 1", linear_datasets[0]),
    ("Linear Dataset 2", linear_datasets[1]),
    ("Polynomial Dataset 2", poly_datasets[0]),
    ("Polynomial Dataset 3", poly_datasets[1]),
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
        pipeline.fit(X_train, y_train)

        # Predictions
        y_train_pred = pipeline.predict(X_train)
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
