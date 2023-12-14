import matplotlib.pyplot as plt
import numpy as np
from math import log10


def plot_2d_curve(X: np.array, y: np.array, ax=None, name=None):
    if not ax:
        fig, ax = plt.subplots()
    if name:
        ax.set_title(name)
    ax.scatter(X, y)


def plot_2d_classification(X: np.array, y: np.array, ax=None):
    if X.shape[0] != y.shape[0]:
        raise Exception("plot_2d_classification: invalid shapes")

    if not ax:
        fig, ax = plt.subplots()

    distincts = np.unique(y)
    for v in distincts:
        indexes = np.where(y == v)
        ax.scatter(X[indexes, 0], X[indexes, 1], label=f"Class {v}")

    plt.legend(loc="upper right")
    ax.set_title("2D Classification dataset")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def plot_decision_boundary(trues: np.array, falses: np.array, ax=None):
    if not ax:
        fig, ax = plt.subplots()

    ax.scatter(
        [i for i in range(len(trues))],
        trues,
        s=len(trues),
        c="b",
        marker="o",
        label="Trues",
    )
    ax.scatter(
        [i for i in range(len(falses))],
        falses,
        s=len(falses),
        c="r",
        marker="s",
        label="Falses",
    )

    plt.legend(loc="upper right")
    ax.set_title("Decision Boundary")
    ax.set_ylabel("Predicted Probability")
