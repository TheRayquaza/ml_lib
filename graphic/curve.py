import matplotlib.pyplot as plt
import numpy as np
from classes.model import Model


def plot_2d_curve(X: np.array, y: np.array, ax=None, name=None):
    if not ax:
        fig, ax = plt.subplots()
    if not name:
        name = "2D Curve"
    ax.set_title(name)
    ax.scatter(X, y)


def plot_2d_classification(X: np.array, y: np.array, fig=None, ax=None, title=None):
    if X.shape[0] != y.shape[0]:
        raise Exception("plot_2d_classification: invalid shapes")

    if not fig and not ax:
        fig = plt.figure()
    if not ax:
        ax = fig.add_subplot()
    if not title:
        title = "2D Classification"

    distincts = np.unique(y)
    for v in distincts:
        indexes = np.where(y == v)
        ax.scatter(X[indexes, 0], X[indexes, 1], label=f"Class {v}")

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")


def plot_3d_classification(X: np.array, y: np.array, ax=None, fig=None, title=None):
    if X.shape[0] != y.shape[0]:
        raise Exception("plot_2d_classification: invalid shapes")

    if not fig and not ax:
        fig = plt.figure()
    if not ax:
        ax = fig.add_subplot(projection="3d")
    if not title:
        title = "3D Classification"

    distincts = np.unique(y)
    for v in distincts:
        indexes = np.where(y == v)
        ax.scatter(X[indexes, 0], X[indexes, 1], X[indexes, 2], label=f"Class {v}")

    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_zlabel("Feature 3")


def plot_decision_boundary(X: np.array, y: np.array, model: Model, ax=None, fig=None):
    if not fig and not ax:
        fig = plt.figure()
    if not ax:
        ax = fig.add_subplot()

    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    y_min, y_max = X[:, 1].min(), X[:, 1].max()
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.3, cmap="viridis")
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap="viridis", edgecolors="k", marker="o", s=100)
