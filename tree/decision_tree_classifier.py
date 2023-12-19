from classes.model import Model
from tree.tree_node import TreeNode
import numpy as np
from metrics.classification_impurity import (
    gini_impurity,
    classification_impurity,
    entropy_impurity,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class DecisionTreeClassifier(Model):
    def __init__(self, max_depth=None, method="gini", n_jobs=None):
        self.root = None
        self.max_depth = max_depth
        self.method = method
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        if not method in ["gini", "entropy", "classification"]:
            raise ValueError("DecisionTreeClassifier: Unknown method", method)

    def calculate_impurity(self, y: np.array) -> float:
        uniques, counts = np.unique(y, return_counts=True)
        if self.method == "gini":
            return gini_impurity(uniques, counts)
        elif self.method == "entropy":
            return entropy_impurity(uniques, counts)
        else:
            return classification_impurity(uniques, counts)

    def __build_tree(self, X: np.array, y: np.array, depth=None):
        uniques = np.unique(y)
        if depth == 0 or len(uniques) <= 1:
            return TreeNode(X, y, None, None, None)

        feature, value, impurity_reduction = self.__find_best_split(X, y)
        root = TreeNode(X, y, impurity_reduction, feature, value)
        X_left, y_left, X_right, y_right = (
            np.zeros((0, X.shape[1])),
            np.zeros((0, 1)),
            np.zeros((0, X.shape[1])),
            np.zeros((0, 1)),
        )
        for i in range(X.shape[0]):
            if X[i, feature] <= value:
                X_left = np.vstack([X_left, X[i : i + 1, :]])
                y_left = np.vstack([y_left, y[i : i + 1]])
            else:
                X_right = np.vstack([X_right, X[i : i + 1, :]])
                y_right = np.vstack([y_right, y[i : i + 1]])
        root.left = self.__build_tree(X_left, y_left, depth - 1 if depth else depth)
        root.right = self.__build_tree(X_right, y_right, depth - 1 if depth else depth)
        return root

    def __find_best_split(self, X: np.array, y: np.array):
        best_feature = None
        best_split_value = None
        best_impurity_reduction = None

        impurity = self.calculate_impurity(y)

        for feature in range(X.shape[1]):
            for v in np.unique(X[:, feature]):
                left_indices = X[:, feature] <= v
                right_indices = ~left_indices
                y_left = y[left_indices].reshape(-1, 1)
                y_right = y[right_indices].reshape(-1, 1)

                impurity_left = self.calculate_impurity(y_left)
                impurity_right = self.calculate_impurity(y_right)

                weighted_impurity = (y_left.shape[0] / y.shape[0]) * impurity_left + (
                    y_right.shape[0] / y.shape[0]
                ) * impurity_right
                impurity_reduction = impurity - weighted_impurity

                if (
                    not best_impurity_reduction
                    or impurity_reduction > best_impurity_reduction
                ):
                    best_impurity_reduction = impurity_reduction
                    best_feature = feature
                    best_split_value = v
        return best_feature, best_split_value, best_impurity_reduction

    def __make_prediction(self, X: np.array):
        current = self.root
        while not current.is_terminal:
            if X[current.feature] > current.value:
                current = current.right
            else:
                current = current.left
        return current.select_value()

    def fit(self, X: np.array, y: np.array):
        self.root = self.__build_tree(X, y, self.max_depth)
        return self

    def predict(self, X: np.array) -> np.array:
        result = np.zeros((X.shape[0], 1))
        if not self.n_jobs:
            for i in range(X.shape[0]):
                result[i] = self.__make_prediction(X[i])
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self.__make_prediction, X[i]): i for i in range(X.shape[0])
            }
            for future in as_completed(future_to_pred):
                result[future_to_pred[future]] = future.result()
        return result
