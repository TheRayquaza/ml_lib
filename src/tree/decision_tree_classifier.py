from classes.model import Model
from tree.tree_node import TreeNode
import numpy as np
import random
from metrics.classification_impurity import (
    gini_impurity,
    classification_impurity,
    entropy_impurity,
)
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class DecisionTreeClassifier(Model):
    def __init__(
        self,
        max_depth=None,
        method="gini",
        n_jobs=None,
        split="best",
        random_state=None,
    ):
        """
        Initializes the Decision Tree Classifier model with specified parameters.

        Parameters
        ----------
            max_depth (int): The maximum depth of the tree.
            method (str): The impurity calculation method.
            n_jobs (int): Number of parallel jobs to run during prediction.
            split (str): The split method.
            random_state (int): Seed for the random number generator.
        """
        random.seed(random_state)
        self.root = None
        self.max_depth = max_depth
        self.method = method
        self.split = split
        self._fitted = False
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        if split not in ["best", "random"]:
            raise ValueError(f"DecisionTreeClassifier: Unknown split method {split}")
        if method not in ["gini", "entropy", "classification"]:
            raise ValueError(f"DecisionTreeClassifier: Unknown method {method}")

    def __str__(self):
        """String representation of the DecisionTreeClassifier class."""
        return "DecisionTreeClassifier"

    def _calculate_impurity(self, y: np.array) -> float:
        """
        Calculates impurity of a set of target values.

        Parameters
        ----------
            y (np.ndarray): The target values.

        Returns
        -------
            float: Impurity value.
        """
        uniques, counts = np.unique(y, return_counts=True)
        if self.method == "gini":
            return gini_impurity(uniques, counts)
        elif self.method == "entropy":
            return entropy_impurity(uniques, counts)
        else:
            return classification_impurity(uniques, counts)

    def _build_tree(self, X: np.array, y: np.array, depth=None):
        """
        Recursively builds the decision tree.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.
            depth (int): The current depth of the tree.

        Returns
        -------
            TreeNode: The root node of the tree.
        """
        uniques = np.unique(y)
        if depth == 0 or len(uniques) <= 1:
            return TreeNode(X, y, None, None, None)

        feature, value, impurity_reduction = 0, 0, 0
        if self.split == "best":
            feature, value, impurity_reduction = self._find_best_split(X, y)
        else:
            feature, value, impurity_reduction = self._find_random_split(X, y)
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
        root.left = self._build_tree(X_left, y_left, depth - 1 if depth else depth)
        root.right = self._build_tree(X_right, y_right, depth - 1 if depth else depth)
        return root

    def _find_random_split(self, X: np.array, y: np.array):
        """
        Finds a random split for the decision tree.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.

        Returns
        -------
            Tuple[int, float, float]: Tuple containing feature, split value, and impurity reduction.
        """
        rd_feature = random.randint(0, X.shape[1] - 1)
        rd_value = random.randint(0, X.shape[0] - 1)
        rd_split_value = X[rd_value, rd_feature]

        impurity = self._calculate_impurity(y)

        left_indices = X[:, rd_feature] <= rd_split_value
        right_indices = ~left_indices
        y_left = y[left_indices].reshape(-1, 1)
        y_right = y[right_indices].reshape(-1, 1)

        impurity_left = self._calculate_impurity(y_left)
        impurity_right = self._calculate_impurity(y_right)

        weighted_impurity = (y_left.shape[0] / y.shape[0]) * impurity_left + (
            y_right.shape[0] / y.shape[0]
        ) * impurity_right
        rd_impurity_reduction = impurity - weighted_impurity

        return rd_feature, rd_split_value, rd_impurity_reduction

    def _find_best_split(self, X: np.array, y: np.array):
        """
        Finds the best split for the decision tree.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.

        Returns
        -------
            Tuple[int, float, float]: Tuple containing feature, split value, and impurity reduction.
        """
        best_feature = None
        best_split_value = None
        best_impurity_reduction = None

        impurity = self._calculate_impurity(y)

        for feature in range(X.shape[1]):
            for v in np.unique(X[:, feature]):
                left_indices = X[:, feature] <= v
                right_indices = ~left_indices
                y_left = y[left_indices].reshape(-1, 1)
                y_right = y[right_indices].reshape(-1, 1)

                impurity_left = self._calculate_impurity(y_left)
                impurity_right = self._calculate_impurity(y_right)

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

    def fit(self, X: np.array, y: np.array):
        """
        Fits the decision tree to the training data.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.

        Returns
        -------
            DecisionTreeClassifier: The fitted model.
        """
        self.X = X
        self.y = y
        self.root = self._build_tree(X, y, self.max_depth)
        self._fitted = True
        return self

    def _make_prediction(self, X: np.array):
        """
        Makes a prediction for a single input using the decision tree.

        Parameters
        ----------
            X (np.ndarray): The input features.

        Returns
        -------
            int: The predicted target value.
        """
        current = self.root
        while not current.is_terminal:
            if X[current.feature] > current.value:
                current = current.right
            else:
                current = current.left
        return current.select_value()

    def predict(self, X: np.array) -> np.array:
        """
        Predicts the target values for a set of input features.

        Parameters
        ----------
            X (np.ndarray): The input features.

        Returns
        -------
            np.ndarray: The predicted target values.
        """
        if not self._fitted:
            raise ValueError("DecisionTreeClassifier: model not fitted with data")
        result = np.zeros((X.shape[0], 1))
        if not self.n_jobs:
            for i in range(X.shape[0]):
                result[i] = self._make_prediction(X[i])
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self._make_prediction, X[i]): i for i in range(X.shape[0])
            }
            for future in as_completed(future_to_pred):
                result[future_to_pred[future]] = future.result()
        return result
