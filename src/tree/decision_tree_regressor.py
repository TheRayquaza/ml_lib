from classes.model import Model
from tree.tree_node import TreeNode
import numpy as np
from metrics.regression_metrics import mse, mae, rmse
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
import random


class DecisionTreeRegressor(Model):
    def __init__(self, max_depth=None, method="mse", n_jobs=None, split="best", random_state=None, name="DecisionTreeRegressor"):
        """
        Initializes the Decision Tree Regressor model with specified parameters.

        Parameters
        ----------
            max_depth (int): The maximum depth of the tree.
            method (str): The impurity calculation method.
            n_jobs (int): Number of parallel jobs to run during prediction.
            split (str): The split method.
            random_state (int): If given, allow reproducibility
            name (str): The name given to the model
        """
        super().__init__(random_state=random_state, name=name)
    
        self.root = None
        self.max_depth = max_depth
        self.method = method
        self.split = split
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs

        if not split in ["best", "random"]:
            raise ValueError("DecisionTreeRegressor: Unknown split method", method)
        if not method in ["mse", "mae", "rmse"]:
            raise ValueError("DecisionTreeRegressor: Unknown method", method)

    def _calculate_impurity(self, y: np.ndarray) -> float:
        """
        Calculates impurity of a set of target values.

        Parameters
        ----------
            y (np.ndarray): The target values.

        Returns
        -------
            float: Impurity value.
        """
        y_pred = np.zeros(y.shape)
        mean = np.mean(y)
        for i in range(y_pred.shape[0]):
            y_pred[i] = mean
        if self.method == "mse":
            return mse(y_pred, y)
        elif self.method == "mae":
            return mae(y_pred, y)
        else:
            return rmse(y_pred, y)

    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth=None):
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
            return TreeNode(X, y, 0, 0, 0)

        feature, value, impurity_reduction = 0, 0, 0
        if self.split == "best":
            feature, value, impurity_reduction = self._find_best_split(X, y)
        else:
            feature, value, impurity_reduction = self._find_random_split(X, y)
        root = TreeNode(X, y, impurity_reduction, feature, value)
        X_left, y_left, X_right, y_right = (
            np.zeros((0, self.samples)),
            np.zeros((0)),
            np.zeros((0, self.samples)),
            np.zeros((0)),
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

    def _find_random_split(self, X: np.ndarray, y: np.ndarray):
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
        rd_feature = np.random.randint(0, self.features)
        rd_index = np.random.randint(0, self.samples)
        rd_split_value = self.X[rd_index, rd_feature]

        impurity = self._calculate_impurity(y)

        left_indices = X[:, rd_feature] <= rd_split_value
        right_indices = ~left_indices
        y_left = y[left_indices]
        y_right = y[right_indices]

        impurity_left = self._calculate_impurity(y_left)
        impurity_right = self._calculate_impurity(y_right)

        weighted_impurity = (y_left.shape[0] / y.shape[0]) * impurity_left + (
            y_right.shape[0] / y.shape[0]
        ) * impurity_right
        rd_impurity_reduction = impurity - weighted_impurity

        return rd_feature, rd_split_value, rd_impurity_reduction

    def _find_best_split(self, X: np.ndarray, y: np.ndarray):
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

        for feature in range(self.features):
            for v in np.unique(X[:, feature]):
                left_indices = X[:, feature] <= v
                right_indices = ~left_indices
                y_left = y[left_indices]
                y_right = y[right_indices]

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

    def _make_prediction(self, X: np.ndarray):
        """
        Makes a prediction using the decision tree.

        Parameters
        ----------
            X (np.ndarray): The input features.

        Returns
        -------
            Any: The predicted value.
        """
        current = self.root
        while not current.is_terminal:
            if X[current.feature] > current.value:
                current = current.right
            else:
                current = current.left
        return current.select_value()

    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fits the decision tree model to the training data.

        Parameters
        ----------
            X (np.ndarray): The input features.
            y (np.ndarray): The target values.

        Returns
        -------
            DecisionTreeRegressor: The fitted model.
        """
        super().fit(X, y)
        self.root = self._build_tree(X, y, self.max_depth)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predicts the target values for input features.

        Parameters
        ----------
            X (np.ndarray): The input features.

        Returns
        -------
            np.ndarray: The predicted target values.
        """
        super().predict(X)
        
        samples = X.shape[0]
        result = np.zeros((samples))

        if not self.n_jobs:
            for i in range(samples):
                result[i] = self._make_prediction(X[i])
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self._make_prediction, X[i]): i for i in range(samples)
            }
            for future in as_completed(future_to_pred):
                result[future_to_pred[future]] = future.result()
        return result
