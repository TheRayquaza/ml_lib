import numpy as np


class TreeNode:
    def __init__(
        self,
        X: np.array,
        y: np.array,
        ig: float,
        feature: int,
        value: float,
        mode="classification",
        left=None,
        right=None,
    ):
        self.X = X
        self.y = y
        self.ig = ig
        self.feature = feature
        self.value = value
        self.left = left
        self.right = right
        self.mode = mode
        if not mode in ["classification", "regression"]:
            raise ValueError(f"TreeNode: cannot instantiate node having unknown mode {mode}")
        
    def select_value(self):
        if self.mode == "classification":
            uniques, counts = np.unique(self.y, return_counts=True)
            return uniques[np.argmax(counts)]
        else:
            return np.mean(self.y)

    @property
    def is_terminal(self) -> bool:
        return self.left == self.right == None
