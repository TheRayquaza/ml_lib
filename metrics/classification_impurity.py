import math
import numpy as np


def gini_impurity(uniques: np.array, counts: np.array) -> float:
    result = 0
    sum_counts = 0
    for i in range(counts.shape[0]):
        sum_counts += counts[i]
    for i in range(uniques.shape[0]):
        result += (counts[i] / sum_counts) ** 2
    return 1 - result


def entropy_impurity(uniques: np.array, counts: np.array) -> float:
    result = 0
    sum_counts = np.sum(counts)
    for i in range(uniques.shape[0]):
        p = counts[i] / sum_counts
        result += p * math.log2(p)
    return -result


def classification_impurity(uniques: np.array, counts: np.array) -> float:
    p = float("-inf")
    sum_counts = np.sum(counts)
    for i in range(uniques.shape[0]):
        p = max(p, counts[i] / sum_counts)
    return 1 - p
