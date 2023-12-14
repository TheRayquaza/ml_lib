import numpy as np
import math


def confusion_matrix(y_true: np.array, y_pred: np.array) -> np.array:
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("confusion_matrix: invalid shapes")

    classes = np.unique(np.concatenate((y_true, y_pred)))
    n_classes = len(classes)

    result = np.zeros((n_classes, n_classes))

    for i in range(len(y_true)):
        true_class = np.where(classes == y_true[i])[0][0]
        pred_class = np.where(classes == y_pred[i])[0][0]
        result[true_class, pred_class] += 1

    return result


def accuracy_score(y_true: np.array, y_pred: np.array, percent=True) -> np.array:
    """
    Calculate the accuracy
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("accuracy_score: invalid shapes")
    if y_true.shape[0] == 0:
        return 0
    if percent:
        return 100 * np.mean(y_true == y_pred)
    else:
        return np.mean(y_true == y_pred)


def precision_score(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Calculate the precision
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("precision_score: invalid shapes")
    if y_true.shape[0] == 0:
        return 0
    cf_matrix = confusion_matrix(y_true, y_pred)
    result = np.zeros(shape=cf_matrix.shape[0])
    for c in range(cf_matrix.shape[0]):
        tp = cf_matrix[c, c]
        fn = np.sum(cf_matrix[c, :]) - tp
        result[c] = tp / (tp + fn)
    return result


def recall_score(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Calculate the recall
    """
    if y_true.shape[0] != y_pred.shape[0]:
        raise ValueError("precision_score: invalid shapes")
    if y_true.shape[0] == 0:
        return 0
    cf_matrix = confusion_matrix(y_true, y_pred)
    result = np.zeros(shape=cf_matrix.shape[0])
    for c in range(cf_matrix.shape[0]):
        tp = cf_matrix[c, c]
        fp = np.sum(cf_matrix[:, c]) - tp
        result[c] = tp / (tp + fp)
    return result


def f1_score(y_true: np.array, y_pred: np.array) -> np.array:
    """
    Calculate the f1 score
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    return 2 * (precision * recall) / (precision + recall)
