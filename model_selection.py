import random
import numpy as np

def train_test_split(X, y, test_size=0.2):
    """
    Split the X and y datasets into two random X_train, X_test, y_train and y_test
    datasets for further purpose
    """
    if X.shape[0] != y.shape[0] or y.shape[1] != 1:
        raise Exception("Invalid shapes")
    L = random.sample(range(0, X.shape[0]), int(0.2 * X.shape[0]))
    X_test, y_test = [], []
    X_train, y_train = [], []
    for index in L:
        X_test.append(X[index])
        y_test.append(y[index])
    for index in range(X.shape[0]):
        if not index in L:
            X_train.append(X[index])
            y_train.append(y[index])
    return np.array(X_train), np.array(X_test), np.array(y_train), np.array(y_test)
    