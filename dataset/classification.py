import numpy as np


def generate_classification_dataset(
    n_samples=100, n_features=2, n_classes=2, random_state=None
):
    np.random.seed(random_state)
    centroids = np.random.rand(n_classes, n_features)
    X, y = [], []
    for _ in range(n_samples):
        class_label = np.random.randint(0, n_classes)
        sample = centroids[class_label] + 0.1 * np.random.randn(n_features)
        X.append(sample)
        y.append(class_label)
    return np.array(X), np.array(y)
