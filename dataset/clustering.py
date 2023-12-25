import numpy as np


def generate_clustering_dataset(
    n_samples=100, n_features=2, n_clusters=3, cluster_std=0.1, random_state=None
):
    np.random.seed(random_state)
    centroids = np.random.rand(n_clusters, n_features)

    X = []
    for _ in range(n_samples):
        class_label = np.random.randint(0, n_clusters)
        sample = centroids[class_label] + cluster_std * np.random.randn(n_features)
        X.append(sample)

    return np.array(X)
