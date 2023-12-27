import numpy as np


def generate_clustering_dataset(
    n_samples=100, n_features=2, n_clusters=3, cluster_std=0.1, random_state=None
):
    """
    Generate a synthetic dataset for clustering.

    This function creates a dataset by generating random centroids for each cluster
    and then adding normally distributed data around these centroids.

    Parameters
    ----------
    n_samples : int, optional
        The number of samples to generate (default is 100).
    n_features : int, optional
        The number of features for each sample (default is 2).
    n_clusters : int, optional
        The number of distinct clusters (default is 3).
    cluster_std : float, optional
        The standard deviation of the clusters (default is 0.1).
    random_state : int, optional
        The seed used by the random number generator (default is None).

    Returns
    -------
    np.ndarray
        The generated samples, an array of shape (n_samples, n_features).
    """
    np.random.seed(random_state)
    centroids = np.random.rand(n_clusters, n_features)

    X = []
    for _ in range(n_samples):
        class_label = np.random.randint(0, n_clusters)
        sample = centroids[class_label] + cluster_std * np.random.randn(n_features)
        X.append(sample)

    return np.array(X)
