import numpy as np
import pytest
from cluster.kmeans import KMeans


def test_kmeans_init():
    n_clusters = 3
    kmeans = KMeans(n_clusters)
    assert kmeans.n_clusters == n_clusters
    assert kmeans.max_iter == 100
    assert kmeans.n_jobs is None
    assert not kmeans._fitted


def test_kmeans_str():
    kmeans = KMeans(3)
    assert str(kmeans) == "KMeans"


def test_fit_predict():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    n_clusters = 2
    kmeans = KMeans(n_clusters)
    kmeans.fit(X)

    assert kmeans._fitted
    assert len(kmeans.centroids) == n_clusters

    predictions = kmeans.predict(X)
    assert len(predictions) == len(X)


def test_predict_without_fit():
    kmeans = KMeans(3)
    X = np.array([[1, 2], [1, 4], [1, 0]])

    with pytest.raises(Exception) as excinfo:
        kmeans.predict(X)
    assert "KMeans: not fitted with data" in str(excinfo.value)


def test_concurrent_fit():
    X = np.array([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
    kmeans = KMeans(2, n_jobs=-1)
    kmeans.fit(X)

    assert kmeans._fitted
    assert kmeans.n_jobs > 1
