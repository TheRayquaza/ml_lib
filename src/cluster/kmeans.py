from classes.model import Model
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count


class KMeans(Model):
    def __init__(self, n_clusters: int, max_iter=100, random_state=None, n_jobs=None, name="KMeans"):
        super().__init__(random_state=random_state, name=name)
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_jobs = cpu_count() if n_jobs == -1 else n_jobs
        self._fitted = False

    def _assign_cluster(self, i: int) -> tuple:
        """
        Assign a data point to the nearest cluster.

        Parameters
        ----------
            i (int): Index of the data point.

        Returns
        -------
            tuple: Tuple containing the assigned cluster index and the data point index.
        """
        for c in range(self.n_clusters):
            self.distances[i, c] = np.linalg.norm(self.X[i] - self.centroids[c])
        return np.argmin(self.distances[i, :]), i

    def _assign_clusters(self):
        """
        Assign all data points to the nearest clusters.
        """
        if self.n_jobs is None:
            for i in range(self.datapoints):
                self.clusters[i] = self._assign_cluster(i)[1]
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self._assign_cluster, i): i for i in range(self.datapoints)
            }
            for future in as_completed(future_to_pred):
                v, c = future.result()
                self.clusters[c] = v

    def _compute_centroid(self, c: int) -> tuple:
        """
        Compute the centroid for a given cluster.

        Parameters
        ----------
            - c: Cluster index.

        Returns
        -------
            tuple: Tuple containing the computed centroid and the cluster index.
        """
        indices = np.where(self.clusters == c)
        X_c = self.X[indices]
        return np.mean(X_c, axis=0) if len(indices) > 0 else self.centroids[c], c

    def _compute_new_centroids(self):
        """
        Compute new centroids for all clusters.
        """
        if self.n_jobs is None:
            for c in range(self.n_clusters):
                self.centroids[c] = self._compute_centroid(c)[0]
        else:
            pool = ThreadPoolExecutor(max_workers=self.n_jobs)
            future_to_pred = {
                pool.submit(self._compute_centroid, c): c
                for c in range(self.n_clusters)
            }
            for future in as_completed(future_to_pred):
                v, c = future.result()
                self.centroids[c] = v

    def fit(self, X: np.ndarray, y: np.ndarray = None):
        """
        Fit the KMeans model to the input data.

        Parameters
        ----------
            - X (np.ndarray): Input data.
            - y (np.ndarray, optional): Target labels. Unused in KMeans.

        Returns
        -------
            KMeans: The fitted KMeans model.
        """
        super().fit(X, y)
        self.centroids = X[
            np.random.choice(np.arange(self.samples), size=self.n_clusters, replace=False)
        ]
        self.last_centroids = None
        self.distances = np.zeros((self.samples, self.n_clusters))
        self.clusters = np.zeros(self.samples)
        iter = 0
        while (
            self.last_centroids is None
            or not np.all(np.equal(self.last_centroids, self.centroids))
        ) and (self.max_iter is None or iter < self.max_iter):
            self.last_centroids = np.copy(self.centroids)
            self._assign_clusters()
            self._compute_new_centroids()
            iter += 1
        return self

    def _find_cluster(self, X: np.ndarray):
        """
        Find the cluster index for a given data point.

        Parameters
        ----------
            X (np.ndarray): Data point.

        Returns
        -------
            int: The index of the assigned cluster.
        """
        best_cluster, min_dist = None, float("inf")
        for c in range(self.n_clusters):
            dist = np.linalg.norm(X - self.centroids[c])
            if best_cluster is None or min_dist > dist:
                best_cluster, min_dist = c, dist
        return best_cluster

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict the cluster index for each data point in the input data.

        Parameters
        ----------
            - X (np.ndarray): Input data.

        Returns
        -------
            np.ndarray: Array of predicted cluster indices.
        """
        if not self._fitted:
            raise Exception("KMeans: not fitted with data")
        samples = X.shape[0]
        y_pred = np.zeros(samples)
        for i in range(samples):
            y_pred[i] = self._find_cluster(X[i])
        return y_pred
