from classes.model import Model
import numpy as np


class KMeans(Model):
    def __init__(self, n_clusters: int, max_iter=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.__fitted = False
        np.random.seed(random_state)

    def __assign_clusters(self):
        for i in range(self.datapoints):
            for c in range(self.n_clusters):
                self.distances[i, c] = np.linalg.norm(self.X[i] - self.centroids[c])
            self.clusters[i] = np.argmin(self.distances[i, :])

    def __compute_new_centroids(self):
        for c in range(self.n_clusters):
            indices = np.where(self.clusters == c)
            X_c = self.X[indices]
            self.centroids[c] = (
                np.mean(X_c, axis=0) if len(indices) > 0 else self.centroids[c]
            )

    def fit(self, X: np.array, y: np.array = None):
        self.__fitted = True
        self.X = X
        self.centroids = X[
            np.random.choice(np.arange(X.shape[0]), size=self.n_clusters, replace=False)
        ]
        self.last_centroids = None
        self.datapoints = X.shape[0]
        self.distances = np.zeros((X.shape[0], self.n_clusters))
        self.clusters = np.zeros(X.shape[0])
        iter = 0
        while (
            self.last_centroids is None
            or not np.all(np.equal(self.last_centroids, self.centroids))
        ) and (self.max_iter is None or iter < self.max_iter):
            self.last_centroids = np.copy(self.centroids)
            self.__assign_clusters()
            self.__compute_new_centroids()
            iter += 1

    def __find_cluster(self, X: np.array):
        best_cluster, min_dist = None, float("inf")
        for c in range(self.n_clusters):
            dist = np.linalg.norm(X - self.centroids[c])
            if best_cluster is None or min_dist > dist:
                best_cluster, min_dist = c, dist
        return best_cluster

    def predict(self, X: np.array) -> np.array:
        y_pred = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            y_pred[i] = self.__find_cluster(X[i])
        return y_pred

    def fit_predict(self, X: np.array, y: np.array = None) -> np.array:
        self.fit(X, y)
        return self.predict(X)
