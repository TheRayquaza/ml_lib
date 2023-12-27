import numpy as np
from dataset.clustering import generate_clustering_dataset
from graphic.curve import plot_2d_clusters, plot_3d_clusters
from cluster.kmeans import KMeans

clusters = 2

X = generate_clustering_dataset(n_samples=10000, n_features=3, n_clusters=clusters)
y_pred = KMeans(n_clusters=clusters, random_state=42, n_jobs=-1).fit_predict(X, None)
plot_3d_clusters(X, y_pred)
