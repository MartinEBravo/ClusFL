import numpy as np


class KMeans:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        n_samples, _ = X.shape
        self.cluster_centers_ = X[
            np.random.choice(n_samples, self.n_clusters, replace=False)
        ]
        self.labels_ = np.zeros(n_samples)

        while True:
            distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_cluster_centers = np.array(
                [
                    np.mean(X[self.labels_ == i], axis=0)
                    if np.any(self.labels_ == i)
                    else self.cluster_centers_[i]
                    for i in range(self.n_clusters)
                ]
            )
            if np.allclose(self.cluster_centers_, new_cluster_centers, atol=1e-4):
                break

            self.cluster_centers_ = new_cluster_centers

        return self


class KMedian:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        n_samples, _ = X.shape
        self.cluster_centers_ = X[
            np.random.choice(n_samples, self.n_clusters, replace=False)
        ]
        self.labels_ = np.zeros(n_samples)

        while True:
            distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_cluster_centers = np.array(
                [
                    np.median(X[self.labels_ == i], axis=0)
                    if np.any(self.labels_ == i)
                    else self.cluster_centers_[i]
                    for i in range(self.n_clusters)
                ]
            )
            if np.allclose(self.cluster_centers_, new_cluster_centers, atol=1e-4):
                break

            self.cluster_centers_ = new_cluster_centers

        return self


class KMedoids:
    def __init__(self, n_clusters):
        self.n_clusters = n_clusters
        self.cluster_centers_ = None
        self.labels_ = None

    def fit(self, X):
        n_samples, _ = X.shape
        self.cluster_centers_ = X[
            np.random.choice(n_samples, self.n_clusters, replace=False)
        ]
        self.labels_ = np.zeros(n_samples)

        while True:
            distances = np.linalg.norm(X[:, None] - self.cluster_centers_, axis=2)
            self.labels_ = np.argmin(distances, axis=1)
            new_cluster_centers = np.array(
                [
                    X[self.labels_ == i][
                        np.argmin(
                            np.linalg.norm(
                                X[self.labels_ == i] - X[self.labels_ == i][:, None],
                                axis=2,
                            ).sum(axis=1)
                        )
                    ]
                    if np.any(self.labels_ == i)
                    else self.cluster_centers_[i]
                    for i in range(self.n_clusters)
                ]
            )
            if np.allclose(self.cluster_centers_, new_cluster_centers, atol=1e-4):
                break

            self.cluster_centers_ = new_cluster_centers

        return self
