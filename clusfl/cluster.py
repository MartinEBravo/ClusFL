import numpy as np


class ClusteringAlgorithm:
    class KMeans:
        def __init__(self, n_clusters):
            self.n_clusters = n_clusters
            self.cluster_centers_ = None
            self.labels_ = None
            self.weights_ = None

        def fit(self, X):
            n_samples, _ = X.shape
            if self.cluster_centers_ is None:
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

            self.weights_ = np.bincount(self.labels_, minlength=self.n_clusters)

            return self

    class KMeanspp:
        def __init__(self, n_clusters):
            self.n_clusters = n_clusters
            self.cluster_centers_ = None
            self.labels_ = None
            self.weights_ = None

        def fit(self, X):
            n_samples, _ = X.shape
            self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
            self.labels_ = np.zeros(n_samples)

            # Inicializar el primer centro aleatoriamente
            self.cluster_centers_[0] = X[np.random.choice(n_samples)]
            for i in range(1, self.n_clusters):
                distances = np.linalg.norm(
                    X[:, None] - self.cluster_centers_[:i], axis=2
                )
                min_distances = np.min(distances, axis=1)
                probabilities = min_distances / min_distances.sum()
                self.cluster_centers_[i] = X[
                    np.random.choice(n_samples, p=probabilities)
                ]

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

            self.weights_ = np.bincount(self.labels_, minlength=self.n_clusters)

            return self

    class KMedian:
        def __init__(self, n_clusters):
            self.n_clusters = n_clusters
            self.cluster_centers_ = None
            self.labels_ = None
            self.weights_ = None

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

            self.weights_ = np.bincount(self.labels_, minlength=self.n_clusters)

            return self

    class KMedianpp:
        def __init__(self, n_clusters):
            self.n_clusters = n_clusters
            self.cluster_centers_ = None
            self.labels_ = None
            self.weights_ = None

        def fit(self, X):
            n_samples, _ = X.shape
            self.cluster_centers_ = np.zeros((self.n_clusters, X.shape[1]))
            self.labels_ = np.zeros(n_samples)

            self.cluster_centers_[0] = X[np.random.choice(n_samples)]
            for i in range(1, self.n_clusters):
                distances = np.linalg.norm(
                    X[:, None] - self.cluster_centers_[:i], axis=2
                )
                min_distances = np.min(distances, axis=1)
                probabilities = min_distances / min_distances.sum()
                self.cluster_centers_[i] = X[
                    np.random.choice(n_samples, p=probabilities)
                ]

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

            self.weights_ = np.bincount(self.labels_, minlength=self.n_clusters)

            return self

    @staticmethod
    def invoke_clustering_model(model, num_clusters):
        if model == "kmeans":
            return ClusteringAlgorithm.KMeans(n_clusters=num_clusters)
        elif model == "kmedian":
            return ClusteringAlgorithm.KMedian(n_clusters=num_clusters)
        elif model == "kmeans++":
            return ClusteringAlgorithm.KMeanspp(n_clusters=num_clusters)
        elif model == "kmedian++":
            return ClusteringAlgorithm.KMedianpp(n_clusters=num_clusters)
        else:
            raise ValueError(f"Unsupported clustering model: {model}")
