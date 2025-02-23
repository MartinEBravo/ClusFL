import numpy as np
from clusfl.data.base_generator import BaseGenerator


class NonIIDGenerator(BaseGenerator):
    def __init__(
        self,
        n_samples=1000,
        n_dimensions=2,
        n_clusters=3,
        cluster_std=1.0,
        random_state=None,
    ):
        """
        Generates Non-IID data by clustering points into different distributions.

        Parameters:
        - n_samples: Total number of data points
        - n_dimensions: Number of dimensions per sample
        - n_clusters: Number of underlying distributions (to simulate heterogeneous clients)
        - cluster_std: Standard deviation of clusters
        - random_state: Seed for reproducibility
        """
        self.n_samples = n_samples
        self.n_dimensions = n_dimensions
        self.n_clusters = n_clusters
        self.cluster_std = cluster_std
        self.random_state = random_state

        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)

        # Generate clustered data
        self.X, self.y = self.generate_clusters()

    def generate_data(self):
        """
        Generate Non-IID data by sampling from multiple Gaussian clusters.
        Returns:
            X (np.array): Data points
            y (np.array): Cluster labels
        """
        X = []
        y = []

        # Define cluster centers randomly
        cluster_centers = np.random.uniform(-5, 5, (self.n_clusters, self.n_dimensions))

        samples_per_cluster = self.n_samples // self.n_clusters
        for i, center in enumerate(cluster_centers):
            cluster_samples = (
                np.random.randn(samples_per_cluster, self.n_dimensions)
                * self.cluster_std
                + center
            )
            X.append(cluster_samples)
            y.extend([i] * samples_per_cluster)  # Assign cluster labels

        X = np.vstack(X)
        y = np.array(y)

        return X, y
