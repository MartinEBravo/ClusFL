import numpy as np
from sklearn.datasets import make_blobs
from scipy.stats import beta, gamma, expon


class DataGenerator:
    @staticmethod
    def generate_data_by_distribution(
        distribution, num_samples, num_features, num_clusters, fixed_centers
    ):
        """Generates data based on the specified distribution."""
        if distribution == "normal":
            data = make_blobs(
                n_samples=num_samples,
                n_features=num_features,
                centers=fixed_centers,
                cluster_std=2.5,
            )
            return data[0], data[1]

        elif distribution == "uniform":
            X = np.random.uniform(0, 12, size=(num_samples, num_features))
            y = np.random.randint(0, num_clusters, size=num_samples)
            return X, y

        elif distribution == "beta":
            X = beta.rvs(2, 5, size=(num_samples, num_features)) * 8
            y = np.random.randint(0, num_clusters, size=num_samples)
            return X, y

        elif distribution == "gamma":
            X = gamma.rvs(2, scale=3, size=(num_samples, num_features)) + 2
            y = np.random.randint(0, num_clusters, size=num_samples)
            return X, y

        elif distribution == "exponential":
            X = expon.rvs(scale=2.5, size=(num_samples, num_features)) + 1
            y = np.random.randint(0, num_clusters, size=num_samples)
            return X, y

        else:
            raise ValueError("Unsupported distribution type.")

    @staticmethod
    def generate_federated_data(
        num_clients,
        num_samples_per_client,
        num_clusters,
        distribution_setup,
        num_features,
        fixed_centers,
    ):
        """Generates federated data with clients having different distributions (Non-IID)."""
        client_data, client_labels = [], []

        # If only a string is passed, make each client have a different distribution
        if isinstance(distribution_setup, str):
            distribution_setup = [distribution_setup] * num_clients

        # If the list is smaller than the number of clients, expand it ensuring randomness
        elif isinstance(distribution_setup, list):
            unique_distributions = len(set(distribution_setup))
            if unique_distributions < num_clients:
                distribution_setup = (
                    distribution_setup * (num_clients // unique_distributions)
                    + distribution_setup[: num_clients % unique_distributions]
                )
                np.random.shuffle(distribution_setup)

        for i in range(num_clients):
            X, y = DataGenerator.generate_data_by_distribution(
                distribution_setup[i],
                num_samples_per_client,
                num_features,
                num_clusters,
                fixed_centers,
            )
            client_data.append(X)
            client_labels.append(y)

        return client_data, client_labels
