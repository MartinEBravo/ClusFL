import numpy as np
from clusfl.experiment import Experiment


# ------------------ PARAMETERS ------------------
num_clients_list = [500, 1000]
num_samples_per_client_list = [500, 1000]
num_clusters = 4
num_features = 2
distribution_types = ["beta", "gamma", "exponential"]
fixed_centers = np.array([[2, 2], [20, 20], [-10, -10], [11, 11]])


# ------------------ EXPERIMENTS ------------------
results = Experiment.compare_clustering_algorithms(
    distribution_setup=distribution_types,
    num_clients=100,
    num_samples_per_client=100,
    num_clusters=num_clusters,
    fixed_centers=fixed_centers,
    num_features=num_features,
    model=["kmeans", "kmedian"],
    iterations=1,
)
