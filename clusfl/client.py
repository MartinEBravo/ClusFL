import numpy as np
from clusfl.utils import Utils


class Client:
    @staticmethod
    def aggregate_cluster_centers(client_data, num_clusters, model="kmeans"):
        """Aggregates cluster centers from client data."""
        cluster_centers = []
        for client in client_data:
            try:
                cfl = Utils.invoke_clustering_model(model, num_clusters)
            except ValueError:
                raise ValueError("Unsupported clustering model.")
            cfl.fit(client)
            cluster_centers.append(cfl.cluster_centers_)
        cluster_centers = np.vstack(cluster_centers)
        return cluster_centers
