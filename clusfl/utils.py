import numpy as np
from clusfl.cluster import KMeans, KMedian, KMedoids


# ------------------ CLUSTERING UTILS ------------------
class ClusteringUtils:
    @staticmethod
    def aggregate_cluster_centers(client_data, num_clusters, model="kmeans"):
        """Aggregates cluster centers from client data."""
        cluster_centers = []
        for client in client_data:
            try:
                cfl = ClusteringUtils.invoke_clustering_model(model, num_clusters)
            except ValueError:
                raise ValueError("Unsupported clustering model.")
            cfl.fit(client)
            cluster_centers.append(cfl.cluster_centers_)
        cluster_centers = np.vstack(cluster_centers)
        aggregator_model = ClusteringUtils.invoke_clustering_model(model, num_clusters)
        aggregator_model.fit(cluster_centers)
        return aggregator_model.cluster_centers_

    @staticmethod
    def invoke_clustering_model(model, num_clusters):
        if model == "kmeans":
            return KMeans(n_clusters=num_clusters)
        elif model == "kmedian":
            return KMedian(n_clusters=num_clusters)
        elif model == "kmedoids":
            return KMedoids(n_clusters=num_clusters)
        else:
            raise ValueError("Unsupported clustering model.")

    @staticmethod
    def match_centers(aggregated_centers, fixed_centers):
        """Matches aggregated centers to actual fixed centers."""
        matched_centers = []
        if len(fixed_centers) < len(aggregated_centers):
            raise ValueError("Fixed centers do not match aggregated cluster count.")

        for center in aggregated_centers:
            distances = [
                np.linalg.norm(center - actual_center)
                for actual_center in fixed_centers
            ]
            matched_centers.append(fixed_centers[np.argmin(distances)])

        return np.array(matched_centers)
