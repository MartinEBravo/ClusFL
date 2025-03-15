from clusfl.dataset import DataGenerator
from clusfl.client import Client
from clusfl.server import Server
from clusfl.utils import ClusteringUtils
import numpy as np
from .metrics import normalized_mutual_info_score, silhouette_score


class FederatedClustering:
    @staticmethod
    def federated_clustering(
        distribution_setup,
        num_clients=10,
        num_samples_per_client=100,
        num_clusters=4,
        fixed_centers=np.array([[0, 0], [10, 0], [0, 10], [10, 10]]),
        num_features=2,
        model="kmeans",
    ):
        """Performs federated clustering."""
        client_data, _ = DataGenerator.generate_federated_data(
            num_clients,
            num_samples_per_client,
            num_clusters,
            distribution_setup,
            num_features,
            fixed_centers,
        )
        cluster_centers = Client.aggregate_cluster_centers(client_data, num_clusters, model=model)
        aggregated_centers = Server.aggregate_cluster_centers(cluster_centers, num_clusters, model=model)
        actual_centers = ClusteringUtils.match_centers(
            aggregated_centers, fixed_centers
        )

        all_points = np.concatenate(client_data)
        all_labels = np.array(
            [
                np.argmin(np.linalg.norm(point - actual_centers, axis=1))
                for point in all_points
            ]
        )
        all_predicted_labels = np.array(
            [
                np.argmin(np.linalg.norm(point - aggregated_centers, axis=1))
                for point in all_points
            ]
        )
        nmi_score = normalized_mutual_info_score(all_labels, all_predicted_labels)
        silhouette_score_value = (
            silhouette_score(all_points, all_predicted_labels)
            if len(np.unique(all_predicted_labels)) > 1
            else 0
        )
        return {
            "all_labels": all_labels,
            "all_predicted_labels": all_predicted_labels,
            "aggregated_centers": aggregated_centers,
            "NMI": nmi_score,
            "silhouette_score": silhouette_score_value,
        }
