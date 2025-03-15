from clusfl.client import Client
from clusfl.server import Server
from clusfl.utils import Utils
from clusfl.metrics import Metrics


class FederatedAlgorithm:
    @staticmethod
    def federated_clustering(
        client_data,
        num_clusters,
        fixed_centers,
        federated_algorithm="fedavg",
    ):
        """Performs federated clustering."""
        aggregated_centers = FederatedAlgorithm.invoke_federated_algorithm(
            client_data, num_clusters, federated_algorithm
        )

        actual_centers = Utils.match_centers(aggregated_centers, fixed_centers)

        all_points, all_labels, all_predicted_labels = Utils.get_all_points(
            client_data, actual_centers, aggregated_centers
        )
        nmi_score, silhouette_score_value = Metrics.get_relevant_metrics(
            all_labels, all_predicted_labels, all_points
        )
        return {
            "all_labels": all_labels,
            "all_predicted_labels": all_predicted_labels,
            "aggregated_centers": aggregated_centers,
            "NMI": nmi_score,
            "silhouette_score": silhouette_score_value,
        }

    @staticmethod
    def invoke_federated_algorithm(
        client_data,
        num_clusters,
        federated_algorithm="simple_kmeans",
    ):
        """Invokes the federated algorithm."""
        if federated_algorithm == "simple_kmeans":
            return FederatedAlgorithm.simple_federated_clustering(
                client_data, num_clusters, model="kmeans"
            )
        elif federated_algorithm == "simple_kmedian":
            return FederatedAlgorithm.simple_federated_clustering(
                client_data, num_clusters, model="kmedian"
            )
        else:
            raise ValueError("Invalid federated algorithm.")

    @staticmethod
    def simple_federated_clustering(client_data, num_clusters, model="kmeans"):
        """Performs simple federated clustering."""
        cluster_centers = Client.aggregate_cluster_centers(
            client_data, num_clusters, model=model
        )
        aggregated_centers = Server.aggregate_cluster_centers(
            cluster_centers, num_clusters, model=model
        )

        return aggregated_centers
