import numpy as np


from clusfl.client import Client
from clusfl.server import Server
from clusfl.utils import Utils
from clusfl.metrics import Metrics
from clusfl.cluster import ClusteringAlgorithm


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
        elif federated_algorithm == "fed_kmeans":
            return FederatedAlgorithm.icpr_federated_clustering(
                client_data, num_clusters, model="kmeans"
            )
        elif federated_algorithm == "fed_kmedian":
            return FederatedAlgorithm.icpr_federated_clustering(
                client_data, num_clusters, model="kmedian"
            )
        else:
            raise ValueError(f"Unsupported federated algorithm: {federated_algorithm}")

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

    @staticmethod
    def icpr_federated_clustering(client_data, num_clusters, model="kmeans", rounds=10):
        """Federated K-Means with correct weight handling and client-server communication."""

        # Step 1: Initialize local cluster centers at clients
        centers = []
        weights = []

        for i in range(len(client_data)):
            modelpp_name = model + "++"
            modelpp = ClusteringAlgorithm.invoke_clustering_model(
                modelpp_name, num_clusters
            )
            modelpp.fit(client_data[i])
            centers.append(modelpp.cluster_centers_)
            weights.append(modelpp.weights_)

        # Convert lists to NumPy arrays
        centers = np.vstack(centers)
        weights = np.concatenate(weights)

        # 🔹 Instead of initializing with zeros, use the first computed global centers
        global_centers = np.copy(centers[:num_clusters])

        # Step 2: Iterate Federated Clustering Rounds
        for _ in range(rounds):
            cfl = ClusteringAlgorithm.invoke_clustering_model(model, num_clusters)

            # 🔹 Expand data using weights (avoid np.repeat issues)
            clustering_data = np.concatenate(
                [np.tile(centers[i], (weights[i], 1)) for i in range(len(weights))]
            )

            # Fit the global clustering model
            cfl.fit(clustering_data)
            global_centers = cfl.cluster_centers_

            # Step 3: Send Global Centers to Clients
            new_centers = []
            new_weights = []

            for i in range(len(client_data)):
                client_model = ClusteringAlgorithm.invoke_clustering_model(
                    model, num_clusters
                )

                # 🔹 Pass `global_centers` as `init_centers`
                client_model.cluster_centers_ = global_centers
                client_model.fit(client_data[i])

                # Collect updated cluster centers & weights
                new_centers.append(client_model.cluster_centers_)
                new_weights.append(client_model.weights_)

            # Convert lists to NumPy arrays again
            centers = np.vstack(new_centers)
            weights = np.concatenate(new_weights)

        # Step 5: Return Final Aggregated Global Centers
        return global_centers
