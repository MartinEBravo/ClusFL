import numpy as np
import tqdm
from clusfl.federated import FederatedAlgorithm
from clusfl.dataset import DataGenerator
from clusfl.metrics import Metrics


class Experiment:
    @staticmethod
    def compare_clustering_algorithms(
        distribution_setup,
        num_clients=10,
        num_samples_per_client=100,
        num_clusters=4,
        fixed_centers=np.array([[0, 0], [10, 0], [0, 10], [10, 10]]),
        num_features=2,
        model=["kmeans", "kmedian"],
        iterations=50,
    ):
        """Compares clustering algorithms with additional stability metrics."""
        print("\n------------------ EXPERIMENT: NON-IID ------------------")
        results = {}
        for m in model:
            results[m] = {}
            results[m]["NMI"] = []
            results[m]["silhouette_score"] = []
            results[m]["avg_distance"] = []
            results[m]["cluster_variance"] = []

        for _ in tqdm.tqdm(range(iterations)):
            client_data, _ = DataGenerator.generate_federated_data(
                num_clients,
                num_samples_per_client,
                num_clusters,
                distribution_setup,
                num_features,
                fixed_centers,
            )

            for m in model:
                fed_results = FederatedAlgorithm.federated_clustering(
                    client_data=client_data,
                    fixed_centers=fixed_centers,
                    num_clusters=num_clusters,
                    federated_algorithm=m,
                )

                # Calculation of additional metrics
                distances = np.linalg.norm(
                    fed_results["aggregated_centers"] - fixed_centers, axis=1
                )
                avg_distance = np.mean(distances)
                cluster_variance = np.var(distances)
                results[m]["NMI"].append(fed_results["NMI"])
                results[m]["silhouette_score"].append(fed_results["silhouette_score"])
                results[m]["avg_distance"].append(avg_distance)
                results[m]["cluster_variance"].append(cluster_variance)

        results_summary = Metrics.show_results(results)
        return results_summary
