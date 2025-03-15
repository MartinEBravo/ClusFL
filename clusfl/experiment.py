import numpy as np
import tqdm
from clusfl.federated import FederatedClustering


class Experiment:
    @staticmethod
    def compare_two_clustering_algorithms(
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
            for m in model:
                fed_results = FederatedClustering.federated_clustering(
                    distribution_setup,
                    num_clients=num_clients,
                    num_samples_per_client=num_samples_per_client,
                    num_clusters=num_clusters,
                    fixed_centers=fixed_centers,
                    num_features=num_features,
                    model=m,
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

        results_summary = Experiment.show_results(results)
        return results_summary

    @staticmethod
    def show_results(results):
        """Displays the results of the clustering experiment."""
        print("\n------------------ RESULTS ------------------")
        results_str = ""
        for model, metrics in results.items():
            results_str += f"\n\nModel: {model}\n"
            for metric, values in metrics.items():
                avg_value = np.mean(values)
                results_str += f"{metric}: {avg_value:.4f} (±{np.std(values):.4f})\n"

        print(results_str)
        return results_str
