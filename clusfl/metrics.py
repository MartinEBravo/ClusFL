from sklearn.metrics import (
    normalized_mutual_info_score,
    silhouette_score,
)
import numpy as np


class Metrics:
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

    @staticmethod
    def get_relevant_metrics(all_labels, all_predicted_labels, all_points):
        """Calculates relevant metrics for clustering."""
        nmi_score = normalized_mutual_info_score(all_labels, all_predicted_labels)
        silhouette_score_value = (
            silhouette_score(all_points, all_predicted_labels)
            if len(np.unique(all_predicted_labels)) > 1
            else 0
        )
        return nmi_score, silhouette_score_value
