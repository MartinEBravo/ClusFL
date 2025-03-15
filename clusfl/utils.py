import numpy as np
from clusfl.cluster import KMeans, KMedian, KMedoids


# ------------------ UTILS ------------------
class Utils:
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

    @staticmethod
    def get_all_points(client_data, actual_centers, aggregated_centers):
        all_points = np.vstack(client_data)
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
        return all_points, all_labels, all_predicted_labels
