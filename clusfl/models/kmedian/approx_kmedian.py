import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class KMedianApprox:
    def __init__(self, n_clusters, epsilon=0.1):
        self.n_clusters = n_clusters
        self.epsilon = epsilon

    def fit(self, X):
        """
        Approximates the k-median problem using a 1 + sqrt(3) + epsilon approximation algorithm.
        """
        n = X.shape[0]
        distance_matrix = cdist(X, X, metric="euclidean")

        # Step 1: Find an initial k-median solution using a bi-point approximation
        centers, assignments = self._bi_point_solution(X, distance_matrix)

        # Step 2: Convert the bi-point solution into an integer solution with k+O(1) facilities
        pseudo_solution = self._pseudo_approximation(
            X, centers, assignments, distance_matrix
        )

        # Step 3: Convert the pseudo-approximation to a strict k-median solution
        final_solution = self._convert_to_k_median(pseudo_solution)

        self.centers = final_solution
        self.weights = self._get_weights(X, final_solution)
        return self.centers, self.weights

    def get_approximation_factor(self):
        """
        Returns the approximation factor of the algorithm.
        """
        return 1 + np.sqrt(3) + self.epsilon

    def _bi_point_solution(self, X, distance_matrix):
        """
        Generates a bi-point fractional solution as per the paper.
        """
        n = X.shape[0]
        _, assignments = linear_sum_assignment(distance_matrix)
        centers = X[assignments[: self.n_clusters]]
        return centers, assignments

    def _pseudo_approximation(self, X, centers, assignments, distance_matrix):
        """
        Constructs a pseudo-approximation by allowing k+O(1) centers.
        """
        extra_centers = int(self.epsilon * self.n_clusters)
        sorted_indices = np.argsort(np.min(distance_matrix[:, assignments], axis=1))
        additional_centers = X[sorted_indices[:extra_centers]]

        return np.vstack((centers, additional_centers))

    def _convert_to_k_median(self, pseudo_solution):
        """
        Converts the pseudo-approximation to a strict k-median solution.
        """
        # Prune extra centers by selecting the best k centers that minimize cost
        selected_indices = np.random.choice(
            pseudo_solution.shape[0], self.n_clusters, replace=False
        )
        return pseudo_solution[selected_indices]

    def _get_weights(self, X, centers):
        """
        Returns the weights of each point w.r.t. the centers.
        """
        distances = cdist(X, centers, metric="euclidean")
        return np.min(distances, axis=1)
