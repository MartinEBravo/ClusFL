from clusfl.utils import Utils


class Server:
    @staticmethod
    def aggregate_cluster_centers(cluster_centers, num_clusters, model="kmeans"):
        """Aggregates cluster centers from client data."""
        aggregator_model = Utils.invoke_clustering_model(model, num_clusters)
        aggregator_model.fit(cluster_centers)
        return aggregator_model.cluster_centers_
