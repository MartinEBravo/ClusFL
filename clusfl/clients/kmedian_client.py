import numpy as np
from clusfl.models.kmedian.kmedian import KMedian


class KMedianClient:
    def __init__(self, client_id, data, n_clusters=8, max_iter=300, method="approx"):
        self.client_id = client_id
        self.data = data
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.weights = None
        self.model = KMedian(
            n_clusters=self.n_clusters, max_iter=self.max_iter, method=method
        )

    def fit(self):
        self.centers, self.weights = self.model.fit(self.data)
        return self.centers, self.weights

    def send_to_server(self, server):
        if self.centers and self.weights:
            server.receive_from_client(self.centers, self.weights)
        else:
            print("No clusters to send to server")
