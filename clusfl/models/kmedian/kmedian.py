import numpy as np

from clusfl.models.base_model import BaseModel
from clusfl.models.kmedian.approx_kmedian import KMedianApprox


class KMedian(BaseModel):
    def __init__(self, n_clusters=8, max_iter=300, method="approx_kmedian"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.model = self._get_model(method)

    def _get_model(self, method):
        methods = {
            "approx_kmedian": KMedianApprox,
        }
        if method in methods:
            return methods[method](self.n_clusters)
        else:
            raise ValueError(f"Invalid method: {method}")

    def fit(self, X):
        self.centers, self.weights = self.model.fit(X)
        return self.centers, self.weights
