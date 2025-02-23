from sklearn.cluster import KMeans


from clusfl.models.base_model import BaseModel


class KMeans(BaseModel):
    def __init__(self, n_clusters=8, max_iter=300):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.model = KMeans(n_clusters=self.n_clusters, max_iter=self.max_iter)

    def fit(self, X):
        self.model.fit(X)
        self.centers = self.model.cluster_centers_
        return self.centers