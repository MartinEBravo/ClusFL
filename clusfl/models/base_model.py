from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def __init__(self, n_clusters=8, max_iter=300, method="method"):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centers = None
        self.labels = None

    @abstractmethod
    def fit(self, X):
        """
        Fit the clustering model to the data.

        Parameters
        ----------
        X : numpy.ndarray
            The data to cluster.

        Returns
        -------
        None
        """
        pass
