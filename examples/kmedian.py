import numpy as np
from clusfl.models import KMedian
from clusfl.data.non_iid_generator import NonIIDGenerator
from sklearn.cluster import KMeans


def test_kmedian_vs_kmeans():
    data_generator = NonIIDGenerator(n_samples=1000, n_clusters=3, cluster_std=1.0)
    X, y = data_generator.generate_data()

    # KMedian clustering
    kmedian = KMedian(n_clusters=3)
    centers, weights = kmedian.fit(X)
    print("KMedian centers:")
    print(centers)
    print("KMedian weights:")
    print(weights)

    # KMeans clustering
    kmeans = KMeans(n_clusters=3)
    kmeans.fit(X)
    print("KMeans centers:")
    print(kmeans.cluster_centers_)

    # Compare loss
    kmedian_loss = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
    kmedian_loss = np.sum(np.min(kmedian_loss, axis=1))
    print(f"KMedian loss: {kmedian_loss}")

    kmeans_loss = np.linalg.norm(X[:, np.newaxis] - kmeans.cluster_centers_, axis=2)
    kmeans_loss = np.sum(np.min(kmeans_loss, axis=1))
    print(f"KMeans loss: {kmeans_loss}")

    assert kmedian_loss < kmeans_loss, "KMedian loss should be lower than KMeans loss"


if __name__ == "__main__":
    test_kmedian_vs_kmeans()
