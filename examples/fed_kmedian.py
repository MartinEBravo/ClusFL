import numpy as np
from clusfl.data.non_iid_generator import NonIIDGenerator
from clusfl.clients.kmedian_client import KMedianClient
from clusfl.server.kmedian_server import KMedianServer


def federated_k_median_non_iid(
    n_clients=10,
    n_clusters=3,
    min_samples=50,
    max_samples=500,
    n_dimensions=2,
    cluster_std=1.0,
    max_iter=300,
    random_state=None,
):
    samples_per_client = np.random.randint(min_samples, max_samples, size=n_clients)
    print("Samples per client:", samples_per_client)

    # Crear servidor
    server = KMedianServer()

    clients = []
    for i in range(n_clients):
        
        # Generar datos no IID
        num_samples = samples_per_client[i]
        data_generator = NonIIDGenerator(
            n_samples=num_samples,
            n_dimensions=n_dimensions,
            n_clusters=n_clusters,
            cluster_std=cluster_std,
            random_state=random_state,
        )
        client_data, _ = data_generator.generate_data()

        # Crear cliente
        client_id = np.random.randint(0, 1000)
        client = KMedianClient(
            client_id=client_id,
            data=client_data,
            n_clusters=n_clusters,
            max_iter=max_iter,
            random_state=random_state,
        )
        clients.append(client)
        client_id = np.random.randint(0, 1000)
        server.add_client(client_id)

    print("Server initialized")

    # Clientes ejecutan k-Median Approx y envían clusters al servidor
    for client in clients:
        id = client.get_id()
        clusters, weights = client.fit(max_iter=max_iter)
        server.receive_from_client(id, clusters, weights)
        print(f"Client {id} sent clusters to server")

    # # El servidor computa los clusters finales
    # global_clusters = server.compute_global_clusters()
    # print("Final Global Clusters:\n", global_clusters)


if __name__ == "__main__":
    federated_k_median_non_iid(
        n_clients=10,
        n_clusters=3,
        min_samples=50,
        max_samples=500,
        n_dimensions=2,
        cluster_std=1.0,
        max_iter=300,
        random_state=42,
    )
