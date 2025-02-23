import numpy as np
from clusfl.server.base_server import BaseServer


class KMedianServer(BaseServer):
    def __init__(self):
        self.clients = []
        self.global_packets = np.array([])

    def add_client(self, client_id):
        if not client_id in self.clients:
            self.clients.append(client_id)
        raise ValueError(f"Client {client_id} already exists")

    def _check_clients(self, client_id):
        if not client_id in self.clients:
            return True
        raise ValueError(f"Client {client_id} not found")

    def receive_from_client(self, client_id, client_centers, client_weights):
        self._check_clients(client_id)
        client_packet = np.array([client_id, client_centers, client_weights])
        self.global_packets = np.append(self.global_packets, client_packet)

