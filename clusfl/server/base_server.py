from abc import ABC, abstractmethod


class BaseServer(ABC):
    """Abstract class for federated learning server."""

    @abstractmethod
    def receive_from_client(self):
        """Receive data from clients."""
        pass
