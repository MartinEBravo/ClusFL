from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract class for distributing data among federated clients."""

    @abstractmethod
    def generate_clusters(self):
        """Generate clusters for each client."""
        pass
