from abc import ABC, abstractmethod


class BaseGenerator(ABC):
    """Abstract class for distributing data among federated clients."""

    @abstractmethod
    def generate_data(self):
        """
        Generate data for clients.
        Returns:
            X (np.array): Data points
            y (np.array): Labels
        """
        pass
