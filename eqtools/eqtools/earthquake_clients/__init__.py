# earthquake_client_factory.py

from .fdsn_client import FDSNClient
from .usgs_client import USGSClient
from .gcmt_client import GCMTClient
from .iris_client import IRISClient
from .logging_config import logger

class EarthquakeClientFactory:
    @staticmethod
    def create_client(client_type, **kwargs):
        """
        Factory method to create earthquake client instances.

        Args:
            client_type (str): The type of the client to create ('fdsn', 'usgs', 'gcmt', 'iris').
            **kwargs: Additional keyword arguments to pass to the client constructor.

        Returns:
            An instance of the requested earthquake client.
        """
        if client_type == 'fdsn':
            return FDSNClient(**kwargs)
        elif client_type == 'usgs':
            return USGSClient(**kwargs)
        elif client_type == 'gcmt':
            return GCMTClient(**kwargs)
        elif client_type == 'iris':
            return IRISClient(**kwargs)
        else:
            logger.error(f"Unknown client type: {client_type}")
            raise ValueError(f"Unknown client type: {client_type}")


if __name__ == "__main__":
    
    # Create a USGS client
    usgs_client = EarthquakeClientFactory.create_client('usgs', include_focal_mechanism=True)
    
    # Use the client to get events
    usgs_client.get_events("2023-01-01", "2023-02-28", 5.0, 9.0, "usgs_earthquake_catalog.csv")
    
    # Create a GCMT client
    gcmt_client = EarthquakeClientFactory.create_client('gcmt')
    
    # Use the client to get events
    gcmt_client.get_events("2023-01-01", "2023-02-28", 5.0, 9.0, "gcmt_earthquake_catalog.csv")