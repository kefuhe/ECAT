import requests
import pandas as pd
from abc import ABC, abstractmethod
from .logging_config import logger


class FDSNClient(ABC):
    def __init__(self, base_url):
        self.base_url = base_url

    def get_params(self, start_time, end_time, min_magnitude, max_magnitude, min_latitude=None, max_latitude=None, 
                   min_longitude=None, max_longitude=None, min_depth=None, max_depth=None):
        params = {
            "format": "text",
            "starttime": start_time,
            "endtime": end_time,
            "minmagnitude": min_magnitude,
            "maxmagnitude": max_magnitude,
        }
        if min_latitude is not None:
            params["minlatitude"] = min_latitude
        if max_latitude is not None:
            params["maxlatitude"] = max_latitude
        if min_longitude is not None:
            params["minlongitude"] = min_longitude
        if max_longitude is not None:
            params["maxlongitude"] = max_longitude
        if min_depth is not None:
            params["mindepth"] = min_depth
        if max_depth is not None:
            params["maxdepth"] = max_depth
        return params

    def send_request(self, params):
        response = requests.get(self.base_url, params=params)
        if response.status_code != 200:
            logger.error(f"Request failed with status code: {response.status_code}")
            logger.error(f"Response content: {response.text}")
            return None
        content_type = response.headers.get('Content-Type')
        if 'application/json' in content_type:
            try:
                return response.json()
            except ValueError as e:
                logger.error(f"Failed to parse JSON: {e}")
                logger.error(f"Response content: {response.text}")
                return None
        elif 'text/plain' in content_type or 'text/csv' in content_type:
            return response.text
        else:
            logger.error(f"Unsupported content type: {content_type}")
            return None

    def get_events(self, start_time, end_time, min_magnitude, max_magnitude, output_file, 
                                    min_latitude=None, max_latitude=None, min_longitude=None, max_longitude=None, min_depth=None, max_depth=None):
        params = self.get_params(start_time, end_time, min_magnitude, max_magnitude, min_latitude, 
                                 max_latitude, min_longitude, max_longitude, min_depth, max_depth)

        # Send request
        data = self.send_request(params)
        if data is None:
            return

        # Extract earthquake information
        earthquakes = self.extract_earthquake_data(data)

        # Convert to DataFrame
        df = pd.DataFrame(earthquakes)

        # Save as CSV file
        df.to_csv(output_file, index=False)
        logger.info(f"Earthquake catalog saved to {output_file}")

    @abstractmethod
    def extract_earthquake_data(self, data):
        """
        Extract earthquake data from the response.

        Args:
            data (str or dict): Response data.

        Returns:
            list: List of dictionaries containing earthquake information.
        """
        pass