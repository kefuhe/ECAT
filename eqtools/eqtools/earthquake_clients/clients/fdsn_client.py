import requests
import pandas as pd
from abc import ABC, abstractmethod
from .logging_config import logger


class FDSNClient(ABC):
    def __init__(self, base_url):
        self.base_url = base_url

    @abstractmethod
    def get_params(self, start_time, end_time, min_magnitude, max_magnitude, min_latitude=None, max_latitude=None, 
                   min_longitude=None, max_longitude=None, min_depth=None, max_depth=None):
        pass

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
        try:
            params = self.get_params(start_time, end_time, min_magnitude, max_magnitude, min_latitude, 
                                    max_latitude, min_longitude, max_longitude, min_depth, max_depth)

            # Send request
            data = self.send_request(params)
            if data is None:
                logger.error('No data received from the FDSN web server')
                return

            # Extract earthquake information
            earthquakes = self.extract_earthquake_data(data)
            if not earthquakes:
                logger.warning('No earthquakes found in the data')
                return

            # Convert to DataFrame
            self.events = pd.DataFrame(earthquakes)
            logger.debug(f'Preview of earthquake data:\n{self.events.head()}')

            # Save as CSV file
            self.to_csv(output_file, sort_by=['time'], ascending=False)
        except Exception as e:
            logger.error(f"An error occurred while fetching earthquake data: {e}")

    def to_csv(self, output_file, sort_by=None, ascending=True):
        if self.events is None:
            logger.error("No data available to export. Please fetch the data first.")
            return
    
        # Convert time column to datetime if sorting by time
        if sort_by and 'time' in sort_by:
            self.events['time'] = pd.to_datetime(self.events['time'])
            logger.debug('Converted time column to datetime.')
    
        # Sort events if sort_by is provided
        if sort_by:
            valid_keys = ['time', 'latitude', 'longitude', 'depth', 'magnitude', 'magType', 'place']
            sort_keys = [key for key in sort_by if key in valid_keys]
            if sort_keys:
                events = self.events.sort_values(by=sort_keys, ascending=ascending)
                logger.debug(f'Sorted earthquake data by {sort_keys}:\n{events.head()}')
            else:
                logger.warning(f'No valid sort keys found in {sort_by}. Skipping sorting.')
        else:
            events = self.events
    
        # Save as CSV file
        events.to_csv(output_file, index=False)
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