import requests
import pandas as pd
from datetime import datetime
import pytz
import re
import numpy as np
from urllib.parse import urljoin
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from .fdsn_client import FDSNClient
from .logging_config import logger

class USGSClient(FDSNClient):
    def __init__(self, include_focal_mechanism=False):
        super().__init__("https://earthquake.usgs.gov/fdsnws/event/1/query")
        self.include_focal_mechanism = include_focal_mechanism

    def get_params(self, start_time, end_time, min_magnitude, max_magnitude, min_latitude=None, max_latitude=None, 
                   min_longitude=None, max_longitude=None, min_depth=None, max_depth=None):
        params = {
            "format": "geojson",
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

    def fetch_detail(self, detail_url):
        try:
            response = requests.get(detail_url)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Request failed with status code: {response.status_code}")
                logger.error(f"Response content: {response.text}")
        except requests.RequestException as e:
            logger.error(f"Error fetching detail from {detail_url}: {e}")
        return None

    def extract_earthquake_data(self, data):
        if not isinstance(data, dict):
            logger.error("Response content is not a valid JSON object")
        earthquakes = []
        detail_urls = []
        earthquakes_with_details = []

        def process_feature(feature):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [None, None, None])
            focal_mechanism = None
            nodal_plane1 = None
            nodal_plane2 = None

            if self.include_focal_mechanism and "products" in properties and "focal-mechanism" in properties["products"]:
                focal_mechanism = properties["products"]["focal-mechanism"][0]["properties"]

            earthquake = {
                "time": datetime.fromtimestamp(properties.get("time", 0) / 1000, tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
                "latitude": coordinates[1],
                "longitude": coordinates[0],
                "depth": coordinates[2],
                "magnitude": properties.get("mag"),
                "magType": properties.get("magType", 'unknown'),  # Add magnitude type
                "place": properties.get("place"),
                "focal_mechanism": focal_mechanism,
                "nodal_plane1": nodal_plane1,
                "nodal_plane2": nodal_plane2
            }
            return earthquake, properties.get("detail")

        # First parallel processing for basic information extraction
        cpu_count = os.cpu_count()
        with ThreadPoolExecutor(max_workers=cpu_count) as executor:
            futures = [executor.submit(process_feature, feature) for feature in data.get("features", [])]

            for future in as_completed(futures):
                try:
                    earthquake, detail_url = future.result()
                    earthquakes.append(earthquake)
                    logger.debug(f'Details URL: {detail_url}')
                    if self.include_focal_mechanism and detail_url:
                        detail_urls.append(detail_url)
                        earthquakes_with_details.append(earthquake)
                except Exception as e:
                    logger.error(f"Error processing feature: {e}")

        # Second parallel processing for detailed information extraction
        if self.include_focal_mechanism:
            with ThreadPoolExecutor(max_workers=cpu_count) as executor:
                details = list(executor.map(self.fetch_detail, detail_urls))

            for earthquake, detail_data in zip(earthquakes_with_details, details):
                if detail_data and "moment-tensor" in detail_data.get("properties", {}).get("products", {}):
                    moment_tensor = detail_data["properties"]["products"]["moment-tensor"][0]["properties"]
                    logger.debug(f"moment_tensor: {moment_tensor.keys()}")
                    earthquake["nodal_plane1"] = {
                        "strike": moment_tensor.get("nodal-plane-1-strike"),
                        "dip": moment_tensor.get("nodal-plane-1-dip"),
                        "rake": moment_tensor.get("nodal-plane-1-rake")
                    }
                    earthquake["nodal_plane2"] = {
                        "strike": moment_tensor.get("nodal-plane-2-strike"),
                        "dip": moment_tensor.get("nodal-plane-2-dip"),
                        "rake": moment_tensor.get("nodal-plane-2-rake")
                    }
        
        # Sort earthquakes by time in descending order (latest first)
        earthquakes = sorted(earthquakes, key=lambda x: datetime.strptime(x["time"], '%Y-%m-%d %H:%M:%S'), reverse=True)

        return earthquakes