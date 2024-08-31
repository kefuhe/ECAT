import requests
import pandas as pd
from datetime import datetime
import pytz
from .fdsn_client import FDSNClient
from .logging_config import logger


class USGSClient(FDSNClient):
    def __init__(self):
        super().__init__("https://earthquake.usgs.gov/fdsnws/event/1/query")

    def get_params(self, start_time, end_time, min_magnitude, max_magnitude, min_latitude=None, max_latitude=None, min_longitude=None, max_longitude=None, min_depth=None, max_depth=None, include_focal_mechanism=False):
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
        if include_focal_mechanism:
            params["producttype"] = "focal-mechanism"
        return params

    def extract_earthquake_data(self, data):
        earthquakes = []
        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [None, None, None])
            focal_mechanism = None
            if "products" in properties and "focal-mechanism" in properties["products"]:
                focal_mechanism = properties["products"]["focal-mechanism"][0]["properties"]
            earthquake = {
                "time": datetime.fromtimestamp(properties.get("time", 0) / 1000, tz=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
                "latitude": coordinates[1],
                "longitude": coordinates[0],
                "depth": coordinates[2],
                "magnitude": properties.get("mag"),
                "magType": properties.get("magType", 'unknown'),  # Add magnitude type
                "place": properties.get("place"),
                "focal_mechanism": focal_mechanism
            }
            earthquakes.append(earthquake)
        return earthquakes

# Example usage
if __name__ == "__main__":
    client = USGSClient()
    start_time = "2023-01-01"
    end_time = "2023-12-31"
    min_magnitude = 5.0
    max_magnitude = 9.0
    output_file = "usgs_earthquake_catalog.csv"
    include_focal_mechanism = True  # Set to False if you don't want to include focal mechanisms
    client.download_earthquake_catalog(start_time, end_time, min_magnitude, max_magnitude, output_file, include_focal_mechanism=include_focal_mechanism)