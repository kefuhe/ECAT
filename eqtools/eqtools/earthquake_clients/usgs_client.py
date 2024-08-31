import requests
import pandas as pd
from datetime import datetime
import pytz
from concurrent.futures import ThreadPoolExecutor
from .fdsn_client import FDSNClient
from .logging_config import logger


class USGSClient(FDSNClient):
    def __init__(self, include_focal_mechanism=False):
        super().__init__("https://earthquake.usgs.gov/fdsnws/event/1/query")
        self.include_focal_mechanism = include_focal_mechanism

    def get_params(self, start_time, end_time, min_magnitude, max_magnitude, min_latitude=None, max_latitude=None, min_longitude=None, max_longitude=None, min_depth=None, max_depth=None):
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
        except requests.RequestException as e:
            logger.error(f"Error fetching detail from {detail_url}: {e}")
        return None

    def extract_earthquake_data(self, data):
        earthquakes = []
        detail_urls = []
        earthquakes_with_details = []

        for feature in data.get("features", []):
            properties = feature.get("properties", {})
            geometry = feature.get("geometry", {})
            coordinates = geometry.get("coordinates", [None, None, None])
            focal_mechanism = None
            nodal_planes = None

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
                "nodal_planes": nodal_planes
            }
            earthquakes.append(earthquake)

            if self.include_focal_mechanism and "detail" in properties:
                detail_urls.append(properties["detail"])
                earthquakes_with_details.append(earthquake)

        if self.include_focal_mechanism:
            with ThreadPoolExecutor(max_workers=10) as executor:
                details = list(executor.map(self.fetch_detail, detail_urls))

            for earthquake, detail_data in zip(earthquakes_with_details, details):
                if detail_data and "moment-tensor" in detail_data.get("properties", {}).get("products", {}):
                    moment_tensor = detail_data["properties"]["products"]["moment-tensor"][0]["properties"]
                    logger.debug(f"moment_tensor: {moment_tensor.keys()}")
                    earthquake["nodal_planes"] = {
                        "nodal_plane_1": {
                            "strike": moment_tensor.get("nodal-plane-1-strike"),
                            "dip": moment_tensor.get("nodal-plane-1-dip"),
                            "rake": moment_tensor.get("nodal-plane-1-rake")
                        },
                        "nodal_plane_2": {
                            "strike": moment_tensor.get("nodal-plane-2-strike"),
                            "dip": moment_tensor.get("nodal-plane-2-dip"),
                            "rake": moment_tensor.get("nodal-plane-2-rake")
                        }
                    }

        return earthquakes

# Example usage
if __name__ == "__main__":
    include_focal_mechanism = True  # Set to False if you don't want to include focal mechanisms
    client = USGSClient(include_focal_mechanism=include_focal_mechanism)
    start_time = "2023-01-01"
    end_time = "2023-12-31"
    min_magnitude = 5.0
    max_magnitude = 9.0
    output_file = "usgs_earthquake_catalog.csv"
    client.download_earthquake_catalog(start_time, end_time, min_magnitude, max_magnitude, output_file)