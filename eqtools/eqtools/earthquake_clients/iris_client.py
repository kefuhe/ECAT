import pandas as pd
from datetime import datetime
import pytz
from .fdsn_client import FDSNClient
from .logging_config import logger


class IRISClient(FDSNClient):
    def __init__(self):
        super().__init__("http://service.iris.edu/fdsnws/event/1/query")

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

    def extract_earthquake_data(self, data):
        earthquakes = []
        # Debugging
        logger.debug('Preview of data:')
        logger.debug(data[:100])
        for line in data.splitlines():
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split('|')
            if len(fields) < 12:
                continue
            try:
                earthquake = {
                    "time": datetime.strptime(fields[1], '%Y-%m-%dT%H:%M:%S').replace(tzinfo=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S'),
                    "latitude": float(fields[2]),
                    "longitude": float(fields[3]),
                    "depth": float(fields[4]),
                    "magnitude": float(fields[10]),
                    "place": fields[12] if len(fields) > 12 else "",
                }
                earthquakes.append(earthquake)
            except (ValueError, IndexError) as e:
                logger.error(f"Error parsing line: {line}")
                logger.error(e)
        return earthquakes