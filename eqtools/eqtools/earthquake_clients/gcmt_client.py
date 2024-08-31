import requests
import pandas as pd
from datetime import datetime
import pytz
import re
import numpy as np
from urllib.parse import urljoin
from .fdsn_client import FDSNClient
from .logging_config import logger


class GCMTClient(FDSNClient):
    def __init__(self):
        super().__init__("http://www.globalcmt.org/cgi-bin/globalcmt-cgi-bin/CMT5/form")

    def get_params(self, start_time, end_time, min_magnitude, max_magnitude, min_depth, max_depth, min_latitude, max_latitude, min_longitude, max_longitude):
        yearbeg, monbeg, daybeg = map(int, start_time.split('-'))
        yearend, monend, dayend = map(int, end_time.split('-'))
        return {
            'itype': 'ymd',
            'yr': yearbeg, 'mo': monbeg, 'day': daybeg,
            'otype': 'ymd',
            'oyr': yearend, 'omo': monend, 'oday': dayend,
            'jyr': 1976, 'jday': 1, 'ojyr': 1976, 'ojday': 1, 'nday': 1,
            'lmw': min_magnitude, 'umw': max_magnitude,
            'lms': 0, 'ums': 10,
            'lmb': 0, 'umb': 10,
            'llat': min_latitude, 'ulat': max_latitude,
            'llon': min_longitude, 'ulon': max_longitude,
            'lhd': min_depth, 'uhd': max_depth,
            'lts': -9999, 'uts': 9999,
            'lpe1': 0, 'upe1': 90,
            'lpe2': 0, 'upe2': 90,
            'list': 5
        }

    def download_earthquake_catalog(self, start_time, end_time, min_magnitude, max_magnitude, output_file, min_depth=0, max_depth=1000, min_latitude=-90, max_latitude=90, min_longitude=-180, max_longitude=180):
        """
        Download earthquake catalog from GCMT and save it as a CSV file.

        Args:
            start_time (str): Start time in 'YYYY-MM-DD' format.
            end_time (str): End time in 'YYYY-MM-DD' format.
            min_magnitude (float): Minimum magnitude.
            max_magnitude (float): Maximum magnitude.
            output_file (str): Output CSV file path.
            min_depth (float): Minimum depth in kilometers.
            max_depth (float): Maximum depth in kilometers.
            min_latitude (float): Minimum latitude.
            max_latitude (float): Maximum latitude.
            min_longitude (float): Minimum longitude.
            max_longitude (float): Maximum longitude.
        """
        all_earthquakes = []
        base_url = self.base_url
        params = self.get_params(start_time, end_time, min_magnitude, max_magnitude, min_depth, max_depth, min_latitude, max_latitude, min_longitude, max_longitude)

        while True:
            # Send GET request
            response = requests.get(base_url, params=params)
            if response.status_code != 200:
                logger.error(f"Request failed with status code: {response.status_code}")
                logger.error(f"Response content: {response.text}")
                break

            # Parse response data
            data = response.text.splitlines()
            # Debugging
            logger.debug(f'Preview of data: {data[:100]}')

            # Extract earthquake information
            earthquakes, more = self._parse_events_page(data)

            if not earthquakes:
                break

            all_earthquakes.extend(earthquakes)

            if more:
                base_url = urljoin(self.base_url, more)  # Update base_url to the "more" link
                params = {}  # Clear params as the "more" link already contains all necessary parameters
            else:
                break

        # Convert to DataFrame
        df = pd.DataFrame(all_earthquakes)

        # Save as CSV file
        df.to_csv(output_file, index=False)
        logger.info(f"Earthquake catalog saved to {output_file}")

    def _parse_events_page(self, lines):
        state = 0
        events = []
        data_obj = None
        more = None

        def complete(data):
            try:
                t = datetime.strptime(f"{data['year']}-{data['month']}-{data['day']} {data['hour']}:{data['minute']}:{int(data['seconds'])}", '%Y-%m-%d %H:%M:%S')
                t = t.replace(tzinfo=pytz.UTC).strftime('%Y-%m-%d %H:%M:%S')

                earthquake = {
                    "time": t,
                    "latitude": data['lat'],
                    "longitude": data['lon'],
                    "depth": data['depth_km'],  # Depth is already in kilometers
                    "magnitude": data['magnitude'],
                    "magType": data.get("magType", 'unknown'), 
                    "place": data['region'],
                    "focal_mechanism": {
                        "mrr": data['mrr'],
                        "mtt": data['mtt'],
                        "mpp": data['mpp'],
                        "mrt": data['mrt'],
                        "mrp": data['mrp'],
                        "mtp": data['mtp'],
                        "exponent": data['exponent']
                    },
                    "fault_plane1": data.get("fault_plane1"),
                    "fault_plane2": data.get("fault_plane2")
                }
                events.append(earthquake)
            except KeyError as e:
                logger.error(f"Error completing data: {e}")

        for line in lines:
            if state == 0:
                m = re.search(r'<a href="([^"]+)">More solutions', line)
                if m:
                    more = m.group(1)

                m = re.search(r'Event name:\s+(\S+)', line)
                if m:
                    if data_obj:
                        complete(data_obj)
                    data_obj = {}
                    data_obj['eventname'] = m.group(1)

                if data_obj:
                    m = re.search(r'Region name:\s+([^<]+)', line)
                    if m:
                        data_obj['region'] = m.group(1).strip()

                    m = re.search(r'Date \(y/m/d\): (\d+)/(\d+)/(\d+)', line)
                    if m:
                        data_obj['year'], data_obj['month'], data_obj['day'] = map(int, m.groups())

                    m = re.search(r'Timing and location information', line)
                    if m:
                        state = 1

            elif state == 1:
                toks = line.split()
                if toks and toks[0] == 'CMT':
                    data_obj['hour'], data_obj['minute'] = map(int, toks[1:3])
                    data_obj['seconds'], data_obj['lat'], data_obj['lon'], data_obj['depth_km'] = map(float, toks[3:])
                    state = 2

            elif state == 2:
                m = re.search(r'Exponent for moment tensor:\s+(\d+)', line)
                if m:
                    data_obj['exponent'] = int(m.group(1))

                toks = line.split()
                if toks and toks[0] == 'CMT':
                    data_obj['mrr'], data_obj['mtt'], data_obj['mpp'], data_obj['mrt'], data_obj['mrp'], data_obj['mtp'] = map(float, toks[1:])
                    data_obj['magnitude'] = self.calculate_magnitude(data_obj)

                m = re.search(r'Mw = ([\d.]+)', line)
                if m:
                    data_obj['magType'] = 'Mw'
                    data_obj['magnitude'] = float(m.group(1))

                m = re.search(r'Fault plane:  strike=(\d+)\s+dip=(\d+)\s+slip=([-\d]+)', line)
                if m:
                    fault_plane = {
                        "strike": int(m.group(1)),
                        "dip": int(m.group(2)),
                        "slip": int(m.group(3))
                    }
                    if 'fault_plane1' not in data_obj:
                        data_obj['fault_plane1'] = fault_plane
                    else:
                        data_obj['fault_plane2'] = fault_plane
                        state = 0

        if data_obj:
            complete(data_obj)

        return events, more

    def calculate_magnitude(self, data):
        """
        Calculate the magnitude of an earthquake.

        Args:
            data (dict): Dictionary containing earthquake data.

        Returns:
            float: Calculated magnitude.
        """
        mrr, mtt, mpp, mrt, mrp, mtp = data['mrr'], data['mtt'], data['mpp'], data['mrt'], data['mrp'], data['mtp']
        exponent = data['exponent']
        moment = np.sqrt((mrr**2 + mtt**2 + mpp**2 + 2*(mrt**2 + mrp**2 + mtp**2)) / 2) * 10**(exponent - 7)
        magnitude = (2 / 3) * np.log10(moment * 1e7) - 10.7
        return magnitude
    
    def extract_earthquake_data(self, data):
        """
        Extract earthquake data from the response.

        Args:
            data (dict): Response data.

        Returns:
            list: List of dictionaries containing earthquake information.
        """
        return self._parse_events_page(data)[0]