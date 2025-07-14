'''
A class that deals with seismic catalogs.
This class can also deal with moment tensors.

Written by R. Jolivet, April 2013.

Modified by Kefeng He in 2022-2024 to add more functionalities.
'''

# Externals
import numpy as np
import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt
from matplotlib.path import Path
import copy
from sklearn.preprocessing import StandardScaler
from scipy.stats import linregress

# Personals
from .SourceInv import SourceInv
from .seismic_clustering import cluster_method_factory, linear_detection_method_factory


class seismiclocations(SourceInv):

    '''
    A class that handles a simple earthquake catalog

    Args:
       * name      : Name of the dataset.

    Kwargs:
       * utmzone   : UTM zone  (optional, default=None)
       * lon0      : Longitude of the center of the UTM zone
       * lat0      : Latitude of the center of the UTM zone
       * ellps     : ellipsoid (optional, default='WGS84')

    '''

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):

        # Base class init
        super(seismiclocations, self).__init__(name, 
                                               utmzone=utmzone, 
                                               ellps=ellps,
                                               lon0=lon0,
                                               lat0=lat0)

        # Initialize the data set type
        self.dtype = 'seismiclocations'
        self.cluster_method_factory = cluster_method_factory
        self.linear_detection_method_factory = linear_detection_method_factory
        self.cluster_labels = {}

        if verbose:
            print ("---------------------------------")
            print ("---------------------------------")
            print ("Initialize Seismicity data set {}".format(self.name))

        # Initialize the location

        # Initialize some things
        self.time = None
        self.lon = None
        self.lat = None
        self.depth = None
        self.mag = None

        # All done
        return

    def read_from_Hauksson(self,filename, header=0, lat_first=True):
        '''
        Read the Seismic catalog from the SCSN networks (Template from E. Hauksson, Caltech). File format is as follows

        +------+-------+-----+------+--------+---------+-----+-----+-------+-----------+
        | year | month | day | hour | minute | seconds | lat | lon | depth | magnitude |
        +------+-------+-----+------+--------+---------+-----+-----+-------+-----------+
        | int  | int   | int | int  | int    | float   |float|float| float | float     |
        +------+-------+-----+------+--------+---------+-----+-----+-------+-----------+

        Args:
            * filename      : Name of the input file.

        Kwargs:
            * header        : Size of the header.

        Returns:
            * None
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename,'r')

        # Read it all
        A = fin.readlines()

        # Initialize the business
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Read the header to figure out where is the magnitude
        desc = A[header-2].split()
        #imag = np.flatnonzero(np.array(desc)=='MAG').tolist()[0]
        #imag += 4

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            # Split the string line
            tmp = A[i].split()

            # Get the values
            yr = int(tmp[0])
            mo = int(tmp[1])
            da = int(tmp[2])
            hr = int(tmp[3])
            mi = int(tmp[4])
            sd = int(np.floor(float(tmp[5])))
            if lat_first:
                lat = float(tmp[6])
                lon = float(tmp[7])
            else:
                lon = float(tmp[6])
                lat = float(tmp[7])
            depth = float(tmp[8])
            mag = float(tmp[9])

            # Create the time object
            d = dt.datetime(yr, mo, da, hr, mi, sd)

            # Store things in self
            self.time.append(d)
            self.lat.append(lat)
            self.lon.append(lon)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lat = np.array(self.lat)
        self.lon = np.array(self.lon)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm coordinates
        self.lonlat2xy()

        # All done
        return

    def read_from_Rietbrock(self, filename, header=1):
        '''
        Read the Seismic catalog from the NCSN networks (Template from F. Waldhauser).

        +-----+------+-------+-----+------+--------+---------+-----------+------------+------------+------------+-------+-----------+
        | id  | year | month | day | hour | minute | seconds | degre lat | minute lat | degree lon | minute lon | depth | magnitude |
        +-----+------+-------+-----+------+--------+---------+-----------+------------+------------+------------+-------+-----------+
        |     | int  |  int  | int | int  |  int   |  float  |   float   |   float    |    float   |    float   | float |   float   |
        +-----+------+-------+-----+------+--------+---------+-----------+------------+------------+------------+-------+-----------+

        Args:
            * filename      : Name of the input file. 

        Kwargs:
            * header        : Size of the header.

        Returns:
            * None
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename,'r')

        # Read it all
        A = fin.readlines()

        # Initialize the business
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):

            # Split the string line 
            tmp = A[i].split()

            # Get the values
            yr = int(tmp[1])
            mo = int(tmp[2])
            da = int(tmp[3])
            hr = int(tmp[4])
            mi = int(tmp[5])
            sd = int(np.floor(float(tmp[6])))
            lat = float(tmp[7][:-1]) + float(tmp[8])/60.
            if tmp[7][-1] in ('s', 'S'):
                lat *= -1.
            lon = float(tmp[9][:-1]) + float(tmp[10])/60.
            if tmp[9][-1] in ('w', 'W'):
                lon *= -1.
            depth = float(tmp[11])
            mag = float(tmp[12])

            # Create the time object
            d = dt.datetime(yr, mo, da, hr, mi, sd)
            
            # Store things in self 
            self.time.append(d)
            self.lat.append(lat)
            self.lon.append(lon)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lat = np.array(self.lat)
        self.lon = np.array(self.lon)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm coordinates
        self.lonlat2xy()

        # All done
        return

    def read_from_SCSN(self,filename, header=65):
        '''
        Read the Seismic catalog from the SCSN networks (Template from F. Waldhauser).

        +------+-------+-----+------+--------+-----+-----+-------+-----------+
        | year | month | day | hour | minute | lat | lon | depth | magnitude |
        +------+-------+-----+------+--------+-----+-----+-------+-----------+
        | int  |  int  | int | int  |  int   |float|float| float |   float   |
        +------+-------+-----+------+--------+-----+-----+-------+-----------+

        Args:
            * filename      : Name of the input file.

        Kwargs:
            * header        : Size of the header.

        Returns:
            * None
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename,'r')

        # Read it all
        A = fin.readlines()

        # Initialize the business
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Read the header to figure out where is the magnitude
        desc = A[header-2].split()
        #imag = np.flatnonzero(np.array(desc)=='MAG').tolist()[0]
        #imag += 4

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            # Split the string line
            tmp = A[i].split()

            # Get the values
            yr = int(tmp[0])
            mo = int(tmp[1])
            da = int(tmp[2])
            hr = int(tmp[3])
            mi = int(tmp[4])
            lat = float(tmp[7])
            lon = float(tmp[8])
            depth = float(tmp[9])
            mag = float(tmp[10])

            # Create the time object
            if mi>=60:
                mi = 59
            if mi<=-1:
                mi = 0
            d = dt.datetime(yr, mo, da, hr, mi)

            # Store things in self
            self.time.append(d)
            self.lat.append(lat)
            self.lon.append(lon)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lat = np.array(self.lat)
        self.lon = np.array(self.lon)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm coordinates
        self.lonlat2xy()

        # All done
        return

    def read_from_NCSN(self,filename, header=65):
        '''
        Read the Seismic catalog from the NCSN networks. Magnitude is in a column determined from the header.
        The rest reads as

        +------+-------+-----+------+--------+-----+-----+-------+
        | year | month | day | hour | minute | lat | lon | depth |
        +------+-------+-----+------+--------+-----+-----+-------+
        | int  |  int  | int | int  |  int   |float|float| float |
        +------+-------+-----+------+--------+-----+-----+-------+

        Args:
            * filename      : Name of the input file.

        Kwargs:
            * header        : Size of the header.

        Returns:
            * None
        '''

        print ("Read from file {} into data set {}".format(filename, self.name))

        # Open the file
        fin = open(filename,'r')

        # Read it all
        A = fin.readlines()

        # Initialize the business
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Read the header to figure out where is the magnitude
        desc = A[header-2].split()
        imag = np.flatnonzero(np.array(desc)=='MAG').tolist()[0]
        imag += 4

        # Loop over the A, there is a header line header
        for i in range(header, len(A)):
            # Split the string line
            tmp = A[i].split()

            # Get the values
            yr = int(tmp[0])
            mo = int(tmp[1])
            da = int(tmp[2])
            hr = int(tmp[3])
            mi = int(tmp[4])
            lat = np.float64(tmp[6])
            lon = np.float64(tmp[7])
            depth = np.float64(tmp[8])
            mag = np.float64(tmp[imag])

            # Create the time object
            d = dt.datetime(yr, mo, da, hr, mi)

            # Store things in self
            self.time.append(d)
            self.lat.append(lat)
            self.lon.append(lon)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lat = np.array(self.lat)
        self.lon = np.array(self.lon)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm coordinates
        self.lonlat2xy()

        # All done
        return

    def read_csi(self, infile, header=0):
        '''
        Reads data from a file written by csi.seismiclocation.write2file
        
        +-----+-----+-------+-----------+----------------------------+
        | lon | lat | depth | magnitude | time (isoformat)           |
        +-----+-----+-------+-----------+----------------------------+
        |float|float| float |   float   | yyy-mm-ddThh:mm:ss.ss      |
        +-----+-----+-------+-----------+----------------------------+

        Args:
            * infile    : input file
            
        Kwargs:
            * header    : length of the header

        Returns:
            * None
        '''

        # open the file
        fin = open(infile, 'r')

        # Read all
        All = fin.readlines()

        # Initialize things
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []

        # Loop
        for i in range(header, len(All)):

            # Get the splitted string
            tmp = All[i].split()

            # Get values
            time = dt.datetime.strptime(tmp[4], "%Y-%m-%dT%H:%M:%S.%f")
            lon = float(tmp[0])
            lat = float(tmp[1])
            depth = float(tmp[2])
            mag = float(tmp[3])

            # Store
            self.time.append(time)
            self.lon.append(lon)
            self.lat.append(lat)
            self.depth.append(depth)
            self.mag.append(mag)

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)

        # Create the utm
        self.lonlat2xy()

        # All done
        return

    def read_ascii(self, infile, header=0, delimiter=None, lat_lon_order=True, time_formats=None):
        '''
        Reads data from an ascii file.
    
        +----------------------------+-----+-----+-------+-----------+
        | time (isoformat)           | lat | lon | depth | magnitude |
        +----------------------------+-----+-----+-------+-----------+
        | yyy-mm-ddThh:mm:ss.ss      |float|float| float |   float   |
        +----------------------------+-----+-----+-------+-----------+
    
        Args:
            * infile        : input file
                    
        Kwargs:
            * header        : length of the header
            * delimiter     : delimiter used in the input file
            * lat_lon_order : order of latitude and longitude in the input file
            * time_formats  : list of possible time formats (default includes common formats)
    
        Returns:
            * None
        '''
    
        # Default time formats if none are provided
        if time_formats is None:
            time_formats = [
                "%Y-%m-%dT%H:%M:%S.%fZ",  # ISO format with microseconds and Z
                "%Y-%m-%d %H:%M:%S.%f",   # Standard format with microseconds
                "%Y-%m-%d %H:%M:%S",      # Standard format without microseconds
                "%Y/%m/%d %H:%M:%S",      # Alternative format with slashes
            ]
    
        # Open the file
        with open(infile, 'r') as fin:
            # Read all lines
            All = fin.readlines()
    
        # Initialize attributes
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []
    
        # Loop through the file, skipping the header
        for i in range(header, len(All)):
            # Split the line using the specified delimiter
            tmp = All[i].strip().split(delimiter)
    
            # Skip empty lines or lines with insufficient columns
            if not tmp or len(tmp) < 5:
                continue
    
            # Parse time with multiple formats
            time = None
            for fmt in time_formats:
                try:
                    time = dt.datetime.strptime(tmp[0], fmt)
                    break
                except ValueError:
                    continue
    
            # Raise an error if no format matches
            if time is None:
                raise ValueError(f"Error parsing time '{tmp[0]}' with available formats: {time_formats}")
    
            # Parse other values
            if lat_lon_order:
                lat = float(tmp[1])
                lon = float(tmp[2])
            else:
                lon = float(tmp[1])
                lat = float(tmp[2])
            depth = float(tmp[3])
            mag = float(tmp[4])
    
            # Append to lists
            self.time.append(time)
            self.lon.append(lon)
            self.lat.append(lat)
            self.depth.append(depth)
            self.mag.append(mag)
    
        # Convert lists to numpy arrays
        self.time = np.array(self.time)
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.depth = np.array(self.depth)
        self.mag = np.array(self.mag)
    
        # Create UTM coordinates
        self.lonlat2xy()
    
        # All done
        return
    
    def read_CMTSolutions(self, infile):
        '''
        Reads data and moment tensors from an ascii file listing CMT solutions format. Go check the GCMT webpage for format description

        Args:
            * infile: Input file.

        Returns:
            * None
        '''

        # open the file
        fin = open(infile, 'r')

        # Read all
        All = fin.readlines()

        # Initialize things
        self.time = []
        self.lon = []
        self.lat = []
        self.depth = []
        self.mag = []
        self.CMTinfo = []

        # Initialize counter
        i = 0

        # Loop over the lines
        while i<len(All):

            # split the line
            line = All[i].split()

            # Check if line is empty
            if len(line)>0:

                # Check the first character
                if line[0][0] in ('P'):

                    # Time
                    yr = int(line[0][4:])
                    mo = int(line[1])
                    da = int(line[2])
                    hr = int(line[3])
                    mn = int(line[4])
                    sd = int(float(line[5]))
                    time = dt.datetime(yr, mo, da, hr, mn, sd)

                    # cmt informations
                    info = {}
                    i += 1
                    for j in range(12):
                        line = All[i].split(':')
                        name = line[0]
                        value = line[1].split()[0]
                        if name not in ('event name'):
                            value = float(value)
                        info[name] = value
                        i += 1

                    # Get values
                    lat = info['latitude']
                    lon = info['longitude']
                    depth = info['depth']

                    # set in self
                    self.time.append(time)
                    self.lon.append(lon)
                    self.lat.append(lat)
                    self.depth.append(depth)
                    self.CMTinfo.append(info)

                # Else
                else:
                    i += 1

            # Else
            else:
                i += 1

        # Close the file
        fin.close()

        # Make arrays
        self.time = np.array(self.time)
        self.lon = np.array(self.lon)
        self.lat = np.array(self.lat)
        self.depth = np.array(self.depth)

        # Compute the magnitudes
        self.Cmt2Dislocation(size=1e-1, mu=44e9, choseplane='nochoice', 
                             moment_from_tensor=True)
        self.Mo2mag()

        # Create the utm
        self.lonlat2xy()

        # All done
        return
    
    def read_from_HypoDD(self, infile, header=0, delimiter='\s+'):
        '''
        Reads data from a file from HypoDD (as given by to me by Dirk Beker).

        Format: event no, lat, lon, depth, internal_x, internal_y, internal_z, err_x, err_y, err_z, year, month, day, hour, minute, second, magnitude, no CC P-times, no CC S-times, no CT P-times, no CT S-times, rms value CC data, rms value CT data, cluster ID

        Args:
            * infile    : input file
            
        Kwargs:
            * header    : length of the header

        Returns:
            * None
        '''

        # Open and load in Pandas
        with open(infile, 'r') as fin:
            data = pd.read_csv(fin, header=header, delimiter=delimiter, names=['event no', 'lat', 'lon', 'depth', 
                                                                               'internal_x', 'internal_y', 'internal_z',
                                                                               'err_x', 'err_y', 'err_z', 
                                                                               'year', 'month', 'day', 'hour', 'minute', 'second', 
                                                                               'mag',
                                                                               'no CC P-times', 'no CC S-times', 
                                                                               'no CT P-times', 'no CT S-times', 
                                                                               'rms value CC data', 'rms value CT data', 'cluster ID'])

        # Initialize things
        self.time = np.array([dt.datetime(y, m, d, h, mi) + dt.timedelta(seconds=s) for y, m, d, h, mi, s in zip(data['year'],
                                                                                                           data['month'],
                                                                                                           data['day'],
                                                                                                           data['hour'],
                                                                                                           data['minute'],
                                                                                                           data['second'])])
        self.lon = data['lon'].values
        self.lat = data['lat'].values
        self.depth = data['depth'].values
        self.mag = data['mag'].astype(float).values

        # Save data
        self.data = data

        # Create the utm
        self.lonlat2xy()

        # All done
        return

    def read_from_dirkBecker(self, infile, header=0, delimiter='\s+'):
        '''
        Reads data from a file given to me by Dirk Becker.

        Format: year,month,day,hour,minute,second,latitude,longitude,depth,magnitude,length of major error ellipsoid half axis,rms value

        Args:
            * infile    : input file
            
        Kwargs:
            * header    : length of the header

        Returns:
            * None
        '''

        # Open and load in Pandas
        with open(infile, 'r') as fin:
            data = pd.read_csv(fin, header=header, delimiter=delimiter, names=['year', 'month', 'day', 'hour', 'minute', 'second', 
                                                                    'lat', 'lon', 'depth', 'mag', 'err', 'rms'])
        
        # Do some time management
        data['time'] = [dt.datetime(y, m, d, h, mi) + dt.timedelta(seconds=s) for y, m, d, h, mi, s in zip(data['year'], 
                                                                                                           data['month'], 
                                                                                                           data['day'], 
                                                                                                           data['hour'], 
                                                                                                           data['minute'], 
                                                                                                           data['second'])]

        # Initialize things
        self.time = np.array([dt.datetime(y, m, d, h, mi) + dt.timedelta(seconds=s) for y, m, d, h, mi, s in zip(data['year'],
                                                                                                           data['month'],
                                                                                                           data['day'],
                                                                                                           data['hour'],
                                                                                                           data['minute'],
                                                                                                           data['second'])])
        self.lon = data['lon'].values
        self.lat = data['lat'].values
        self.depth = data['depth'].values
        self.mag = data['mag'].astype(float).values
        self.rms = data['rms'].values
        self.err = data['err'].values

        # Save data
        self.data = data

        # Create the utm
        self.lonlat2xy()

        # All done
        return

    def selectbox(self, minlon, maxlon, minlat, maxlat, depth=100000., mindep=0.0):
        '''
        Select the earthquakes in a box defined by min and max, lat and lon.

        Args:
            * minlon        : Minimum longitude.
            * maxlon        : Maximum longitude.
            * minlat        : Minimum latitude.
            * maxlat        : Maximum latitude.

        Kwargs:
            * depth         : Maximum depth
            * mindepth      : Minimum depth

        Returns:
            * None. Direclty kicks out earthquakes that are not in the box
        '''

        # Store the corners
        self.minlon = minlon
        self.maxlon = maxlon
        self.minlat = minlat
        self.maxlat = maxlat

        # Select on latitude and longitude
        print( "Selecting the earthquakes in the box Lon: {} to {} and Lat: {} to {}".format(minlon, maxlon, minlat, maxlat))
        u = np.flatnonzero((self.lat>minlat) & (self.lat<maxlat) & (self.lon>minlon) & (self.lon<maxlon) & (self.depth<depth) & (self.depth>mindep))

        # make the selection
        self._select(u)

        # All done
        return

    def select_polygon(self, polygon, mindep=0.0, maxdep=100000.):
        '''
        Select the earthquakes in a polygon and depth range.

        Args:
            * polygon       : List of (lon, lat) pairs defining the polygon.
            * mindep        : Minimum depth.
            * maxdep        : Maximum depth.

        Returns:
            * None. Directly removes earthquakes that are not in the polygon and depth range.
        '''

        # Create a Path object from the polygon
        path = Path(polygon)

        # Create an array of (lon, lat) pairs
        points = np.column_stack((self.lon, self.lat))

        # Check which points are in the polygon
        in_polygon = path.contains_points(points)

        # Check which points are in the depth range
        in_depth_range = (self.depth > mindep) & (self.depth < maxdep)

        # Find the indices of the points that are in both the polygon and the depth range
        u = np.flatnonzero(in_polygon & in_depth_range)

        # Make the selection
        self._select(u)

        # All done
        return

    def selecttime(self, start=[2001, 1, 1], end=[2101, 1, 1]):
        '''
        Selects the earthquake in between two dates. Dates can be datetime.datetime or lists.

        Args:
            * start     : Beginning of the period [yyyy, mm, dd]
            * end       : End of the period [yyyy, mm, dd]

        Returns:
            * None. Direclty kicks out earthquakes that are not within start-end period
        '''

        # check start and end
        if (start.__class__ is float) or (start.__class__ is int) :
            st = dt.datetime(start, 1, 1)
        if (start.__class__ is list):
            if len(start) == 1:
                st = dt.datetime(start[0], 1, 1)
            elif len(start) == 2:
                st = dt.datetime(start[0], start[1], 1)
            elif len(start) == 3:
                st = dt.datetime(start[0], start[1], start[2])
            elif len(start) == 4:
                st = dt.datetime(start[0], start[1], start[2], start[3])
            elif len(start) == 5:
                st = dt.datetime(start[0], start[1], start[2], start[3], start[4])
            elif len(start) == 6:
                st = dt.datetime(start[0], start[1], start[2], start[3], start[4], start[5])
        if start.__class__ is dt.datetime:
            st = start

        if (end.__class__ is float) or (end.__class__ is int) :
            ed = dt.datetime(int(end), 1, 1)
        if (end.__class__ is list):
            if len(end) == 1:
                ed = dt.datetime(end[0], 1, 1)
            elif len(end) == 2:
                ed = dt.datetime(end[0], end[1], 1)
            elif len(end) == 3:
                ed = dt.datetime(end[0], end[1], end[2])
            elif len(end) == 4:
                ed = dt.datetime(end[0], end[1], end[2], end[3])
            elif len(end) == 5:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4])
            elif len(end) == 6:
                ed = dt.datetime(end[0], end[1], end[2], end[3], end[4], end[5])
        if end.__class__ is dt.datetime:
            ed = end

        # Get values
        print ("Selecting earthquake between {} and {}".format(st.isoformat(),ed.isoformat()))
        u = np.flatnonzero((self.time > st) & (self.time < ed))

        # Selection
        self._select(u)

        # All done
        return

    def selectmagnitude(self, minimum, maximum=10):
        '''
        Selects the earthquakes between two magnitudes.

        Args:
            * minimum   : Minimum earthquake magnitude wanted.

        Kwargs:    
            * maximum   : Maximum earthquake magnitude wanted.

        Returns:
            * None. Directly kicks out earthquakes that are not within the wanted magnitude range
        '''

        # Get the magnitude
        mag = self.mag

        # get indexes
        print ("Selecting earthquake between magnitudes {} and {}".format(minimum, maximum))
        u = np.flatnonzero((self.mag > minimum) & (self.mag < maximum))

        # Selection
        self._select(u)

        # All done
        return

    def computeGR(self, plot=False, bins=20):
        '''
        Plots the Gutemberg-Richter distribution.

        Kwargs:
            * plot      : make a figure
            * bins      : how many bins to use

        Returns:
            * None
        '''

        # Get the magnitude
        mag = self.mag

        # Get the histogram
        h, x = np.histogram(self.mag, bins=bins)
        x = (x[1:] + x[:-1])/2.

        # Store that somewhere
        self.Histogram = [x, h]

        # plot the values
        if plot:
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.semilogy(x, h, '.r', markersize=10, linewidth=1)
            plt.show()

        # All done
        return

    def fitBvalue(self, method='mle', min_mag=None, bin_width=0.1, delta_mag=0):
        '''
        Estimate Gutenberg-Richter b-value using Maximum Likelihood Estimation (MLE) or Least Squares (LS).
    
        Args:
            mags      : array-like, earthquake magnitudes
            method    : 'mle' or 'ls', fitting method ('mle' for maximum likelihood, 'ls' for least squares)
            min_mag   : minimum magnitude to include (default: minimum in mags)
            bin_width : bin width for least squares method (default: 0.1)
            delta_mag : delta magnitude for MLE method (default: 0)
    
        Returns:
            b_value   : estimated b-value
        '''
        if method == 'mle':
            from seismo_tools import calc_b_value_mle
            b = calc_b_value_mle(self.mag, min_mag=min_mag, delta_mag=delta_mag)
            return b
        elif method == 'ls':
            from seismo_tools import calc_b_value_least_squares
            b, a, fit_params = calc_b_value_least_squares(self.mag, bin_width=bin_width, fit_min_mag=min_mag)
            return b, a, fit_params
        else:
            raise ValueError("Method must be 'mle' or 'ls'.")

    def distance2fault(self, faults, distance=5.):
        '''
        Selects the earthquakes that are located less than distance away from the fault plane.

        Args:
            * faults    : List of faults

        Kwargs:
            * distance  : Threshold

        Returns:   
            * None. Selected events are kept. Others are deleted.
        '''

        # Create the list
        u = []

        # holder
        self.distancetofaults = {}

        # Loop over the faults
        for fault in faults:
            dis = np.array(self._getDistance2FaultPlane(fault))
            ut = np.flatnonzero( dis < distance )
            for i in ut:
                u.append(i)
            # Save the distance to the fault for further use
            self.distancetofaults[fault.name] = dis

        # make u an array
        u = np.array(u)
        u = np.unique(u)

        # Selection
        self._select(u)

        # Selection
        for fault in faults:
            self.distancetofaults[fault.name] = self.distancetofaults[fault.name][u]

        # All done
        return

    def distance2trace(self, faults, distance=5.):
        '''
        Selects the earthquakes that are located less than 'distance' km away from a given surface fault trace.

        Args:
            * faults    : list of instances of faults. These need to have {xf} and {yf} attributes

        Kwargs:
            * distance  : threshold distance.

        Returns:
            * None. Selected earthquakes are kept. Others are deleted.
        '''

        # Shapely (Change for matplotlib path later)
        import shapely.geometry as sg

        # Create a list with the earthquakes locations
        LL = np.vstack((self.x, self.y)).T.tolist()

        # Create a MultiPoint object
        PP = sg.MultiPoint(LL)

        # Loop over faults
        u = []
        for fault in faults:
            dis = []
            # Build a line object
            FF = np.vstack((fault.xf, fault.yf)).T.tolist()
            trace = sg.LineString(FF)
            # Get the distance between each point and this line
            for uu in PP.geoms:
                dis.append(trace.distance(uu))
            dis = np.array(dis)
            # Get the indexes of the ones that are close to the fault
            ut = np.flatnonzero( dis < distance )
            # Fill in u
            for i in ut:
                u.append(i)

        # make u an array
        u = np.array(u)
        u = np.unique(u)

        # selection
        self._select(u)

        # All done
        return

    def delete2Close2Trace(self, faults, distance=1.):
        '''
        Deletes the earthquakes that are too close from the fault trace.

        Args:
            * faults    : list of instances of faults. These need to have {xf} and {yf} attributes

        Kwargs:
            * distance  : threshold distance.

        Returns:
            * None. Direclty modifies the list of earthquakes
            
        '''

        # Shapely (Change for matplotlib path later)
        import shapely.geometry as sg

        # Create a list with the earthquakes locations
        LL = np.vstack((self.x, self.y)).T.tolist()

        # Create a MultiPoint object
        PP = sg.MultiPoint(LL)

        # Loop over faults
        u = []
        for fault in faults:
            dis = []
            # Build a line object
            FF = np.vstack((fault.xf, fault.yf)).T.tolist()
            trace = sg.LineString(FF)
            # Get the distance between each point and this line
            for uu in PP.geoms:
                dis.append(trace.distance(uu))
            dis = np.array(dis)
            # Get the indexes of the ones that are away from the fault
            ut = np.flatnonzero( dis < distance )
            # Fill in u
            for i in ut:
                u.append(i)

        # Selection
        self._select(u)

        # All done
        return

    def ProjectOnFaultTrace(self, fault, discretized=True, filename=None):
        '''
        Projects the location of the earthquake along the fault trace. This routine is not a 3D one, it just deals with the surface trace of the fault.

        Args:
            * fault:       Fault object that has a surface trace ({xf} and {yf} attributes)

        Kwargs:
            * discretized: If True, then it uses the discretized fault, not the trace. Never tested with False.
            * filename: Store in a text file
        
        Returns:
            * None. Everything is stored in the dictionary {Projected}
        '''

        # Import needed stuff
        import scipy.spatial.distance as scidis

        # Check that we are in the same utmzone
        assert (fault.utmzone==self.utmzone), 'Fault {} utmzone is not seismiclocation {} utmzone: {} <==> {}'.format(
                    fault.name, self.name, fault.utmzone, self.utmzone)

        # discretized?
        if discretized:
            # Check
            assert (fault.xi is not None), 'Fault {} needs a discretized surface trace, Abort....'.format(fault.name)
            # Get x, y
            x = fault.xi
            y = fault.yi
            nf = fault.xi.shape[0]
        else:
            # Check
            assert (fault.xf is not None), 'Fault {} needs a surface trace, Abort....'.format(fault.name)
            # Get x, y
            x = fault.xf
            y = fault.yf
            nf = fault.xf.shape[0]

        # Compute the cumulative distance along the fault trace
        dis = fault.cumdistance(discretized=discretized)

        # If you want to store that in a file
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | time | depth | mag | AlongStrikeDistance | DistanceToFault \n')

        # Create the structure that'll hold everything
        if not hasattr(self, 'Projected'):
            self.Projected = {}

        # Create the structure that holds this particular one
        self.Projected['{}'.format(fault.name)] = {}
        proj = self.Projected['{}'.format(fault.name)]
        proj['x'] = []
        proj['y'] = []
        proj['lon'] = []
        proj['lat'] = []
        proj['time'] = []
        proj['depth'] = []
        proj['mag'] = []
        proj['AlongStrikeDistance'] = []
        proj['DistanceToFault'] = []

        # Iterate on the earthquakes
        for i in range(self.time.shape[0]):
            # Get earthquake
            qx = self.x[i]
            qy = self.y[i]
            qlon = self.lon[i]
            qlat = self.lat[i]
            qtime = self.time[i]
            qz = self.depth[i]
            qmag = self.mag[i]
            # Get distances
            d = scidis.cdist([[qx, qy]], [ [x[j], y[j]] for j in range(nf)])[0]
            # Get the smallest
            imin1 = d.argmin()
            dmin1 = d.min()
            d[imin1] = 9999999999.
            imin2 = d.argmin()
            dmin2 = d.min()
            dtot= dmin1 + dmin2
            # Do the spatial position
            qx = (x[imin1]*dmin1 + x[imin2]*dmin2)/dtot
            qy = (y[imin1]*dmin1 + y[imin2]*dmin2)/dtot
            # Put into lon lat
            qlon, qlat = self.xy2ll(qx, qy)
            # Compute the AlongStrike Distance
            if dmin1<dmin2:
                jm = imin1
            else:
                jm = imin2
            qdis = dis[jm] + np.sqrt( (qx-x[jm])**2 + (qy-y[jm])**2 )
            # Compute the distance to the fault
            dl = np.sqrt( (x[imin1]-x[imin2])**2 + (y[imin1]-y[imin2])**2 ) # 3 side of the triangle
            semiperi = (dmin1 + dmin2 + dl)/2.                              # Semi-perimeter of the triangle
            # A = semiperi*(semiperi-dmin1)*(semiperi-dmin2)*(semiperi-dl)    # Area of the triangle (Heron's formula)
            A = np.sqrt(semiperi*(semiperi-dmin1)*(semiperi-dmin2)*(semiperi-dl))    # Area of the triangle (Heron's formula)
            qh = 2*A/dl                                                     # Height of the triangle
            # Store all that in a structure
            proj['x'].append(qx)
            proj['y'].append(qy)
            proj['lon'].append(qlon)
            proj['lat'].append(qlat)
            proj['time'].append(qtime)
            proj['depth'].append(qz)
            proj['mag'].append(qmag)
            proj['AlongStrikeDistance'].append(qdis)
            proj['DistanceToFault'].append(qh)
            # Write to file?
            if filename is not None:
                fout.write('{} {} {} {} {} {} {} \n'.format(qlon, qlat, qtime, qz, qmag, qdis, qh))

        # Close the file
        if filename is not None:
            fout.close()

        # All done
        return

    def MapHistogram(self, binwidth=1.0, plot=False, normed=True):
        '''
        Builds a 2D histogram of the earthquakes locations.

        Kwargs:
            * binwidth  : width of the bins used for histogram.
            * plot      : True/False
            * normed    : Normed the histgram

        Returns:
            * None. Histogram is stored in the {histogram} attribute
        '''

        # Build x- and y-bins
        xbins = np.arange(self.x.min(), self.x.max(), binwidth)
        ybins = np.arange(self.y.min(), self.y.max(), binwidth)

        # Build the histogram
        hist, xedges, yedges = np.histogram2d(self.x, self.y, bins=[xbins, ybins], normed=normed)

        # Store the histogram
        self.histogram = {}
        self.histogram['xedges'] = xedges
        self.histogram['yedges'] = yedges
        self.histogram['values'] = hist

        # Store (x,y) locations
        x = (xedges[1:] - xedges[:-1])/2. + xedges[:-1]
        y = (yedges[1:] - yedges[:-1])/2. + yedges[:-1]
        x, y = np.meshgrid(x,y)
        x = x.flatten()
        y = y.flatten()
        self.histogram['x'] = x
        self.histogram['y'] = y

        # Pass it in lon lat
        lon, lat = self.xy2ll(x,y)
        self.histogram['lon'] = lon
        self.histogram['lat'] = lat

        # Plot
        if plot:
            plt.figure()
            plt.imshow(hist, interpolation='nearest', extent=[lon.min(), lon.max(), lat.min(), lat.max()])
            plt.colorbar(orientation='horizontal', shrink=0.6)
            plt.show()

        # All done
        return

    def BuildHistogramsAlongFaultTrace(self, fault, filename, normed=True, width=10.0, bins=50, plot=False, planemode='verticalfault', Range=(-5.0, 5.0), reference=None):
        '''
        Builds a histogram of the earthquake distribution along the fault trace.

        Args:
            * fault : instance of fault with {xf} and {yf} attributes
            * filename: Name of output file

        Kwargs:
            * normed: Norm the histogram
            * width: Width of the averaging cell (distance along strike)
            * bins  : number of bins
            * plot  : True/False
            * planemode: 'verticalfault' or 'bestfit'. verticalfault will assume the fault is vertical. bestfit will fit a dip angle within the cloud of events (can produce weird results sometimes).
            * Range : Range for histogram computation
            * reference: Tuple of float to set the reference of the domain

        Returns:
            * None. Everything is stored in output files
        '''

        # Import needed stuffs
        import scipy.stats as stats

        # Need a projected earthquake set
        assert hasattr(self, 'Projected'), 'No dictionary of Projected earthquakes is available'
        assert ('{}'.format(fault.name)), 'No projection of earthquakes associated with fault {} is available'.format(fault.name)

        # Need a discretized fault trace
        assert hasattr(fault, 'xi'), 'No discretized fault trace is available'

        # Open the file
        frough = open('Rough_'+filename, 'w')
        frough.write('# Along Strike Distance (km) | Distance to the fault plane (km) | Counts \n')
        fsmooth = open(filename, 'w')
        fsmooth.write('# Along Strike Distance (km) | Distance to the fault plane (km) | Counts \n')

        # Get the projected earthquakes
        x = self.x
        y = self.y
        lon = self.lon
        lat = self.lat
        z = self.depth
        Das = self.Projected['{}'.format(fault.name)]['AlongStrikeDistance']
        Dff = self.Projected['{}'.format(fault.name)]['DistanceToFault']

        # Get the fault trace
        xf = fault.xi
        yf = fault.yi

        # And the corresponding distance along the fault
        df = fault.cumdistance(discretized=True)

        # Reference
        if reference is not None:
            xr, yr = self.ll2xy(reference[0], reference[1])
            RefD = np.sqrt( (xr-xf)**2 + (yr-yf)**2 )
            dmin = np.argmin(RefD)
            Ref = df[dmin]
        else:
            Ref = 0.0

        # On every point of the fault
        for i in range(len(xf)):

            # 1. Get the earthquakes in between x-width/2. and x+width/2.
            U = np.flatnonzero( (Das<=df[i]+width/2.) & (Das>=df[i]-width/2.) ).tolist()

            # 2. Get the corresponding earthquakes
            xe = np.array([x[u] for u in U])
            ye = np.array([y[u] for u in U])
            ze = np.array([z[u] for u in U])
            de = np.array([Dff[u] for u in U])

            if planemode in ('bestfit'):

                if xe.shape[0]>3: # Only do the thing if we have more than 3 earthquakes

                    # 3.1 Get those earthquakes in between -2.0 and 2.0 km from the fault trace
                    #     A reasonable thing to do would be to code a L1-regression, but for some places it is a nigthmare
                    M = np.flatnonzero( (de<=2.) & (de>=-2.) ).tolist()
                    xee = np.array([xe[m] for m in M])
                    yee = np.array([ye[m] for m in M])
                    zee = np.array([ze[m] for m in M])

                    # 3.2 Find the center of the best fitting plane
                    c = [xee.sum()/len(xee), yee.sum()/len(yee), zee.sum()/len(zee)]

                    # 3.3 Find the normal of the best fitting plane
                    sumxx = ((xee-c[0])*(xee-c[0])).sum()
                    sumyy = ((yee-c[1])*(yee-c[1])).sum()
                    sumzz = ((zee-c[2])*(zee-c[2])).sum()
                    sumxy = ((xee-c[0])*(yee-c[1])).sum()
                    sumxz = ((xee-c[0])*(zee-c[2])).sum()
                    sumyz = ((yee-c[1])*(zee-c[2])).sum()
                    M = np.array([ [sumxx, sumxy, sumxz],
                                   [sumxy, sumyy, sumyz],
                                   [sumxz, sumyz, sumzz] ])
                    w, v = np.linalg.eig(M)
                    u = w.argmin()
                    N = v[:,u]

                    # 3.4 If the normal points toward the west, rotate it 180 deg
                    if N[0]<0.0:
                        N[0] *= -1.0
                        N[1] *= -1.0
                        N[2] *= -1.0

                    # 3.5 Project earthquakes on that plane and get the distance between the EQ and the plane
                    vecs = np.array([[xe[u]-c[0],ye[u]-c[1], ze[u]-c[2]] for u in range(len(xe))]).T
                    distance = np.dot(N, vecs)
                    xn = xe - distance*N[0]
                    yn = ye - distance*N[1]
                    zn = ze - distance*N[2]

                else:
                    distance = []
                    xn = []
                    yn = []
                    zn = []

            elif planemode in ('verticalfault'):

                # 3.1 Get the fault points
                F = np.flatnonzero( (df<=df[i]+10.) & (df>=df[i]-10.) ).tolist()
                xif = np.array([xf[u] for u in F])
                yif = np.array([yf[u] for u in F])

                # 3.2 Get the center of the fault points
                c = [xif.sum()/len(xif), yif.sum()/len(yif), ze.mean()]

                # 3.3 Get the normal to the fault
                sumxx = ((xif-c[0])*(xif-c[0])).sum()
                sumyy = ((yif-c[1])*(yif-c[1])).sum()
                sumxy = ((xif-c[0])*(yif-c[1])).sum()
                M = np.array([ [sumxx, sumxy],
                               [sumxy, sumyy] ])
                w, v = np.linalg.eig(M)
                u = w.argmin()
                N = np.zeros(3,)
                N[:2] = v[:,u]

                # 3.4 Get the projected earthquakes
                xn = [self.Projected['{}'.format(fault.name)]['x'][u] for u in U]
                yn = [self.Projected['{}'.format(fault.name)]['y'][u] for u in U]
                zn = [self.Projected['{}'.format(fault.name)]['depth'][u] for u in U]

                # 3.5 If the normal points toward the west, rotate it 180 deg
                if N[0]<0.0:
                    N[0] *= -1.0
                    N[1] *= -1.0
                    N[2] *= -1.0

                # 3.6 Get the distance to the fault
                vecs = np.array([[xe[u]-c[0],ye[u]-c[1], ze[u]-c[2]] for u in range(len(xe))]).T
                if vecs.shape[0]>0:
                    distance = np.dot(N, vecs)
                else:
                    distance = []

            # 4. Compute the histogram of the distance
            if len(distance)>10:
                hist, edges = np.histogram(distance, bins=bins, normed=normed, range=Range)
            else:
                hist, edges = np.histogram(distance, bins=bins, normed=False, range=Range)

            # 5. Store the rough histograms in the xy file
            xi = df[i] - Ref
            for k in range(len(hist)):
                yi = (edges[k+1] - edges[k])/2. + edges[k]
                zi = hist[k]
                frough.write('{} {} {} \n'.format(xi, yi, zi))

            # 5. Fit the histogram with a Gaussian and center it
            if len(distance)>10:
                Mu, Sigma = stats.norm.fit(distance)
                Sdis = distance - Mu
                hist, edges = np.histogram(Sdis, bins=bins, normed=normed, range=Range)
                yis = (edges[1:] - edges[:-1])/2. + edges[:-1]
                wis = stats.norm.pdf(yis, Mu, Sigma)
                # Write the smoothed and entered histograms in the file
                for k in range(len(hist)):
                    yi = yis[k]
                    zi = hist[k]
                    wi = wis[k]
                    fsmooth.write('{} {} {} {} \n'.format(xi, yi, zi, wi))
            else:
                hist, edges = np.histogram(distance, bins=bins, normed=False, range=Range)

            # check
            if plot: #  and df[i]>107. and df[i]<108.
                import matplotlib.pyplot as plt
                fig = plt.figure(23)
                ax3 = fig.add_subplot(211, projection='3d')
                ax3.scatter3D(xe, ye, ze, s=5.0, color='k')
                ymin, ymax = ax3.get_ylim(); xmin, xmax = ax3.get_xlim()
                ax3.plot(xf, yf, '-r')
                ax3.scatter3D(xn, yn, zn, s=2.0, color='r')
                ax3.plot([c[0], c[0]+N[0]], [c[1], c[1]+N[1]], [c[2], c[2]+N[2]], '-r')
                ax3.set_xlim([xmin, xmax]); ax3.set_ylim([ymin, ymax])
                axh = fig.add_subplot(212)
                T = axh.hist(distance, bins=bins, normed=True, range=Range)
                edge = T[1]
                hist = T[0]
                cent = (edge[1:] - edge[:-1])/2. + edge[:-1]
                aa = np.arange(cent[0], cent[-1], 0.1)
                g = np.zeros((hist.size,1))
                g[:,0] = stats.norm.pdf(cent, Mu, Sigma)
                A, res, rank, s = np.linalg.lstsq(g, hist)
                axh.plot(aa, A*stats.norm.pdf(aa, Mu, Sigma), '-r')
                plt.show()

        # Close the file
        frough.close()
        fsmooth.close()

        # All done
        return

    def getEarthquakesOnPatches(self, fault, epsilon=0.01):
        '''
        Project each earthquake on a fault patch. Should work with any fault patch type.

        Args:
            * fault : instance of a fault class

        Kwargs:
            * epsilon   : float comparison tolerance number

        Returns:
            * InPatch, a list of events in each patch
        '''

        # Make a list for each patch of the earthquakes that are in the patch
        InPatch = []

        # 1. Compute the side and normal vectors for each patch
        for p in fault.patch:

            # 1.1 Get the side vectors
            v1 = np.array([ p[1][0]-p[0][0], p[1][1]-p[0][1], p[1][2]-p[0][2] ])
            # v2 = np.array([ p[3][0]-p[0][0], p[3][1]-p[0][1], p[3][2]-p[0][2] ]) by kfh, 10/11/2020
            v2 = np.array([ p[2][0]-p[0][0], p[2][1]-p[0][1], p[2][2]-p[0][2] ])

            # 1.2 Dot product is the normal vector
            v3 = np.cross(v1, v2)

            # 1.3 Normalize the normal vector so that it has a length of 1 km
            l = np.linalg.norm(v3)
            v3 /= l

            # Get a few things
            A = p[0]; x = self.x; y = self.y; z = self.depth

            # 2.1 Compute the distance between each earthquake and the plane of the patch
            distances = (A[0]*v3[0] + A[1]*v3[1] + A[2]*v3[2]) / ((v3[0]+x)*v3[0] + (v3[1]+y)*v3[1] + (v3[2] + z)*v3[2])

            # 2.2 Compute the projection of the earthquake on the plane of the patch
            x_proj = distances*(v3[0] + x)
            y_proj = distances*(v3[1] + y)
            z_proj = distances*(v3[2] + z)
            projected = np.array([ [x_proj[i], y_proj[i], z_proj[i]] for i in range(x_proj.shape[0])])

            # 2.3 Check if the point is inside the polygon (works with any kind of fault patch)
            angle = np.zeros(x_proj.shape[0])
            for i in range(len(p)):
                if i<(len(p)-1):
                    j = i+1
                else:
                    j = 0
                a = np.array([p[i][0], p[i][1], p[i][2]])
                b = np.array([p[j][0], p[j][1], p[j][2]])
                v1 = a - projected
                v2 = b - projected
                angle += np.arccos(np.sum(np.multiply(v1, v2), axis=1)/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)))
            iInside = np.flatnonzero(angle>=(2*np.pi-epsilon))
            InPatch.append(iInside)

        # All done
        return InPatch

    def getEarthquakesOnPatcheskfh(self, fault, epsilon=0.01):
        '''
        Project each earthquake on a fault patch. Should work with any fault patch type.

        Args:
            * fault : instance of a fault class

        Kwargs:
            * epsilon   : float comparison tolerance number

        Returns:
            * InPatch, a list of events in each patch
        '''

        # Make a list for each patch of the earthquakes that are in the patch
        InPatch = []

        # 1. Compute the side and normal vectors for each patch
        for p in fault.patch:

            # 1.1 Get the side vectors
            v1 = np.array([ p[1][0]-p[0][0], p[1][1]-p[0][1], -p[1][2]+p[0][2] ])
            # v2 = np.array([ p[3][0]-p[0][0], p[3][1]-p[0][1], p[3][2]-p[0][2] ]) # commented by kfh, 10/11/2020
            v2 = np.array([ p[2][0]-p[0][0], p[2][1]-p[0][1], -p[2][2]+p[0][2] ])

            # 1.2 Dot product is the normal vector
            v3 = np.cross(v1, v2)

            # 1.3 Normalize the normal vector so that it has a length of 1 km
            l = np.linalg.norm(v3)
            v3 /= l

            # Get a few things
            A = p[0]; x = self.x; y = self.y; z = -self.depth

            # 2.1 Compute the distance between each earthquake and the plane of the patch
            # distances = (A[0]*v3[0] + A[1]*v3[1] + A[2]*v3[2]) / ((v3[0]+x)*v3[0] + (v3[1]+y)*v3[1] + (v3[2] + z)*v3[2])

            # 2.2 Compute the projection of the earthquake on the plane of the patch
            # x_proj = distances*(v3[0] + x)
            # y_proj = distances*(v3[1] + y)
            # z_proj = distances*(v3[2] + z)
            # 重新基于u'_vec = u_vec - a_vec*n_vec/(u_vec*n_vec)*u_vec 来改写的投影公式
            # u_proj = u_vec - u'_vec*n_vec 进行投影
            # 前面的投影公式，貌似是进行缩放而不是投影，而这种缩放实际导致的就是投影点偏移
            x1 = x - (A[0]*v3[0] + A[1]*v3[1] - A[2]*v3[2]) / (x*v3[0] + y*v3[1] + z*v3[2])*x
            y1 = y - (A[0]*v3[0] + A[1]*v3[1] - A[2]*v3[2]) / (x*v3[0] + y*v3[1] + z*v3[2])*y
            z1 = z - (A[0]*v3[0] + A[1]*v3[1] - A[2]*v3[2]) / (x*v3[0] + y*v3[1] + z*v3[2])*z
            
            x_proj = x - (x1*v3[0] + y1*v3[1] + z1*v3[2])*v3[0]
            y_proj = y - (x1*v3[0] + y1*v3[1] + z1*v3[2])*v3[1]
            z_proj = z - (x1*v3[0] + y1*v3[1] + z1*v3[2])*v3[2]
            
            projected = np.array([ [x_proj[i], y_proj[i], z_proj[i]] for i in range(x_proj.shape[0])])

            # 2.3 Check if the point is inside the polygon (works with any kind of fault patch)
            angle = np.zeros(x_proj.shape[0])
            for i in range(len(p)):
                if i<(len(p)-1):
                    j = i+1
                else:
                    j = 0
                a = np.array([p[i][0], p[i][1], -p[i][2]])
                b = np.array([p[j][0], p[j][1], -p[j][2]])
                v1 = a - projected
                v2 = b - projected
                angle += np.arccos(np.sum(np.multiply(v1, v2), axis=1)/(np.linalg.norm(v1, axis=1)*np.linalg.norm(v2, axis=1)))
            iInside = np.flatnonzero(angle>=(2*np.pi-epsilon))
            InPatch.append(iInside)

        # All done
        return InPatch

    def getClosestFaultPatch(self, fault):
        '''
        Returns a list of index for all the earthquakes containing the index of the closest fault patch.
    
        Args:
            * fault : an instance of a fault class
    
        Returns:
            * ipatch, a list of the index of the closest patch for each event
        '''
    
        # Create a list
        ipatch = []
    
        # Import KDTree
        from scipy.spatial import KDTree
    
        # Create a list of patch centers
        Centers = [fault.getpatchgeometry(i, center=True)[:3] for i in range(len(fault.patch))]
    
        # Create a KDTree for the patch centers
        tree = KDTree(Centers)
    
        # Create a list of points
        Earthquakes = [[self.x[i], self.y[i], self.depth[i]] for i in range(self.x.shape[0])]
    
        # Query the KDTree for the closest patch for each earthquake
        _, indices = tree.query(Earthquakes)
    
        # Append the indices to the ipatch list
        ipatch.extend(indices)
    
        # All done
        return ipatch

    def mag2Mo(self):
        '''
        Computes the moment from the magnitude. Result in N.m

        Returns:
            * None. Result is in the {Mo} attribute

        '''

        # Compute
        self.Mo = 10.**(1.5*self.mag + 9.1)

        # All done
        return

    def Mo2mag(self):
        '''
        Compute the magnitude from the moment.

        Returns:
            * None. Result is in the {mag} attribute
        '''

        self.mag = 2./3. * (np.log10(self.Mo) - 9.1)

        # All done
        return

    def momentEvolution(self, plot=False, outfile=None):
        '''
        Computes the evolution of the moment with time.

        Kwargs:
            * plot  : True/False
            * outfile: output file

        Returns:
            * None
        '''

        # Make sure these are sorted
        self.sortInTime()

        # Lets consider self.mag is the moment magnitude :-)
        self.mag2Mo()

        # Compute the cumulative moment
        self.cumMo = np.cumsum(self.Mo)

        # Compute the cumulative number of earthquakes
        self.cumEQ = np.cumsum(np.ones(len(self.mag)))

        # Output?
        if outfile is not None:
            fout = open(outfile, 'w')
            fout.write('# Time | Cum. Moment (N.m) | Cum. Num. of Eq. \n')
            # First point
            t = self.time[0].isoformat()
            Mo = 0.1
            Ec = 0
            fout.write('{} {} {} \n'.format(t, Mo, Ec))
            # All the rest
            for i in range(1,len(self.time)):
                t = self.time[i].isoformat()
                Mo = self.cumMo[i-1]
                Ec = self.cumEQ[i-1]
                fout.write('{} {} {} \n'.format(t, Mo, Ec))
                t = self.time[i].isoformat()
                Mo = self.cumMo[i]
                Ec = self.cumEQ[i]
                fout.write('{} {} {} \n'.format(t, Mo, Ec))
            fout.close()

        # Plot?
        if plot:
            fig = plt.figure(1)
            axmo = fig.add_subplot(111)
            axec = axmo.twinx()
            axmo.plot(self.time, self.cumMo, '-', color='black', label='Cum. Moment (N.m)')
            axec.plot(self.time, self.cumEQ, '-', color='gray', label='Cum. # of Eq')
            #axmo.legend()
            #axec.legend()
            plt.show()

        # All done
        return

    def sortInTime(self):
        '''
        Sorts the earthquakes in Time

        Returns:
            * None. Directly modifies the object
        '''

        # Get the ordering
        i = np.argsort(self.time)

        # selection
        self._select(i)

        # All done
        return

    def write2file(self, filename, add_column=None):
        '''
        Write the earthquakes to a file.

        Args:
            * filename      : Name of the output file.

        Kwargs:
            * add_column    : array or list of length equal to the number of events

        Returns:
            * None
        '''
        import pandas as pd
        # open the file
        fout = open(filename, 'w')

        # Write a header
        fout.write('# Lon | Lat | Depth (km) | Mw | time \n')

        # Loop over the earthquakes
        for u in range(len(self.lon)):
            pdutime = pd.to_datetime(self.time[u])
            if add_column is not None:
                last = '{} {}'.format(pdutime.isoformat(), add_column[u])
            else:
                last = '{}'.format(pdutime.isoformat())
            fout.write('{} {} {} {} {} \n'.format(self.lon[u], self.lat[u], self.depth[u], self.mag[u], last))

        # Close the file
        fout.close()

        # all done
        return

    def writeSelectedMagRange(self, filename, minMag=5.0, maxMag=10.):
        '''
        Write to a file the earthquakes with a magnitude larger than minMag and smaller than maxMag.

        Args:
            * filename  : Name of the output file

        Kwargs:
            * minMag    : minimum Magnitude.
            * maxMag    : maximum Magnitude.

        Returns:   
            * None
        '''

        # Create a new object
        eq = copy.deepcopy(self)

        # Remove the small earthquakes
        eq.selectmagnitude(minMag, maximum=maxMag)

        # Write to a file
        eq.write2file(filename)

        # Delete the object
        del eq

        # All done
        return

    def Cmt2Dislocation(self, size=1, mu=30e9, choseplane='nochoice', 
                              moment_from_tensor=False, verbose=True):
        '''
        Builds a list of single square patch faults from the cmt solutions. If no condition is given, it returns the first value.

        Kwargs:
            * size          : Size of one side of the fault patch (km).
            * mu            : Shear modulus (Pa).
            * choseplane    : Choice of the focal plane to use (can be 'smallestdip', 'highestdip', 'nochoice')
            * moment_from_tensor: Computes the scalar moment from the cmt.
            * verbose       : talk to me

        Returns:
            * None. Attribute {faults} is a list of faults with a single patch corresponding to the chosen fault plane
        '''

        if verbose:
            print('---------------------------------')
            print('---------------------------------')
            print('Convert CMTs to dislocation')

        # Import what is needed
        from .planarfault import planarfault

        # Create a list of faults
        self.faults = []

        # Check something
        if not hasattr(self, 'Mo'):
            self.Mo = np.zeros(self.lon.shape)

        # Loop on the earthquakes
        for i in range(len(self.CMTinfo)):

            # Get the event
            eq = self.CMTinfo[i]

            # Get the event name
            event = eq['event name']

            # Get the moment tensor
            cmt = [ [eq['Mrr'], eq['Mrt'], eq['Mrp']],
                    [eq['Mrt'], eq['Mtt'], eq['Mtp']],
                    [eq['Mrp'], eq['Mtp'], eq['Mpp']] ]

            # Get strike dip rake
            sdr1, sdr2, Mo = self._cmt2strikediprake(cmt, returnMo=True)
            self.CMTinfo[i]['cmt'] = cmt

            # Moment
            if moment_from_tensor:
                self.Mo[i] = Mo

            # Condition to chose strike dip rake
            if choseplane in ('smallestdip'):
                if sdr1[1]<sdr2[1]:
                    sdr = sdr1
                else:
                    sdr = sdr2
            elif choseplane in ('highestdip'):
                if sdr1[1]>sdr2[1]:
                    sdr = sdr1
                else:
                    sdr = sdr2
            elif choseplane in ('nochoice'):
                sdr = sdr1
            strike, dip, rake = sdr
            self.CMTinfo[i]['sdr1'] = sdr1
            self.CMTinfo[i]['sdr2'] = sdr2

            # Get the depth
            depth = eq['depth']

            # Shear Modulus (I should code PREM here)
            if (mu.__class__ is float):
                Mu = mu

            # Build a planar fault
            fault = planarfault(event, 
                                utmzone=self.utmzone, 
                                lon0=self.lon0, 
                                lat0=self.lat0, 
                                ellps=self.ellps,
                                verbose=False)
            fault.buildPatches(self.lon[i], self.lat[i], depth, strike*180./np.pi, dip*180./np.pi, size, size, 1, 1, verbose=False)

            # Components of slip
            ss = np.cos(rake) * self.Mo[i] / (Mu * size * size * 1000. * 1000.)
            ds = np.sin(rake) * self.Mo[i] / (Mu * size * size * 1000. * 1000.)

            # Set slip
            fault.slip[0,0] = ss
            fault.slip[0,1] = ds
            fault.slip[0,2] = 0.0

            # Put the fault in the list
            self.faults.append(fault)

            # Save the strike, dip rake infos
            eq['strike'] = strike
            eq['dip'] = dip
            eq['rake'] = rake

        # all done
        return

    def mergeCatalog(self, catalog):
        '''
        Merges another catalog into this one.

        Args:
            * catalog:    Seismic location object.
        
        Returns:
            * None
        '''

        # Merge variables
        self.mag = np.hstack((self.mag, catalog.mag))
        self.lat = np.hstack((self.lat, catalog.lat))
        self.lon = np.hstack((self.lon, catalog.lon))
        self.depth = np.hstack((self.depth, catalog.depth))
        self.time = np.hstack((self.time, catalog.time))

        # Compute the xy
        self.lonlat2xy()

        # all done
        return

    def lonlat2xy(self):
        '''
        Pass the position into the utm coordinate system.

        Returns:
            * None
        '''

        x, y = self.ll2xy(self.lon, self.lat)
        self.x = x
        self.y = y

        # All done
        return

    def xy2lonlat(self):
        '''
        Pass the position from utm to lonlat.

        Returns:
            * None
        '''

        lon, lat = self.xy2ll(self.x, self.y)
        self.lon = lon
        self.lat = lat

        # all done
        return

    def getprofile(self, name, loncenter=None, latcenter=None, length=None, azimuth=None, 
                   width=None, lonend=None, latend=None, ref_to_start=False):
        '''
        Project the seismic locations onto a profile. Works on the lat/lon coordinates system.

        Args:
            * name              : Name of the profile.
            * loncenter         : Profile origin along longitude.
            * latcenter         : Profile origin along latitude.
            * length            : Length of profile.
            * azimuth           : Azimuth in degrees. 0 is north, 90 is east.
            * width             : Width of the profile.
            * lonend            : Profile end along longitude (optional).
            * latend            : Profile end along latitude (optional).
            * ref_to_start      : If True, the distance is computed from the start of the profile.
        
            Way to call the function:
            * getprofile('name', loncenter=, latcenter=, length=, azimuth=, width=),
                where loncenter, latcenter are the coordinates of the center of the profile,
                length is the length of the profile, azimuth is the azimuth of the profile in degrees,
                and width is the width of the profile.
            * getprofile('name', loncenter=, latcenter=, lonend=, latend=, width=),
                where loncenter, latcenter are the coordinates of the starting point of the profile,
                length is the length of the profile, azimuth is the azimuth of the profile in degrees,
                and width is the width of the profile.

        Returns:
            * None. Profiles are stored in the attribute {profiles}
        '''

        # If lonend and latend are provided, calculate length and azimuth
        if lonend is not None and latend is not None:
            xs, ys = self.ll2xy(loncenter, latcenter)
            xe, ye = self.ll2xy(lonend, latend)
            length = np.sqrt((xe - xs)**2 + (ye - ys)**2)
            azimuth = np.degrees(np.arctan2(ye - ys, xe - xs))
            azimuth = (90 - azimuth + 360) % 360
            xc, yc = (xs + xe) / 2., (ys + ye) / 2.
            loncenter, latcenter = self.xy2ll(xc, yc)

        # the profiles are in a dictionary
        if not hasattr(self, 'profiles'):
            self.profiles = {}

        # Convert the lat/lon of the center into UTM.
        xc, yc = self.ll2xy(loncenter, latcenter)

        # Get the profile
        Dalong, Mag, Depth, Time, Dacros, boxll, xe1, ye1, xe2, ye2, lon, lat, Bol = self.coord2prof(
                xc, yc, length, azimuth, width, ref_to_start=ref_to_start)

        # Store it in the profile list
        self.profiles[name] = {}
        dic = self.profiles[name]
        dic['Center'] = [loncenter, latcenter]
        dic['Length'] = length
        dic['Width'] = width
        dic['Box'] = np.array(boxll)
        dic['Lon'] = lon
        dic['Lat'] = lat
        dic['Magnitude'] = Mag
        dic['Depth'] = Depth
        dic['Time'] = Time
        dic['Distance'] = np.array(Dalong)
        dic['Normal Distance'] = np.array(Dacros)
        dic['EndPoints'] = [[xe1, ye1], [xe2, ye2]]
        lone1, late1 = self.xy2ll(xe1, ye1)
        lone2, late2 = self.xy2ll(xe2, ye2)
        dic['EndPointsLL'] = [[lone1, late1],
                            [lone2, late2]]
        dic['Indices'] = Bol                              

        # All done
        return
    
    def add_index_range_to_profile(self, name, distance_range=None, normal_distance_range=None):
        '''
        Add index range to a profile.

        Args:
            * name                  : Name of the profile.
            * distance_range        : Range of Distance, a tuple like (min, max).
            * normal_distance_range : Range of Normal Distance, a tuple like (min, max).

        Returns:
            * None. The index ranges are added to the profile in the attribute {profiles}
        '''

        # Check if the profile exists
        if name not in self.profiles:
            raise ValueError(f"No profile named {name}")

        # Get the profile
        profile = self.profiles[name]

        # Get the indices for Distance and Normal Distance within the given ranges
        if distance_range is None:
            distance_indices = np.arange(len(profile['Distance']))
        else:
            distance_indices = np.where((profile['Distance'] >= distance_range[0]) & 
                                        (profile['Distance'] <= distance_range[1]))[0]

        if normal_distance_range is None:
            normal_distance_indices = np.arange(len(profile['Normal Distance']))
        else:
            normal_distance_indices = np.where((profile['Normal Distance'] >= normal_distance_range[0]) & 
                                               (profile['Normal Distance'] <= normal_distance_range[1]))[0]

        # Add the indices to the profile
        profile['Distance Indices'] = distance_indices
        profile['Normal Distance Indices'] = normal_distance_indices

        # All done
        return

    def get_available_cluster_methods(self):
        """Return the names of all registered cluster methods."""
        return self.cluster_method_factory.get_registered_methods()
    
    def get_available_linear_detection_methods(self):
        return self.linear_detection_method_factory.get_registered_methods()
    
    def _extract_linear_clusters(self, X_scaled, name, linear_detection_methods={}):
        '''
        Extract the clusters that are more linear based on the provided linear detection methods.
    
        Args:
            * X_scaled    : The preprocessed data.
            * linear_detection_methods: A dictionary where the keys are the names of methods for linear detection,
                                        and the values are dictionaries of properties to set on the methods.
    
        Returns:
            * None. The cluster labels are updated in the attribute {cluster_labels}
        '''
        labels = self.cluster_labels[name]
        
        for i in np.unique(labels):
            if i != -1:  # -1 label is for noise in DBSCAN, HDBSCAN, etc.
                cluster_points = X_scaled[labels == i]
                for method_name, properties in linear_detection_methods.items():
                    method = self.linear_detection_method_factory.get_method(method_name)
                    for prop_name, prop_value in properties.items():
                        setattr(method, prop_name, prop_value)
                    if not method.detect(cluster_points):
                        self.cluster_labels[name][labels == i] = -1
                        break

    def cluster_earthquakes(self, name=None, on_profile=False, method='dbscan', 
                            linear_detection_methods={}, **kwargs):
        if not self.cluster_method_factory.get_method(method):
            raise ValueError(f"Unknown method {method}")
        
        if on_profile:
            profile = self._get_profile(name)
            data = np.column_stack((profile['Distance'], profile['Depth']))
        else:
            data = np.column_stack((self.x, self.y))  # assuming self.x and self.y contains all earthquakes data
        
        preprocess_kwargs = {k: kwargs.pop(k) for k in ['depth_weight', 'distance_weight', 'remove_slope'] if k in kwargs}
        X_scaled = self._preprocess_data(data, **preprocess_kwargs)
        self.cluster_labels[name] = self.cluster_method_factory.get_method(method).cluster(X_scaled, **kwargs)

        if linear_detection_methods:
             self._extract_linear_clusters(X_scaled, name, linear_detection_methods)

    def cluster_earthquakes_on_profile(self, name, method='dbscan', 
                                       linear_detection_methods={}, **kwargs):
        self.cluster_earthquakes(name=name, on_profile=True, method=method, 
                                 linear_detection_methods=linear_detection_methods, **kwargs)
        # Add cluster labels to the profile
        self.profiles[name]['Cluster Labels'] = self.get_cluster_labels(name)

    def _get_profile(self, name):
        if name not in self.profiles:
            raise ValueError(f"No profile named {name}")
        return self.profiles[name]
    
    def get_cluster_labels(self, name):
        return self.cluster_labels.get(name, None)

    def _preprocess_data(self, data, depth_weight=1, distance_weight=1, remove_slope=False):
        '''
        Preprocess the data for clustering. This includes stacking the distance and depth data, 
        optionally removing the slope in the depth dimension, standardizing the data, and applying weights.

        Args:
            * profile           : The profile data, a dictionary with 'Distance' and 'Depth' as keys.
            * depth_weight      : The weight of the depth dimension. Default is 1.
            * distance_weight   : The weight of the distance dimension. Default is 1.
            * remove_slope      : Whether to remove the slope in the depth dimension. Default is False.

        Returns:
            * X_scaled          : The preprocessed data, a 2D array with the same number of rows as the input data and 2 columns for the standardized and weighted distance and depth.
        '''
        X = data
        if remove_slope:
            slope, _, _, _, _ = linregress(X[:, 0], X[:, 1])
            X[:, 1] -= X[:, 0] * slope
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_scaled[:, 0] *= depth_weight
        X_scaled[:, 1] *= distance_weight
        return X_scaled

    def coord2prof(self, xc, yc, length, azimuth, width, ref_to_start=False):
        '''
        Routine returning the profile

        Args:
            * xc                : X pos of center
            * yc                : Y pos of center
            * length            : length of the profile.
            * azimuth           : azimuth of the profile.
            * width             : width of the profile.

        Kwargs:
            * ref_to_start      : If True, the distance is computed from the start of the profile.

        Returns:
            * dis                 : Distance from the center
            * mag                 : Magnitude
            * depth               : Depth
            * norm                : distance perpendicular to profile
            * boxll               : lon lat coordinates of the profile box used
            * xe1, ye1            : coordinates (UTM) of the profile endpoint
            * xe2, ye2            : coordinates (UTM) of the profile endpoint
        '''

        # Azimuth into radians
        alpha = azimuth*np.pi/180.

        # Copmute the across points of the profile
        xa1 = xc - (width/2.)*np.cos(alpha)
        ya1 = yc + (width/2.)*np.sin(alpha)
        xa2 = xc + (width/2.)*np.cos(alpha)
        ya2 = yc - (width/2.)*np.sin(alpha)

        # Compute the endpoints of the profile
        xe1 = xc + (length/2.)*np.sin(alpha)
        ye1 = yc + (length/2.)*np.cos(alpha)
        xe2 = xc - (length/2.)*np.sin(alpha)
        ye2 = yc - (length/2.)*np.cos(alpha)

        # Convert the endpoints
        elon1, elat1 = self.xy2ll(xe1, ye1)
        elon2, elat2 = self.xy2ll(xe2, ye2)

        # Design a box in the UTM coordinate system.
        x1 = xe1 - (width/2.)*np.cos(alpha)
        y1 = ye1 + (width/2.)*np.sin(alpha)
        x2 = xe1 + (width/2.)*np.cos(alpha)
        y2 = ye1 - (width/2.)*np.sin(alpha)
        x3 = xe2 + (width/2.)*np.cos(alpha)
        y3 = ye2 - (width/2.)*np.sin(alpha)
        x4 = xe2 - (width/2.)*np.cos(alpha)
        y4 = ye2 + (width/2.)*np.sin(alpha)

        # Convert the box into lon/lat for further things
        lon1, lat1 = self.xy2ll(x1, y1)
        lon2, lat2 = self.xy2ll(x2, y2)
        lon3, lat3 = self.xy2ll(x3, y3)
        lon4, lat4 = self.xy2ll(x4, y4)

        # make the box
        box = []
        box.append([x1, y1])
        box.append([x2, y2])
        box.append([x3, y3])
        box.append([x4, y4])

        # make latlon box
        boxll = []
        boxll.append([lon1, lat1])
        boxll.append([lon2, lat2])
        boxll.append([lon3, lat3])
        boxll.append([lon4, lat4])

        # Get the InSAR points in this box.
        # 1. import shapely and nxutils
        import matplotlib.path as path
        import shapely.geometry as geom

        # 2. Create an array with the positions
        XY = np.vstack((self.x, self.y)).T

        # 3. Create a box
        rect = path.Path(box, closed=False)

        # 4. Find those who are inside
        Bol = rect.contains_points(XY)

        # 4. Get these values
        xg = self.x[Bol]
        yg = self.y[Bol]
        lon = self.lon[Bol]
        lat = self.lat[Bol]
        mag = self.mag[Bol]
        time = self.time[Bol]
        depth = self.depth[Bol]

        # Check if lengths are ok
        if len(xg) > 5:

            # 5. Get the sign of the scalar product between the line and the point
            vec = np.array([xe1-xc, ye1-yc])
            xy = np.vstack((xg-xc, yg-yc)).T
            sign = np.sign(np.dot(xy, vec))

            # 6. Compute the distance (along, across profile) and get the velocity
            # Create the list that will hold these values
            Dacros = []; Dalong = [];
            # Build lines of the profile
            Lalong = geom.LineString([[xe1, ye1], [xe2, ye2]])
            Lacros = geom.LineString([[xa1, ya1], [xa2, ya2]])
            # Build a multipoint
            PP = geom.MultiPoint(np.vstack((xg,yg)).T.tolist())
            # Loop on the points
            for p in range(len(PP.geoms)):
                Dalong.append(Lacros.distance(PP.geoms[p])*sign[p])
                Dacros.append(Lalong.distance(PP.geoms[p]))

        else:
            Dalong = mag
            Dacros = mag
        
        if ref_to_start:
            # Get the distance to the start
            Dalong = np.array(Dalong)
            PP = geom.MultiPoint(np.array([[xe2, ye2]]).tolist())
            offset = Lacros.distance(PP.geoms[0])*-1
            Dalong = Dalong - offset

        Dalong = np.array(Dalong)
        Dacros = np.array(Dacros)

        # Toss out nans
        jj = np.flatnonzero(np.isfinite(mag)).tolist()
        mag = mag[jj]
        lon = lon[jj]
        lat = lat[jj]
        Dalong = Dalong[jj]
        Dacros = Dacros[jj]
        depth = depth[jj]
        time = time[jj]

        # All done
        return Dalong, mag, depth, time, Dacros, boxll, xe1, ye1, xe2, ye2, lon, lat, Bol

    def writeProfile2File(self, name, filename, fault=None):
        '''
        Writes the profile named 'name' to the ascii file filename.

        Args:
            * name      : name of the profile you want to write
            * filename  : Name of the output file

        Kwargs:
            * fault     : Add fault for intersection

        Returns:
            * None
        '''

        # open a file
        fout = open(filename, 'w')

        # Get the dictionary
        dic = self.profiles[name]

        # Write the header
        fout.write('#---------------------------------------------------\n')
        fout.write('# Profile Generated with CSI\n')
        fout.write('# Center: {} {} \n'.format(dic['Center'][0], dic['Center'][1]))
        fout.write('# Endpoints: \n')
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][0][0], dic['EndPointsLL'][0][1]))
        fout.write('#           {} {} \n'.format(dic['EndPointsLL'][1][0], dic['EndPointsLL'][1][1]))
        fout.write('# Box Points: \n')
        fout.write('#           {} {} \n'.format(dic['Box'][0][0],dic['Box'][0][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][1][0],dic['Box'][1][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][2][0],dic['Box'][2][1]))
        fout.write('#           {} {} \n'.format(dic['Box'][3][0],dic['Box'][3][1]))

        # Place faults in the header
        if fault is not None:
            if fault.__class__ is not list:
                fault = [fault]
            fout.write('# Fault Positions: \n')
            for f in fault:
                d = self.intersectProfileFault(name, f)
                fout.write('# {}           {} \n'.format(f.name, d))

        fout.write('#---------------------------------------------------\n')

        # Write the values
        for i in range(len(dic['Distance'])):
            d = dic['Distance'][i]
            z = dic['Depth'][i]
            Mp = dic['Magnitude'][i]
            Lon = dic['Lon'][i]
            Lat = dic['Lat'][i]
            if np.isfinite(Mp):
                fout.write('{} {} {} {} {} \n'.format(d, z, Mp, Lon, Lat))

        # Close the file
        fout.close()

        # all done
        return

    def intersectProfileFault(self, name, fault):
        '''
        Gets the distance between the fault/profile intersection and the profile center.

        Args:
            * name      : name of the profile.
            * fault     : fault object from verticalfault.

        Returns:
            * None
        '''

        # Import shapely
        import shapely.geometry as geom

        # Grab the fault trace
        xf = fault.xf
        yf = fault.yf

        # Grab the profile
        prof = self.profiles[name]

        # import shapely
        import shapely.geometry as geom

        # Build a linestring with the profile center
        Lp = geom.LineString(prof['EndPoints'])

        # Build a linestring with the fault
        ff = []
        for i in range(len(xf)):
            ff.append([xf[i], yf[i]])
        Lf = geom.LineString(ff)

        # Get the intersection
        if Lp.crosses(Lf):
            Pi = Lp.intersection(Lf)
            if type(Pi) is geom.point.Point:
                p = Pi.coords[0]
            else:
                return None
        else:
            return None

        # Get the center
        lonc, latc = prof['Center']
        xc, yc = self.ll2xy(lonc, latc)

        # Get the sign
        xa,ya = prof['EndPoints'][0]
        vec1 = [xa-xc, ya-yc]
        vec2 = [p[0]-xc, p[1]-yc]
        sign = np.sign(np.dot(vec1, vec2))

        # Compute the distance to the center
        d = np.sqrt( (xc-p[0])**2 + (yc-p[1])**2)*sign

        # All done
        return d

#PRIVATE METHODS
    def _getDistance2Fault(self, fault):
        '''
        Computes the distance between the fault trace and all the earthquakes.
        '''

        # Import
        import shapely.geometry as sg

        # Create a list with earthquakes
        LL = np.vstack((self.x, self.y)).T.tolist()
        PP = sg.MultiPoint(LL)

        # Build a line object
        FF = np.vstack((fault.xf, fault.yf)).T.tolist()
        trace = sg.LineString(FF)

        # Distance
        dis = []
        for p in PP.geoms:
            dis.append(trace.distance(p))

        # All done
        return dis

    def _getDistance2FaultPlane(self, fault):
        '''
        Computes the distance between the fault plane and all the earthquakes.
        '''

        # import scipy
        import scipy.spatial.distance as scidis

        # Create a list
        dis = []

        # Create the list of vertices
        if fault.patchType in ('triangle', 'triangletent'):
            vertices = fault.Vertices
        elif fault.patchType in ('rectangle'):
            vertices = []
            for p in fault.patch:
                for i in range(4):
                    vertices.append(p[i])

        # Loop on the earthquakes
        for i in range(self.mag.shape[0]):

            # Get position
            x = self.x[i]
            y = self.y[i]
            z = self.depth[i]

            # Get the min distance
            d = scidis.cdist([[x, y, z]], vertices).min()

            # Append
            dis.append(d)

        # All done
        return dis

    def _delete(self, u):
        '''
        delete earthquakes
        '''

        # Select the stations
        self.lon = np.delete(self.lon, u)
        self.lat = np.delete(self.lat, u)
        self.x = np.delete(self.x,u)
        self.y = np.delete(self.y,u)
        self.time = np.delete(self.time,u)
        self.depth = np.delete(self.depth,u)
        self.mag = np.delete(self.mag,u)

        # Conditional
        if hasattr(self, 'CMTinfo'):
            self.CMTinfo = np.array(self.CMTinfo)
            self.CMTinfo = np.delete(self.CMTinfo,u)
            self.CMTinfo = self.CMTinfo.tolist()

        # Conditional
        if hasattr(self, 'err'): self.err = np.delete(self.err, u)

        # Conditional
        if hasattr(self, 'Mo'): self.Mo = np.delete(self.Mo,u)

        # Conditional
        if hasattr(self, 'rms'): self.rms = np.delete(self.rms,u)

        # All done
        return

    def _select(self, u):
        '''
        Makes a selection.
        '''

        # Select the stations
        self.lon = self.lon[u]
        self.lat = self.lat[u]
        self.x = self.x[u]
        self.y = self.y[u]
        self.time = self.time[u]
        self.depth = self.depth[u]
        self.mag = self.mag[u]

        # Conditional
        if hasattr(self, 'CMTinfo'):
            self.CMTinfo = np.array(self.CMTinfo)
            self.CMTinfo = self.CMTinfo[u]
            self.CMTinfo = self.CMTinfo.tolist()

        if hasattr(self, 'err'): self.err = self.err[u]

        # Conditional
        if hasattr(self, 'Mo'): self.Mo = self.Mo[u]

        # Conditional
        if hasattr(self, 'rms'): self.rms = self.rms[u]

        # All done
        return

    def _cmt2strikediprake(self, cmt, returnMo=False):
        '''
        From a moment tensor in Harvard convention, returns 2 tuples of (strike, dip, rake)
        Args:
            * cmt   : Array (3,3) with the CMT.
        '''

        # 1. Compute the eigenvalues and eigenvectors
        EigValues, EigVectors = np.linalg.eig(cmt)

        # 2. Sort them => T = max(Eig)
        #                 N = Neutral
        #                 P = min(Eig)
        #    Then, n = (T+P)/sqrt(2)    # Normal
        #          s = (P-T)/sqrt(2)    # Slip
        T = EigVectors[:,np.argmax(EigValues)]
        P = EigVectors[:,np.argmin(EigValues)]
        n = (T+P)/np.sqrt(2.)
        s = (T-P)/np.sqrt(2.)

        # 3. Compute the moment
        Mo = (np.abs(np.min(EigValues)) + np.abs(np.max(EigValues)))
        Mo /= 2e7

        # 4. Get strike, dip and rake from vectors
        sdr1 = self._ns2sdr(n,s)
        sdr2 = self._ns2sdr(s,n)

        # All done
        if returnMo:
            return sdr1, sdr2, Mo
        else:
            return sdr1, sdr2

    def _ns2sdr(self, n, s, epsilon=0.0001):
        '''
        From the normal and the slip vector, returns the strike, dip and rake.
        Args:
            * n     : Normal vector.
            * s     : Slip vector.
        '''

        # Case: If normal downwards, flip them
        if n[0]<0.:
            n = -1.0*n
            s = -1.0*s

        # Case: if normal is vertical (i.e. if the plane is horizontal)
        if n[0]>(1-epsilon):
            strike = 0.0
            dip = 0.0
            rake = np.arctan2(-s[2], -s[1])

        # Case: if normal is horizontal (i.e. plane if vertical)
        elif n[0]<epsilon:
            strike = np.arctan2(n[1], n[2])
            dip = np.pi/2.
            rake = np.arctan2(s[0], -s[1]*n[2] + s[2]*n[1])

        # General Case:
        else:
            strike = np.arctan2(n[1], n[2])
            dip = np.arccos(n[0])
            rake = np.arctan2(-s[1]*n[1]-s[2]*n[2], (-s[1]*n[2]+s[2]*n[1])*n[0])

        # Strike
        if strike < 0.:
            strike += 2*np.pi

        # All done
        return strike, dip, rake

    def plot_profiles(self, fault, plot_aftershocks=False, figsize=None, 
                      style=['notebook'], fontsize=None, save_fig=False, 
                      file_path='profile.png', dpi=300, scatter_props=None,
                      show=True, draw_trace_arrow=True):
        from eqtools.plottools import DegreeFormatter
        from eqtools.plottools import sci_plot_style
        with sci_plot_style(style=style, fontsize=fontsize, figsize=figsize):
        # Set default properties for plotting
        # with plt.style.context(style):
        #     plt.rcParams['font.family'] = 'sans-serif'
        #     plt.rcParams['axes.formatter.use_mathtext'] = False
        #     plt.rcParams['text.usetex'] = False
        #     plt.rcParams['mathtext.fontset'] = 'dejavusans'
        #     plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica','DejaVu Sans', 'Bitstream Vera Sans', 
        #                                     'Computer Modern Sans Serif', 'Lucida Grande', 'Verdana', 'Geneva', 
        #                                     'Lucid', 'Avant Garde', 'sans-serif']

        #     if fontsize is not None:
        #         plt.rcParams['axes.labelsize'] = fontsize
        #         plt.rcParams['xtick.labelsize'] = fontsize
        #         plt.rcParams['ytick.labelsize'] = fontsize
        #         plt.rcParams['legend.fontsize'] = fontsize
        #         plt.rcParams['font.size'] = fontsize
            
        #     if figsize is not None:
        #         plt.rcParams['figure.figsize'] = figsize

            # Convert the top_strike from degrees to radians
            top_strike_rad = np.radians(90.0 - fault.top_strike[0])

            # Compute the top_strike vector
            top_strike_vector = [np.cos(top_strike_rad), np.sin(top_strike_rad)]

            # Get the start and end points of the fault trace
            start_point = [fault.lon[0], fault.lat[0]]
            second_point = [fault.lon[1], fault.lat[1]]
            penultimate_point = [fault.lon[-2], fault.lat[-2]]
            end_point = [fault.lon[-1], fault.lat[-1]]

            # Compute the vectors for the start and end segments of the fault trace
            start_vector = [second_point[0] - start_point[0], second_point[1] - start_point[1]]
            # Compute the dot product of the start_vector and the top_strike_vector
            dot_product = start_vector[0] * top_strike_vector[0] + start_vector[1] * top_strike_vector[1]

            # Plot fault trace
            plt.plot(fault.lon, fault.lat, 'r-')

            if draw_trace_arrow:
                if dot_product < 0:
                    # If the start segment is more aligned with the top_strike, draw the arrow at the start
                    plt.annotate('', xy=start_point, xytext=second_point, 
                                 arrowprops=dict(arrowstyle='->', linewidth=1, color='r', shrinkA=0, shrinkB=0))
                else:
                    # Otherwise, draw the arrow at the end
                    plt.annotate('', xy=end_point, xytext=penultimate_point, 
                                 arrowprops=dict(arrowstyle='->', linewidth=1, color='r', shrinkA=0, shrinkB=0))

            # Plot profiles
            for iprofile in self.profiles:
                profile = self.profiles[iprofile]
                box = profile['Box']
                end_points_ll = profile['EndPointsLL']

                # 画Box用虚线，添加第一个点到最后以闭合
                box_closed = np.append(box, [box[0]], axis=0)
                plt.plot(box_closed[:, 0], box_closed[:, 1], 'k--')

                # 画EndPointsLL用黑色实线
                # plt.plot([point[0] for point in end_points_ll], [point[1] for point in end_points_ll], 'k-')

                # Get the start and end points
                start_point = end_points_ll[1]
                end_point = end_points_ll[0]

                # Compute the vector from start to end
                dx = end_point[0] - start_point[0]
                dy = end_point[1] - start_point[1]

                # Draw the vector with an arrow
                # plt.arrow(start_point[0], start_point[1], dx, dy, 
                #           color='k', length_includes_head=True, head_width=0.05, head_length=0.1)
                # Create an arrow with two short oblique lines
                plt.annotate('', xy=end_point, xytext=start_point, 
                             arrowprops=dict(arrowstyle='->', linewidth=1, color='k', shrinkA=0, shrinkB=0))

            # 如果plot_aftershocks为True，画余震
            if plot_aftershocks:
                if scatter_props is None:
                    scatter_props = {'color': '#1a6faf'}
                plt.scatter(self.lon, self.lat, **scatter_props)

            # 设置x轴和y轴的比例相等
            plt.gca().set_aspect('equal', adjustable='box')
            # 设置x轴和y轴的标签
            formatter = DegreeFormatter()
            ax = plt.gca()
            ax.xaxis.set_major_formatter(formatter)
            ax.yaxis.set_major_formatter(formatter)

            # 如果save_fig为True，保存图形
            if save_fig:
                plt.savefig(file_path, dpi=dpi, bbox_inches='tight')

            # 显示图形
            if show:
                plt.show()
            else:
                plt.close()


    def plot_profiles_with_map(self, names, x_key='Distance', y_key='Depth', figsize=(10, 8), 
                               scatter_props=None, save_fig=False, file_path='profiles.png', 
                               dpi=300, show=True, invert_yaxis=True, map_use_degrees=True):
        '''
        Plot profiles based on the given names, including a map view and a profile view.
    
        Args:
            * names          : Name or list of names of the profiles to plot.
            * x_key          : Key for x-axis values in the profile dictionary. Default is 'Distance'.
            * y_key          : Key for y-axis values in the profile dictionary. Default is 'Depth'.
            * figsize        : Size of the figure. Default is (10, 8).
            * scatter_props  : Properties for the scatter plot. Default is None.
            * save_fig       : Whether to save the figure. Default is False.
            * file_path      : Path to save the figure. Default is 'profiles.png'.
            * dpi            : DPI for saving the figure. Default is 300.
            * show           : Whether to show the figure. Default is True.
            * invert_yaxis   : Whether to invert the y-axis. Default is True.
            * map_use_degrees: Whether to use degrees for the axes. Default is True
    
        Returns:
            * None
        '''
    
        import matplotlib.pyplot as plt
    
        # Ensure names is a list
        if isinstance(names, str):
            names = [names]
    
        # Set default scatter properties if not provided
        if scatter_props is None:
            scatter_props = {'color': '#1a6faf', 's': 10, 'alpha': 0.6}
    
        # Create a figure
        fig, axes = plt.subplots(len(names), 2, figsize=figsize, gridspec_kw={'width_ratios': [1, 3]})
    
        if len(names) == 1:
            axes = [axes]
    
        for i, name in enumerate(names):
            profile = self.profiles[name]
    
            # Plot the profile box and line on the map
            ax_map = axes[i][0]
            box = profile['Box']
            end_points_ll = profile['EndPointsLL']
    
            # Plot all earthquakes
            ax_map.scatter(self.lon, self.lat, **scatter_props)
    
            # Plot the profile box
            box_closed = np.append(box, [box[0]], axis=0)
            ax_map.plot(box_closed[:, 0], box_closed[:, 1], 'k--')
    
            # Plot the profile line
            ax_map.plot([point[0] for point in end_points_ll], [point[1] for point in end_points_ll], 'k-')
    
            # Plot the profile arrow
            start_point = end_points_ll[1]
            end_point = end_points_ll[0]
            ax_map.annotate('', xy=end_point, xytext=start_point, arrowprops=dict(arrowstyle='->', linewidth=1, color='k', shrinkA=0, shrinkB=0))
    
            # Set aspect ratio and labels
            ax_map.set_aspect('equal', adjustable='box')
            ax_map.set_xlabel('Longitude')
            ax_map.set_ylabel('Latitude')
    
            # Plot the profile detail view
            ax_profile = axes[i][1]
            x_values = profile[x_key]
            y_values = profile[y_key]
            ax_profile.scatter(x_values, y_values, **scatter_props)
    
            # Set labels
            ax_profile.set_xlabel(x_key)
            ax_profile.set_ylabel(y_key)

            # Invert y-axis if requested
            if invert_yaxis:
                ax_profile.invert_yaxis()

        # Set aspect ratio for the first column
        for ax in axes[:, 0]:
            ax.set_aspect('equal', adjustable='box')
            if map_use_degrees:
                from eqtools.plottools import set_degree_formatter
                set_degree_formatter(ax)

        # Adjust layout
        plt.tight_layout()
    
        # Save the figure if requested
        if save_fig:
            plt.savefig(file_path, dpi=dpi, bbox_inches='tight')
    
        # Show the figure if requested
        if show:
            plt.show()
        else:
            plt.close()
    
        # All done
        return

    def plot_profiles_map(self, names, scatter_props=None, save_fig=False, file_path='profiles_map.png', dpi=300, 
                          show=True, figsize=(7.5, 6.5), map_equal=True, use_degrees=True,
                          xlim=None, ylim=None, show_profile_names=True):
        '''
        Plot all profiles on a single map.
    
        Args:
            * names          : Name or list of names of the profiles to plot.
            * scatter_props  : Properties for the scatter plot. Default is None.
            * save_fig       : Whether to save the figure. Default is False.
            * file_path      : Path to save the figure. Default is 'profiles_map.png'.
            * dpi            : DPI for saving the figure. Default is 300.
            * show           : Whether to show the figure. Default is True.
            * figsize        : Size of the figure. Default is (10, 8).
            * map_equal      : Whether to set the aspect ratio of the map to be equal. Default is True.
            * use_degrees    : Whether to use degrees for the axes. Default is True
            * xlim           : Limits for the x-axis. Default is None.
            * ylim           : Limits for the y-axis. Default is None.
            * show_profile_names: Whether to show profile names at the start point. Default is True.
    
        Returns:
            * None
        '''
    
        import matplotlib.pyplot as plt
    
        # Ensure names is a list
        if isinstance(names, str):
            names = [names]
    
        # Set default scatter properties if not provided
        if scatter_props is None:
            scatter_props = {'color': '#1a6faf', 's': 10, 'alpha': 0.6}
    
        # Create a figure for the map
        fig_map, ax_map = plt.subplots(figsize=figsize)
    
        # Plot all earthquakes on the map
        ax_map.scatter(self.lon, self.lat, **scatter_props)
    
        for i, name in enumerate(names):
            profile = self.profiles[name]
    
            # Plot the profile box and line on the map
            box = profile['Box']
            end_points_ll = profile['EndPointsLL']
    
            # Plot the profile box
            box_closed = np.append(box, [box[0]], axis=0)
            ax_map.plot(box_closed[:, 0], box_closed[:, 1], 'k--')
    
            # Plot the profile line
            ax_map.plot([point[0] for point in end_points_ll], [point[1] for point in end_points_ll], 'k-')
    
            # Plot the profile arrow
            start_point = end_points_ll[1]
            end_point = end_points_ll[0]
            ax_map.annotate('', xy=end_point, xytext=start_point, arrowprops=dict(arrowstyle='->', linewidth=1, color='k', shrinkA=0, shrinkB=0))
    
            # Add profile name at the start point if requested
            if show_profile_names:
                # label = f"({chr(97 + i)}) {name}"
                label = f'{name}'
                ax_map.text(start_point[0], start_point[1], label, fontsize=10, va='bottom', ha='right')
    
        # Set aspect ratio and labels for the map
        if map_equal:
            ax_map.set_aspect('equal', adjustable='box')
        ax_map.set_xlabel('Longitude')
        ax_map.set_ylabel('Latitude')
    
        if xlim is not None:
            ax_map.set_xlim(xlim)
        if ylim is not None:
            ax_map.set_ylim(ylim)
    
        if use_degrees:
            from eqtools.plottools import set_degree_formatter
            set_degree_formatter(ax_map)
    
        # Save the figure if requested
        if save_fig:
            fig_map.savefig(file_path, dpi=dpi, bbox_inches='tight')
    
        # Show the figure if requested
        if show:
            plt.show()
        else:
            plt.close(fig_map)
    
        # All done
        return
    
    def plot_profiles_detail(self, names, x_key='Distance', y_key='Depth', figsize=(10, 8), scatter_props=None, 
                             save_fig=False, file_path='profiles_detail.png', dpi=300, show=True, invert_yaxis=True, 
                             profile_equal=False, color_key='Depth', cmap='cmc.hawaii_r', subplots_per_row=3, 
                             add_label_prefix=True, label_position=(0.05, 0.95)):
        '''
        Plot detailed profiles based on the given names.
    
        Args:
            * names          : Name or list of names of the profiles to plot.
            * x_key          : Key for x-axis values in the profile dictionary. Default is 'Distance'.
            * y_key          : Key for y-axis values in the profile dictionary. Default is 'Depth'.
            * figsize        : Size of the figure. Default is (10, 8).
            * scatter_props  : Properties for the scatter plot. Default is None.
            * save_fig       : Whether to save the figure. Default is False.
            * file_path      : Path to save the figure. Default is 'profiles_detail.png'.
            * dpi            : DPI for saving the figure. Default is 300.
            * show           : Whether to show the figure. Default is True.
            * invert_yaxis   : Whether to invert the y-axis. Default is True.
            * profile_equal  : Whether to set the aspect ratio of the profile to be equal. Default is False.
            * color_key      : Key for color values in the profile dictionary. Default is 'Depth'.
            * cmap           : Colormap for the color values. Default is 'cmc.hawaii_r'.
            * subplots_per_row: Number of subplots per row. Default is 3.
            * add_label_prefix: Whether to add a letter prefix to the profile labels. Default is True.
            * label_position : Position of the label in the subplot. Default is (0.05, 0.95).
    
        Returns:
            * None
        '''
    
        import matplotlib.pyplot as plt
        import cmcrameri as cmc
    
        # Ensure names is a list
        if isinstance(names, str):
            names = [names]
    
        # Set default scatter properties if not provided
        if scatter_props is None:
            scatter_props = {'s': 10, 'alpha': 0.6}
    
        # Calculate the number of rows needed
        num_profiles = len(names)
        num_rows = (num_profiles + subplots_per_row - 1) // subplots_per_row
    
        # Create a figure for the profiles
        fig_profiles, axes_profiles = plt.subplots(num_rows, subplots_per_row, figsize=figsize)
        axes_profiles = axes_profiles.flatten()
    
        for i, name in enumerate(names):
            profile = self.profiles[name]
    
            # Plot the profile detail view
            ax_profile = axes_profiles[i]
            x_values = profile[x_key]
            y_values = profile[y_key]
            if color_key is not None:
                color_values = profile[color_key]
                if 'c' in scatter_props:
                    del scatter_props['c']
                scatter = ax_profile.scatter(x_values, y_values, c=color_values, cmap=cmap, **scatter_props)
            else:
                scatter = ax_profile.scatter(x_values, y_values, **scatter_props)
    
            # Set labels
            ax_profile.set_xlabel(x_key)
            ax_profile.set_ylabel(y_key)
    
            # Invert y-axis if requested
            if invert_yaxis:
                ax_profile.invert_yaxis()
    
            # Set aspect ratio for the profile if requested
            if profile_equal:
                ax_profile.set_aspect('equal', adjustable='box')
    
            # Add color bar
            cbar = plt.colorbar(scatter, ax=ax_profile)
            cbar.set_label(color_key)
    
            # Add profile label with optional prefix
            label = f"({chr(97 + i)}) {name}" if add_label_prefix else name
            ax_profile.text(label_position[0], label_position[1], label, transform=ax_profile.transAxes, fontsize=12, verticalalignment='top')
    
        # Remove unused subplots
        for j in range(i + 1, len(axes_profiles)):
            fig_profiles.delaxes(axes_profiles[j])
    
        # Adjust layout for profiles
        plt.tight_layout()
    
        # Save the figure if requested
        if save_fig:
            fig_profiles.savefig(file_path, dpi=dpi, bbox_inches='tight')
    
        # Show the figure if requested
        if show:
            plt.show()
        else:
            plt.close(fig_profiles)
    
        # All done
        return
    
    def plot_epicenter_combo(
        self, xlim=None, ylim=None,
        show_kde=True, show_hist=True, bins=50, kde_levels=5,
        style=['science', 'no-latex'], figsize=None, plot_style=None,
        title="Epicenter Distribution (with Density)", save_path=None
    ):
        '''
        Plots the epicenter distribution with optional KDE and histogram.

        Args:
            * xlim          : Limits for the x-axis.
            * ylim          : Limits for the y-axis.
            * show_kde      : Whether to show the KDE plot. Default is True.
            * show_hist     : Whether to show the histogram. Default is True.
            * bins          : Number of bins for the histogram. Default is 50.
            * kde_levels    : Number of levels for the KDE contour plot. Default is 5.
            * style         : Style for the plot. Default is ['science', 'no-latex'].
            * figsize       : Size of the figure. Default is None.
            * plot_style    : Additional style parameters for the sci_plot_style plot. Default is None.
            * title         : Title of the plot. Default is "Epicenter Distribution (with Density)".
            * save_path     : Path to save the figure. Default is None.

        Returns:
            * None
        '''

        from seismo_tools import plot_epicenter_combo
        # Call the plot_epicenter_combo function from seismo_tools
        plot_epicenter_combo(
            self.lon, self.lat, xlim=xlim, ylim=ylim,
            show_kde=show_kde, show_hist=show_hist, bins=bins, kde_levels=kde_levels,
            style=style, figsize=figsize, plot_style=plot_style,
            title=title, save_path=save_path
        )
    
    def plot_mag_and_cumulative(self, style=['science', 'no-latex'], figsize='double', plot_style=None,
                                title="Time-Magnitude Stick Plot & Cumulative Number", save_path=None):
        """
        Plots the time-magnitude stick plot and cumulative number of earthquakes.
        Args:
            * style         : Style for the plot. Default is ['science', 'no-latex'].
            * figsize       : Size of the figure. Default is 'double'.
            * plot_style    : Additional style parameters for the sci_plot_style plot. Default is None.
            * title         : Title of the plot. Default is "Time-Magnitude Stick Plot & Cumulative Number".
            * save_path     : Path to save the figure. Default is None.
        Returns:
            * None
        """
        from seismo_tools import plot_mag_and_cumulative
        # Call the plot_mag_and_cumulative function from seismo_tools
        plot_mag_and_cumulative(
            self.time, self.mag, style=style, figsize=figsize,
            plot_style=plot_style, title=title, save_path=save_path
        )
    
    def plot_gutenberg_richter(
        self, bin_width=0.1, fit_min_mag=None, fit_method="ls", delta_mag=0.0,
        style=['science'], figsize=None, plot_style=None,
        title="Gutenberg-Richter (logN-M) Relation", save_path=None
    ):
        """
        Plots the Gutenberg-Richter relation for the earthquake catalog.
        
        Args:
            * bin_width     : Width of the bins for the histogram. Default is 0.1.
            * fit_min_mag   : Minimum magnitude for fitting the Gutenberg-Richter relation. Default is None.
            * fit_method    : Method for fitting the relation. Default is "ls" (least squares).
            * delta_mag     : Offset to apply to the magnitudes. Default is 0.0.
            * style         : Style for the plot. Default is ['science'].
            * figsize       : Size of the figure. Default is None.
            * plot_style    : Additional style parameters for the sci_plot_style plot. Default is None.
            * title         : Title of the plot. Default is "Gutenberg-Richter (logN-M) Relation".
            * save_path     : Path to save the figure. Default is None.

        Returns:
            * None
        """
        from seismo_tools import plot_gutenberg_richter
        # Call the plot_gutenberg_richter function from seismo_tools
        plot_gutenberg_richter(
            self.mag, bin_width=bin_width, fit_min_mag=fit_min_mag,
            fit_method=fit_method, delta_mag=delta_mag,
            style=style, figsize=figsize, plot_style=plot_style,
            title=title, save_path=save_path
        )
# EOF
