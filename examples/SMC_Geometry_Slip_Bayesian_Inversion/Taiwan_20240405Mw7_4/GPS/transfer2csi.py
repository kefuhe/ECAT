from csi.gps import gps
import pandas as pd
import numpy as np

filename = r'us7000m9g4_forweb_NGL.txt'

# Read the data
lon0 = 122
lat0 = 24

mygps = gps('mygps', lon0=lon0, lat0=lat0)
mygps.read_from_enu(filename, header=2)
mygps.plot(drawCoastlines=False)