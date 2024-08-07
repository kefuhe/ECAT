"""
Functions to read common file types into structures in Python.
Example: a multi-segment file with polygons or lines, as could be used in GMT.
"""


def read_gmt_multisegment_latlon(input_file, split_delimiter=' '):
    """
    Generalized GMT multisegment file reader.
    Returns lon and lat in a list of lists, each element with a single segment.

    :param input_file: name of input file
    :type input_file: string
    :param split_delimiter: delimiter between values on the same line, defaults to space
    :type split_delimiter: string, optional
    :returns: list of lons, list of lats
    :rtype: list
    """
    print("reading gmt multisegment file %s" % input_file)
    ifile = open(input_file)
    lon_collection, lat_collection = [], []
    lon_temp, lat_temp = [], []
    for line in ifile:
        if line.split()[0] == '>>' or line.split()[0] == '>':
            if lon_temp:
                lon_collection.append(lon_temp)
                lat_collection.append(lat_temp)
            lon_temp, lat_temp = [], []
            continue
        else:
            temp = line.split(split_delimiter)
            lon_temp.append(float(temp[0]))
            lat_temp.append(float(temp[1]))
    lon_collection.append(lon_temp)
    lat_collection.append(lat_temp)
    return lon_collection, lat_collection
