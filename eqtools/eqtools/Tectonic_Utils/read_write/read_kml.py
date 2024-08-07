

def read_simple_kml(infile):
    """
    Read a simple box drawn in Google Earth and saved as a KML file with field 'coordinates'.
    For more complicated KMLs with many elements, I'm switching to use the Python library "fastkml" for reading.

    :param infile: kml file with simple box
    :type infile: string
    :returns: lons, lats as lists that represent the coordinates of box vertices
    :rtype: list, list
    """
    print("Reading %s into arrays of lon and lat..." % infile)
    start = 0
    lats, lons = [], []
    ifile = open(infile, 'r')
    for line in ifile:
        if start == 1:
            temp = line.split()
            for item in temp:
                lons.append(float(item.split(',')[0]))
                lats.append(float(item.split(',')[1]))
            break
        if "coordinates" in line:
            start = 1
    return lons, lats
