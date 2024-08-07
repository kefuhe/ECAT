"""
Convert between local enu, local llh, and global xyz coordinates
Translated from the Matlab toolkit of the Stanford lab.
"""

import numpy as np

# Datum list --------------------------------------------------------------------
data = {
    'ABINDAN              ': [-1.121450e+002, -5.475071e-005,  -162,   -12,   206],
    'AFGOOYE              ': [-1.080000e+002,  4.807955e-007,   -43,  -163,    45],
    'AIN EL ABD 1970      ': [-2.510000e+002, -1.419270e-005,  -150,  -251,    -2],
    'ANNA 1 ASTRO 1965    ': [-2.300000e+001, -8.120449e-008,  -491,   -22,   435],
    'ARC 1950             ': [-1.121450e+002, -5.475071e-005,  -143,   -90,  -294],
    'ARC 1960             ': [-1.121450e+002, -5.475071e-005,  -160,    -8,  -300],
    'ASCENSION ISLAND 1958': [-2.510000e+002, -1.419270e-005,  -207,   107,    52],
    'ASTRO B4 SOROL ATOLL ': [-2.510000e+002, -1.419270e-005,   114,  -116,  -333],
    'ASTRO BEACON "E"     ': [-2.510000e+002, -1.419270e-005,   145,    75,  -272],
    'ASTRO DOS 71/4       ': [-2.510000e+002, -1.419270e-005,  -320,   550,  -494],
    'ASTRONOMIC STN 1952  ': [-2.510000e+002, -1.419270e-005,   124,  -234,   -25],
    'AUSTRALIAN GEOD 1966 ': [-2.300000e+001, -8.120449e-008,  -133,   -48,   148],
    'AUSTRALIAN GEOD 1984 ': [-2.300000e+001, -8.120449e-008,  -134,   -48,   149],
    'BD 72                ': [-2.510000e+002, -1.419270e-005,  -126,    80,  -101],
    'BELLEVUE (IGN)       ': [-2.510000e+002, -1.419270e-005,  -127,  -769,   472],
    'BERMUDA 1957         ': [-6.940000e+001, -3.726464e-005,   -73,   213,   296],
    'BOGOTA OBSRVATRY     ': [-2.510000e+002, -1.419270e-005,   307,   304,  -318],
    'CAMPO INCHAUSPE      ': [-2.510000e+002, -1.419270e-005,  -148,   136,    90],
    'CANTON ASTRO 1966    ': [-2.510000e+002, -1.419270e-005,   298,  -304,  -375],
    'CAPE                 ': [-1.121450e+002, -5.475071e-005,  -136,  -108,  -292],
    'CAPE CANAVERAL       ': [-6.940000e+001, -3.726464e-005,    -2,   150,   181],
    'CARTHAGE             ': [-1.121450e+002, -5.475071e-005,  -263,     6,   431],
    'CH-1903              ': [07.398450e+002,  1.003748e-005,   674,    15,   405],
    'CHATHAM 1971         ': [-2.510000e+002, -1.419270e-005,   175,   -38,   113],
    'CHUA ASTRO           ': [-2.510000e+002, -1.419270e-005,  -134,   229,   -29],
    'CORREGO ALEGRE       ': [-2.510000e+002, -1.419270e-005,  -206,   172,    -6],
    'DJAKARTA (BATAVIA)   ': [07.398450e+002,  1.003748e-005,  -377,   681,   -50],
    'DOS 1968             ': [-2.510000e+002, -1.419270e-005,   230,  -199,  -752],
    'EASTER ISLAND 1967   ': [-2.510000e+002, -1.419270e-005,   211,   147,   111],
    'EUROPEAN 1950        ': [-2.510000e+002, -1.419270e-005,   -87,   -98,  -121],
    'EUROPEAN 1979        ': [-2.510000e+002, -1.419270e-005,   -86,   -98,  -119],
    'FINLAND HAYFORD      ': [-2.510000e+002, -1.419270e-005,   -78,  -231,   -97],
    'GANDAJIKA BASE       ': [-2.510000e+002, -1.419270e-005,  -133,  -321,    50],
    'GEODETIC DATUM 1949  ': [-2.510000e+002, -1.419270e-005,    84,   -22,   209],
    'GUAM 1963            ': [-6.940000e+001, -3.726464e-005,  -100,  -248,   259],
    'GUX 1 ASTRO          ': [-2.510000e+002, -1.419270e-005,   252,  -209,  -751],
    'HJORSEY 1955         ': [-2.510000e+002, -1.419270e-005,   -73,    46,   -86],
    'HONG KONG 1963       ': [-2.510000e+002, -1.419270e-005,  -156,  -271,  -189],
    'HU-TZU-SHAN          ': [-2.510000e+002, -1.419270e-005,  -637,  -549,  -203],
    'INDIAN BANGLADESH    ': [08.606550e+002,  2.836137e-005,   289,   734,   257],
    'INDIAN THAILAND      ': [08.606550e+002,  2.836137e-005,   214,   836,   303],
    'IRELAND 1965         ': [07.968110e+002,  1.196002e-005,   506,  -122,   611],
    'ISRAEL               ': [-1.637890e+002, -5.473908e-005,  -235,   -85,   264],
    'ISTS 073 ASTRO 1969  ': [-2.510000e+002, -1.419270e-005,   208,  -435,  -229],
    'JOHNSTON ISLAND      ': [-2.510000e+002, -1.419270e-005,   191,   -77,  -204],
    'KANDAWALA            ': [08.606550e+002,  2.836137e-005,   -97,   787,    86],
    'KERGUELEN ISLAND     ': [-2.510000e+002, -1.419270e-005,   145,  -187,   103],
    'KERTAU 1948          ': [08.329370e+002,  2.836137e-005,   -11,   851,     5],
    'L.C. 5 ASTRO         ': [-6.940000e+001, -3.726464e-005,    42,   124,   147],
    'LIBERIA 1964         ': [-1.121450e+002, -5.475071e-005,   -90,    40,    88],
    'LUZON MINDANAO       ': [-6.940000e+001, -3.726464e-005,  -133,   -79,   -72],
    'LUZON PHILIPPINES    ': [-6.940000e+001, -3.726464e-005,  -133,   -77,   -51],
    'MAHE 1971            ': [-1.121450e+002, -5.475071e-005,    41,  -220,  -134],
    'MARCO ASTRO          ': [-2.510000e+002, -1.419270e-005,  -289,  -124,    60],
    'MASSAWA              ': [07.398450e+002,  1.003748e-005,   639,   405,    60],
    'MERCHICH             ': [-1.121450e+002, -5.475071e-005,    31,   146,    47],
    'MICHELIN             ': [01.614000e+003,  1.127918e-004,  1118,    23,    66],
    'MIDWAY ASTRO 1961    ': [-2.510000e+002, -1.419270e-005,   912,   -58,  1227],
    'MINNA                ': [-1.121450e+002, -5.475071e-005,   -92,   -93,   122],
    'NAD27 ALASKA         ': [-6.940000e+001, -3.726464e-005,    -5,   135,   172],
    'NAD27 BAHAMAS        ': [-6.940000e+001, -3.726464e-005,    -4,   154,   178],
    'NAD27 CANADA         ': [-6.940000e+001, -3.726464e-005,   -10,   158,   187],
    'NAD27 CANAL ZONE     ': [-6.940000e+001, -3.726464e-005,     0,   125,   201],
    'NAD27 CARIBBEAN      ': [-6.940000e+001, -3.726464e-005,    -7,   152,   178],
    'NAD27 CENTRAL        ': [-6.940000e+001, -3.726464e-005,     0,   125,   194],
    'NAD27 CONUS          ': [-6.940000e+001, -3.726464e-005,    -8,   160,   176],
    'NAD27 CUBA           ': [-6.940000e+001, -3.726464e-005,    -9,   152,   178],
    'NAD27 GREENLAND      ': [-6.940000e+001, -3.726464e-005,    11,   114,   195],
    'NAD27 MEXICO         ': [-6.940000e+001, -3.726464e-005,   -12,   130,   190],
    'NAD27 SAN SALVADOR   ': [-6.940000e+001, -3.726464e-005,     1,   140,   165],
    'NAD83                ': [00.000000e+000, -1.643484e-011,     0,     0,     0],
    'NAHRWN MASIRAH ILND  ': [-1.121450e+002, -5.475071e-005,  -247,  -148,   369],
    'NAHRWN SAUDI ARABIA  ': [-1.121450e+002, -5.575071e-005,  -231,  -196,   482],
    'NAHRWN UNITED ARAB   ': [-1.121450e+002, -5.475071e-005,  -249,  -156,   381],
    'NAPARIMA BWI         ': [-2.510000e+002, -1.419270e-005,    -2,   374,   172],
    'NETHERLANDS          ': [07.400000e+002,  1.003748e-005,   593,    26,   478],
    'OBSERVATORIO 1966    ': [-2.510000e+002, -1.419270e-005,  -425,  -169,    81],
    'OLD EGYPTIAN         ': [-6.300000e+001,  4.807955e-007,  -130,   110,   -13],
    'OLD HAWAIIAN         ': [-6.940000e+001, -3.726464e-005,    61,  -285,  -181],
    'OMAN                 ': [-1.121450e+002, -5.475071e-005,  -346,    -1,   224],
    'ORD SRVY GRT BRITN   ': [05.736040e+002,  1.196002e-005,   375,  -111,   431],
    'PICO DE LAS NIEVES   ': [-2.510000e+002, -1.419270e-005,  -307,   -92,   127],
    'PITCAIRN ASTRO 1967  ': [-2.510000e+002, -1.419270e-005,   185,   165,    42],
    'POTSDAM              ': [07.398000e+002,  1.003748e-005,   587,    16,   393],
    'PROV SO AMRICN 1956  ': [-2.510000e+002, -1.419270e-005,  -288,   175,  -376],
    'PROV SO CHILEAN 1963 ': [-2.510000e+002, -1.419270e-005,    16,   196,    93],
    'PUERTO RICO          ': [-6.940000e+001, -3.726464e-005,    11,    72,  -101],
    'QATAR NATIONAL       ': [-2.510000e+002, -1.419270e-005,  -128,  -283,    22],
    'QORNOQ               ': [-2.510000e+002, -1.419270e-005,   164,   138,  -189],
    'REUNION              ': [-2.510000e+002, -1.419270e-005,    94,  -948, -1262],
    'ROME 1940            ': [-2.510000e+002, -1.419270e-005,  -225,   -65,     9],
    'RT 90                ': [07.398450e+002,  1.003748e-005,   498,   -36,   568],
    'S-42                 ': [-1.080000e+002,  4.807600e-007,    23,  -124,   -84],
    'SANTO (DOS)          ': [-2.510000e+002, -1.419270e-005,   170,    42,    84],
    'SAO BRAZ             ': [-2.510000e+002, -1.419270e-005,  -203,   141,    53],
    'SAPPER HILL 1943     ': [-2.510000e+002, -1.419270e-005,  -355,    16,    74],
    'SCHWARZECK           ': [06.531350e+002,  1.003748e-005,   616,    97,  -251],
    'SOUTH AMERICAN 1969  ': [-2.300000e+001, -8.120449e-008,   -57,     1,   -41],
    'SOUTH ASIA           ': [-1.800000e+001,  4.807955e-007,     7,   -10,   -26],
    'SOUTHEAST BASE       ': [-2.510000e+002, -1.419270e-005,  -499,  -249,   314],
    'SOUTHWEST BASE       ': [-2.510000e+002, -1.419270e-005,  -104,   167,   -38],
    'TIMBALAI 1948        ': [08.606550e+002,  2.836137e-005,  -689,   691,   -46],
    'TOKYO                ': [07.398450e+002,  1.003748e-005,  -128,   481,   664],
    'TRISTAN ASTRO 1968   ': [-2.510000e+002, -1.419270e-005,  -632,   438,  -609],
    'VITI LEVU 1916       ': [-1.121450e+002, -5.475071e-005,    51,   391,   -36],
    'WAKE-ENIWETOK 1960   ': [-1.330000e+002, -1.419270e-005,   101,    52,   -39],
    'WGS 72               ': [02.000000e+000,  3.121058e-008,     0,     0,     5],
    'WGS 84               ': [00.000000e+000,  0.000000e+000,     0,     0,     0],
    'ZANDERIJ             ': [-2.510000e+002, -1.419270e-005,  -265,   120,  -358]
}


def get_datums(names=None):
    """
    Returns da, df, dX, dY, dZ given a specific datum.
    DATUMVALUE=data(DATUMNAMES) returns the datum parameters for datum specified by string array DATUMNAMES.
    These parameters are defined as differences to the WGS-84 ellipsoid:
    * da = WGS-84 equatorial radius minus the specified datum equatorial radius (meters)
    * df = WGS-84 flattening minus the specified datum flattening
    * dX = X-coordinate of WGS-84 geocenter minus the specified datum X-coordinate (meters)
    * dY = Y-coordinate of WGS-84 geocenter minus the specified datum Y-coordinate (meters)
    * dZ = Z-coordinate of WGS-84 geocenter minus the specified datum Z-coordinate (meters)

    For reference:
    * WGS-84 Equatorial Radius (a) = 6378137.0
    * WGS-84 Flattening (f) = 1/298.257223563

    Calling the function without input arguments returns a list of available datums.
    Unmatched datums return NaNs.

    :param names: string
    :type names: string
    :returns: 5 floats representing the chosen datum relative to WGS-84
    :rtype: array
    """

    if not names:    # Return list of available datums if called with no input arguments.
        return data.keys()
    # Read the database. Match requested datums with those available.
    all_keys = data.keys()            # collect keys
    value = np.zeros((1, 5))   # initialize return vaule
    modified_name = "{:<21}".format(names.upper())
    if modified_name in all_keys:
        value[0, :] = data[modified_name]
    else:
        value[0, :] = [np.nan, np.nan, np.nan, np.nan, np.nan]
    return value
