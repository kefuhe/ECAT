'''
A parent class that deals with triangular patches fault

Written by Bryan Riel, Z. Duputel and R. Jolivet November 2013

Modified by Kefeng He in 2022-2024 to speed up the code and add more functionalities.
'''

# Externals
import numpy as np
import pyproj as pp
import matplotlib.pyplot as plt
import scipy.interpolate as sciint
from . import triangularDisp as tdisp
from scipy.linalg import block_diag
import copy
import sys
import os
from numba import jit

# Personals
from .Fault import Fault
from .geodeticplot import geodeticplot as geoplot
from .gps import gps as gpsclass


@jit(nopython=True)
def calculate_triangle_areas(vertices, faces):
    v1 = vertices[faces[:, 0]]
    v2 = vertices[faces[:, 1]]
    v3 = vertices[faces[:, 2]]
    a = np.sqrt(np.sum((v2 - v1)**2, axis=1))
    b = np.sqrt(np.sum((v3 - v2)**2, axis=1))
    c = np.sqrt(np.sum((v1 - v3)**2, axis=1))
    s = (a + b + c) / 2
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    return areas


class TriangularPatches(Fault):
    '''
    Classes implementing a fault made of triangular patches. Inherits from Fault

    Args:
        * name      : Name of the fault.

    Kwargs:
        * utmzone   : UTM zone  (optional, default=None)
        * lon0      : Longitude of the center of the UTM zone
        * lat0      : Latitude of the center of the UTM zone
        * ellps     : ellipsoid (optional, default='WGS84')
        * verbose   : Speak to me (default=True)
    '''

    # ----------------------------------------------------------------------
    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True, lon0=None, lat0=None):

        # Base class init
        super(TriangularPatches,self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose)

        # Specify the type of patch
        self.patchType = 'triangle'

        # The case of vertical faults with triangular patches is tricky, so we leave this option here
        self.vertical = False

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setdepth(self):
        '''
        Set depth patch attributes

        Returns:
            * None
        '''

        # Set depth
        self.depth = np.max([v[2] for v in self.Vertices])
        self.z_patches = np.linspace(self.depth, 0.0, 5)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def computeArea(self):
        '''
        Computes the area of all triangles.

        Returns:
            * None
        '''

        # Area
        self.area = []

        # Loop over patches
        for patch in self.patch:
            self.area.append(self.patchArea(patch))

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def patchArea(self, patch):
        '''
        Returns the area of one patch.

        Args:   
            * patch : one item of the patch list.

        Returns:
            * Area  : float
        '''

        # Get vertices of patch
        p1, p2, p3 = list(patch)

        # Compute side lengths
        a = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2 + (p2[2] - p1[2])**2)
        b = np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2 + (p3[2] - p2[2])**2)
        c = np.sqrt((p1[0] - p3[0])**2 + (p1[1] - p3[1])**2 + (p1[2] - p3[2])**2)

        # Compute area using numerically stable Heron's formula
        c,b,a = np.sort([a, b, c])
        area = 0.25 * np.sqrt((a + (b + c)) * (c - (a - b))
                       * (c + (a - b)) * (a + (b - c)))

        # All Done
        return area
    # ----------------------------------------------------------------------

    def compute_triangle_areas(self):
        self.area = calculate_triangle_areas(self.Vertices, self.Faces)
        return self.area
    
    def compute_patch_areas(self):
        self.area = calculate_triangle_areas(self.Vertices, self.Faces)
        return self.area

    # ----------------------------------------------------------------------
    def splitPatch(self, patch):
        '''
        Splits a patch into 4 patches, based on the mid-point of each side.

        Args:
            * patch : item of the patch list.

        Returns:
            * t1, t2, t3, t4    : 4 patches
        '''

        # Get corners
        p1, p2, p3 = list(patch)
        if type(p1) is not list:
            p1 = p1.tolist()
        if type(p2) is not list:
            p2 = p2.tolist()
        if type(p3) is not list:
            p3 = p3.tolist()

        # Compute mid-points
        p12 = [p1[0] + (p2[0]-p1[0])/2., 
               p1[1] + (p2[1]-p1[1])/2.,
               p1[2] + (p2[2]-p1[2])/2.]
        p23 = [p2[0] + (p3[0]-p2[0])/2.,
               p2[1] + (p3[1]-p2[1])/2.,
               p2[2] + (p3[2]-p2[2])/2.]
        p31 = [p3[0] + (p1[0]-p3[0])/2.,
               p3[1] + (p1[1]-p3[1])/2.,
               p3[2] + (p1[2]-p3[2])/2.]

        # make 4 triangles
        t1 = [p1, p12, p31]
        t2 = [p12, p2, p23]
        t3 = [p31, p23, p3]
        t4 = [p31, p12, p23]

        # All done
        return t1, t2, t3, t4
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def selectPatches(self,minlon,maxlon,minlat,maxlat,mindep,maxdep):
        '''
        Removes patches that are outside of a 3D box.

        Args:   
            * minlon        : west longitude
            * maxlon        : east longitude
            * minlat        : south latitude
            * maxlat        : north latitude
            * mindep        : Minimum depth
            * maxdep        : Maximum depth

        Returns:
            * None
        '''

        xmin,ymin = self.ll2xy(minlon,minlat)
        xmax,ymax = self.ll2xy(maxlon,maxlat)

        for p in range(len(self.patch)-1,-1,-1):
            x1, x2, x3, width, length, strike, dip = self.getpatchgeometry(p)
            if x1<xmin or x1>xmax or x2<ymin or x2>ymax or x3<mindep or x3>maxdep:
                self.deletepatch(p)

        for i in range(len(self.xf)-1,-1,-1):
            x1 = self.xf[i]
            x2 = self.yf[i]
            if x1<xmin or x1>xmax or x2<ymin or x2>ymax:
                self.xf = np.delete(self.xf,i)
                self.yf = np.delete(self.yf,i)

        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def vertices2ll(self):
        '''
        Converts all the vertices into lonlat coordinates.

        Returns:    
            * None
        '''

        # Create a list
        vertices_ll = []

        # iterate
        for vertex in self.Vertices:
            lon, lat = self.xy2ll(vertex[0], vertex[1])
            vertices_ll.append([lon, lat, vertex[2]])

        # Save
        self.Vertices_ll = np.array(vertices_ll)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def setVerticesFromPatches(self):
        '''
        Takes the patches and constructs a list of Vertices and Faces
    
        Returns:
            * None
        '''
    
        # Get patches
        patches = np.array(self.patch)
    
        # Flatten the patches array
        flat_patches = patches.reshape(-1, patches.shape[-1])
    
        # Use cKDTree to find unique vertices
        from scipy.spatial import cKDTree
        tree = cKDTree(flat_patches)
        _, unique_indices = np.unique(tree.query(flat_patches)[1], return_index=True)
        vertices = flat_patches[unique_indices]
    
        # Create a dictionary for vertex indices
        vertex_index = {tuple(vertex): idx for idx, vertex in enumerate(vertices)}
    
        # Iterate to build Faces
        faces = []
        for patch in patches:
            face = [vertex_index[tuple(vertex)] for vertex in patch]
            faces.append(face)
    
        # Convert faces list to NumPy array
        faces = np.array(faces)
    
        # Set them
        self.Vertices = vertices
        self.Faces = faces
    
        # 2 lon lat
        self.vertices2ll()
    
        # All done
        return
    # ----------------------------------------------------------------------
    
    # ----------------------------------------------------------------------
    def patches2triangles(self, fault, numberOfTriangles=4):
        '''
        Takes a fault with rectangular patches and splits them into triangles to 
        initialize self.

        Args:
            * fault             : instance of rectangular patches.

        Kwargs:
            * numberOfTriangles : Split each patch in 2 or 4 (default) triangle

        Returns:
            * None
        '''

        # Initialize the lists of patches
        self.patch = []
        self.patchll = []

        # Initialize vertices and faces
        vertices = []
        faces = []

        # Each patch is being splitted in 2 or 4 triangles
        for patch in fault.patch:

            # Add vertices
            for vertex in patch.tolist():
                if vertex not in vertices:
                    vertices.append(vertex)

            # Find the vertices in the list
            i0 = np.flatnonzero(np.array([patch[0].tolist()==v for v in vertices]))[0]
            i1 = np.flatnonzero(np.array([patch[1].tolist()==v for v in vertices]))[0]
            i2 = np.flatnonzero(np.array([patch[2].tolist()==v for v in vertices]))[0]
            i3 = np.flatnonzero(np.array([patch[3].tolist()==v for v in vertices]))[0]

            if numberOfTriangles==4:
                
                # Get the center
                center = np.array(fault.getpatchgeometry(patch, center=True)[:3])
                vertices.append(list(center))
                ic = np.flatnonzero(np.array([center.tolist()==v for v in vertices]))[0]

                # Split in 4
                t1 = np.array([patch[0], patch[1], center])
                t2 = np.array([patch[1], patch[2], center])
                t3 = np.array([patch[2], patch[3], center])
                t4 = np.array([patch[3], patch[0], center])

                # faces
                fs = ([i0, i1, ic],
                      [i1, i2, ic],
                      [i2, i3, ic],
                      [i3, i0, ic])
        
                # patches
                ps = [t1, t2, t3, t4]

            elif numberOfTriangles==2:
        
                # Split in 2
                t1 = np.array([patch[0], patch[1], patch[2]])
                t2 = np.array([patch[2], patch[3], patch[0]])

                # faces
                fs = ([i0, i1, i2], [i2, i3, i0])

                # patches
                ps = (t1, t2)
            
            else:
                assert False, 'numberOfTriangles should be 2 or 4'

            for f,p in zip(fs, ps):
                faces.append(f)
                self.patch.append(p)

        # Save
        self.Vertices = np.array(vertices)
        self.Faces = np.array(faces)

        # Convert
        self.vertices2ll()
        self.patch2ll()

        # Initialize slip
        self.initializeslip()

        # Set fault trace
        self.xf = fault.xf
        self.yf = fault.yf
        self.lon = fault.lon
        self.lat = fault.lat
        if hasattr(fault, 'xi'):
            self.xi = fault.xi
            self.yi = fault.yi
            self.loni = fault.loni
            self.lati = fault.lati

        # Set depth
        self.setdepth()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def readPatchesFromFile(self, filename, readpatchindex=True, 
                            donotreadslip=False, gmtslip=True,
                            inputCoordinates='lonlat'):
        '''
        Reads patches from a GMT formatted file.

        Args:
            * filename          : Name of the file

        Kwargs:
            * inputCoordinates  : Default is 'lonlat'. Can be 'utm'
            * readpatchindex    : Default True.
            * donotreadslip     : Default is False. If True, does not read the slip
            * gmtslip           : A -Zxxx is in the header of each patch
            * inputCoordinates  : Default is 'lonlat', can be 'xyz'

        Returns:
            * None
        '''

        # create the lists
        self.patch = []
        self.patchll = []
        if readpatchindex:
            self.index_parameter = []
        if not donotreadslip:
            Slip = []

        # open the files
        fin = open(filename, 'r') 
        
        # read all the lines
        A = fin.readlines()

        # depth
        D = 0.0
        d = 10000.

        # Index
        if gmtslip:
            ipatch = 3
        else:
            ipatch = 2

        # Loop over the file
        i = 0
        while i<len(A):
            
            # Assert it works
            assert A[i].split()[0] == '>', 'Not a patch, reformat your file...'
            # Get the Patch Id
            ids = 0
            if readpatchindex:
                self.index_parameter.append([np.int64(A[i].split()[ipatch]),np.int64(A[i].split()[ipatch+1]),np.int64(A[i].split()[ipatch+2])])
                ids += 3
            # Get the slip value
            if not donotreadslip:
                step = ids if ids == 0 else ids + 1
                if len(A[i].split())>ipatch + step:
                    slip = np.array([np.float_(A[i].split()[ipatch+step]), np.float_(A[i].split()[ipatch+step+1]), np.float_(A[i].split()[ipatch+step+2])])
                else:
                    slip = np.array([0.0, 0.0, 0.0])
                Slip.append(slip)
            # get the values
            if inputCoordinates in ('lonlat'):
                lon1, lat1, z1 = A[i+1].split()
                lon2, lat2, z2 = A[i+2].split()
                lon3, lat3, z3 = A[i+3].split()
                # Pass as floating point
                lon1 = float(lon1); lat1 = float(lat1); z1 = float(z1)
                lon2 = float(lon2); lat2 = float(lat2); z2 = float(z2)
                lon3 = float(lon3); lat3 = float(lat3); z3 = float(z3)
                # translate to utm
                x1, y1 = self.ll2xy(lon1, lat1)
                x2, y2 = self.ll2xy(lon2, lat2)
                x3, y3 = self.ll2xy(lon3, lat3)
            elif inputCoordinates in ('xyz'):
                x1, y1, z1 = A[i+1].split()
                x2, y2, z2 = A[i+2].split()
                x3, y3, z3 = A[i+3].split()
                # Pass as floating point
                x1 = float(x1); y1 = float(y1); z1 = float(z1)
                x2 = float(x2); y2 = float(y2); z2 = float(z2)
                x3 = float(x3); y3 = float(y3); z3 = float(z3)
                # translate to utm
                lon1, lat1 = self.xy2ll(x1, y1)
                lon2, lat2 = self.xy2ll(x2, y2)
                lon3, lat3 = self.xy2ll(x3, y3)
            # Depth
            mm = min([float(z1), float(z2), float(z3)])
            mx = max([float(z1), float(z2), float(z3)])
            if D<mx:
                D=mx
            if d>mm:
                d=mm
            # Set points
            p1 = [x1, y1, z1]; p1ll = [lon1, lat1, z1]
            p2 = [x2, y2, z2]; p2ll = [lon2, lat2, z2]
            p3 = [x3, y3, z3]; p3ll = [lon3, lat3, z3]
            # Store these
            p = [p1, p2, p3]
            pll = [p1ll, p2ll, p3ll]
            p = np.array(p)
            pll = np.array(pll)
            # Store these in the lists
            self.patch.append(p)
            self.patchll.append(pll)
            # increase i
            i += 4

        # Close the file
        fin.close()

        # depth
        self.depth = D
        self.top = d
        self.z_patches = np.linspace(0,D,5)
        self.factor_depth = 1.

        # Patches 2 vertices
        self.setVerticesFromPatches()
        self.numpatch = self.Faces.shape[0]

        # Translate slip to np.array
        if not donotreadslip:
            self.initializeslip(values=np.array(Slip))
        else:
            self.initializeslip()
        if readpatchindex:
            self.index_parameter = np.array(self.index_parameter)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def readPatchesFromAbaqus(self, vertexfile, topofile, readpatchindex=True, 
                             projstr=None, lon0=None, lat0=None):
        """
        Read patches from Abaqus format files
        
        Optimized version: Better code organization and error handling
        """
        try:
            # Data loading and preprocessing
            Vertex, Topology = self._load_and_preprocess_files(
                vertexfile, topofile, readpatchindex
            )
            
            # Coordinate transformation
            Vertices_ll, Vertices, proj_info = self._setup_projection_and_transform(
                projstr, lon0, lat0, Vertex
            )
            
            # Set object attributes
            self._set_coordinate_attributes(Vertices_ll, Vertices, proj_info)
            
            # Extract patches and faces
            self._extract_patches_and_faces(Vertex, Topology)
            
            # Set depth and slip attributes
            self._finalize_fault_setup(Vertices[:, 2])
            
        except Exception as e:
            raise RuntimeError(f"Failed to read Abaqus files: {e}") from e
    
    def _load_and_preprocess_files(self, vertexfile, topofile, readpatchindex):
        """Load and preprocess input files"""
        import pandas as pd
        
        # Read vertex file
        Vertex = pd.read_csv(vertexfile, comment='*')
        vcol = Vertex.columns.str.replace(' ', '')
        Vertex = Vertex.set_axis(vcol, axis=1)
        Vertex.num -= 1  # Convert to 0-based indexing
        
        # Read topology file
        Topology = pd.read_csv(topofile, comment='*')
        tcol = Topology.columns.str.replace(' ', '')
        Topology = Topology.set_axis(tcol, axis=1)
        Topology -= 1  # Convert to 0-based indexing
        
        # Handle index parameters
        if readpatchindex:
            self.vertex_parameter = np.unique(
                np.sort(Topology[['a', 'b', 'c']].values.flatten())
            )
            self.index_parameter = Topology[['a', 'b', 'c']].values.copy()
            self.faces_parameter = Topology[['a', 'b', 'c']].values.copy()
            
            # Keep only vertices used in topology
            Vertex = Vertex.set_index('num')
            Vertex = Vertex.loc[self.vertex_parameter, :]
        
        return Vertex, Topology
    
    def _set_coordinate_attributes(self, Vertices_ll, Vertices, proj_info):
        """Set coordinate-related attributes"""
        self.Vertices_ll = Vertices_ll
        self.Vertices = Vertices
        self.projection_info = proj_info  # Save projection info for later use
        
        if hasattr(self, 'verbose') and self.verbose:
            print(f"Projection info: {proj_info}")
            print(f"Coordinate ranges: "
                  f"Longitude {Vertices_ll[:, 0].min():.6f} - {Vertices_ll[:, 0].max():.6f}, "
                  f"Latitude {Vertices_ll[:, 1].min():.6f} - {Vertices_ll[:, 1].max():.6f}")
    
    def _extract_patches_and_faces(self, Vertex, Topology):
        """Extract patches and faces"""
        patches = []
        Faces = []
        
        for i in range(Topology.shape[0]):
            tri = Topology.iloc[i][['a', 'b', 'c']].values
            patches.append(Vertex.loc[tri].values)
            
            # Build face index mapping
            face = []
            for ntrivert in tri:
                face_idx = np.argwhere(Vertex.index == ntrivert).item(0)
                face.append(face_idx)
            Faces.append(face)
        
        self.Faces = np.array(Faces)
        self.patch = patches
        self.patch2ll()
        self.numpatch = len(self.patch)
    
    def _finalize_fault_setup(self, depths):
        """Complete fault setup"""
        self.top = depths.min()
        self.depth = depths.max()
        self.z_patches = np.linspace(0, self.depth, 5)
        self.factor_depth = 1.0
        
        # Initialize slip
        self.initializeslip()

    def _setup_projection_and_transform(self, projstr, lon0, lat0, Vertex):
        """
        Setup projection and execute coordinate transformation
        
        Returns:
        --------
        tuple: (Vertices_ll, Vertices, proj_info)
        """
        import pandas as pd
        from pyproj import Proj, Transformer, CRS
        from pyproj.database import query_utm_crs_info
        from pyproj.aoi import AreaOfInterest
        import re
        
        if projstr is None:
            return self._handle_lonlat_input(Vertex)
        
        # Parse projection parameters
        proj, proj_info = self._parse_projection_string(projstr, lon0, lat0)
        
        # Execute coordinate transformation
        return self._transform_coordinates(proj, Vertex, proj_info)
    
    def _parse_projection_string(self, projstr, lon0, lat0):
        """Parse projection string and create projection object"""
        import re
        from pyproj import Proj, CRS
        from pyproj.database import query_utm_crs_info
        from pyproj.aoi import AreaOfInterest
        
        proj_info = {'type': 'custom', 'string': projstr}
        
        if 'utm' in projstr:
            if 'zone' in projstr:
                # UTM projection with specified zone
                proj = Proj(projstr)
                proj_info['type'] = 'utm_with_zone'
            else:
                # UTM projection without specified zone, auto-calculate
                proj, proj_info = self._auto_determine_utm_zone(projstr, lon0, lat0)
        else:
            # Non-UTM projection (e.g., tmerc, gauss)
            proj = Proj(projstr)
            proj_info['type'] = 'other'
        
        return proj, proj_info
    
    def _auto_determine_utm_zone(self, projstr, lon0, lat0):
        """Automatically determine optimal UTM zone"""
        import re
        from pyproj import CRS
        from pyproj import Proj
        from pyproj.database import query_utm_crs_info
        from pyproj.aoi import AreaOfInterest
        
        # Try to extract coordinates from projstr
        if lon0 is None or lat0 is None:
            lon0_match = re.search(r'\+lon_0=(-?\d+(\.\d+)?)', projstr)
            lat0_match = re.search(r'\+lat_0=(-?\d+(\.\d+)?)', projstr)
            if lon0_match and lat0_match:
                lon0 = float(lon0_match.group(1))
                lat0 = float(lat0_match.group(1))
            else:
                raise ValueError(
                    "UTM zone not specified and cannot extract lon0 and lat0 from projstr, "
                    "please provide lon0 and lat0 parameters"
                )
        
        # Query optimal UTM zone
        utm_crs_list = query_utm_crs_info(
            datum_name="WGS 84",
            area_of_interest=AreaOfInterest(
                west_lon_degree=lon0 - 2.,
                south_lat_degree=lat0 - 2.,
                east_lon_degree=lon0 + 2,
                north_lat_degree=lat0 + 2
            ),
        )
        
        if not utm_crs_list:
            raise ValueError(f"Cannot find suitable UTM zone for coordinates ({lon0}, {lat0})")
        
        utm_crs = CRS.from_epsg(utm_crs_list[0].code)
        proj = Proj(utm_crs)
        
        proj_info = {
            'type': 'utm_auto',
            'original_string': projstr,
            'lon0': lon0,
            'lat0': lat0,
            'epsg_code': utm_crs_list[0].code,
            'zone_info': utm_crs_list[0]
        }
        
        return proj, proj_info
    
    def _transform_coordinates(self, proj, Vertex, proj_info):
        """Execute coordinate transformation"""
        from pyproj import Transformer
        
        # Create transformer, force longitude-latitude order
        transformer = Transformer.from_proj(
            proj, 
            proj.to_latlong(), 
            always_xy=True  # Ensure output is (longitude, latitude) order
        )
        
        # Execute coordinate transformation
        lon, lat = transformer.transform(Vertex.x.values, Vertex.y.values)
        
        # Depth conversion (meters to kilometers, negative values for depth)
        depth = -Vertex.z.values / 1000.0
        
        # Create longitude/latitude coordinate array
        Vertices_ll = np.vstack((lon, lat, depth)).T
        
        # Convert to UTM coordinates
        x, y = self.ll2xy(lon, lat)
        
        # Update Vertex DataFrame
        Vertex.x = x
        Vertex.y = y
        Vertex.z = depth
        
        # Create UTM coordinate array
        Vertices = np.vstack((x, y, depth)).T
        
        return Vertices_ll, Vertices, proj_info
    
    def _handle_lonlat_input(self, Vertex):
        """Handle longitude/latitude format input data"""
        lon, lat = Vertex.x.values, Vertex.y.values
        depth = -Vertex.z.values / 1000.0
        
        Vertices_ll = np.vstack((lon, lat, depth)).T
        
        x, y = self.ll2xy(lon, lat)
        Vertex.x = x
        Vertex.y = y
        Vertex.z = depth
        
        Vertices = np.vstack((x, y, depth)).T
        
        proj_info = {'type': 'lonlat', 'string': 'Geographic coordinates'}
        
        return Vertices_ll, Vertices, proj_info
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def readGocadPatches(self, filename, neg_depth=False, utm=False, factor_xy=1.0,
                         factor_depth=1.0, verbose=False):
        '''
        Load a triangulated surface from a Gocad formatted file. Vertices 
        must be in geographical coordinates.

        Args:
            * filename:  tsurf file to read

        Kwargs:
            * neg_depth: if true, use negative depth
            * utm: if true, input file is given as utm coordinates (if false -> lon/lat)
            * factor_xy: if utm==True, multiplication factor for x and y
            * factor_depth: multiplication factor for z
            * verbose: Speak to me

        Returns:
            * None
        '''

        # Initialize the lists of patches
        self.patch   = []
        self.patchll = []

        # Factor to correct input negative depths (we want depths to be positive)
        if neg_depth:
            negFactor = -1.0
        else:
            negFactor =  1.0

        # Get the geographic vertices and connectivities from the Gocad file
        with open(filename, 'r') as fid:
            vertices = []
            vids     = []
            faces    = []
            for line in fid:
                if line.startswith('VRTX'):
                    items = line.split()
                    name, vid, x, y, z = items[:5]
                    vids.append(vid)
                    vertices.append([float(x), float(y), negFactor*float(z)])
                elif line.startswith('TRGL'):
                    name, p1, p2, p3 = line.split()
                    faces.append([int(p1), int(p2), int(p3)])
            fid.close()
            vids = np.array(vids,dtype=int)
            i0 = np.min(vids)
            vids = vids - i0
            i    = np.argsort(vids)
            vertices = np.array(vertices, dtype=float)[i,:]
            faces = np.array(faces, dtype=int) - i0

        # Resample vertices to UTM
        if utm:
            vx = vertices[:,0].copy()*factor_xy
            vy = vertices[:,1].copy()*factor_xy
            vertices[:,0],vertices[:,1] = self.xy2ll(vx,vy)
        else:
            vx, vy = self.ll2xy(vertices[:,0], vertices[:,1])
        vz = vertices[:,2]*factor_depth
        self.factor_depth = factor_depth
        self.Vertices = np.column_stack((vx, vy, vz))
        self.Vertices_ll = vertices
        self.Faces = faces
        if verbose:
            print('min/max depth: {} km/ {} km'.format(vz.min(),vz.max()))
            print('min/max lat: {} deg/ {} deg'.format(vertices[:,1].min(),vertices[:,1].max()))
            print('min/max lon: {} deg/ {} deg'.format(vertices[:,0].min(),vertices[:,0].max()))
            print('min/max x: {} km/ {} km'.format(vx.min(),vx.max()))
            print('min/max y: {} km/ {} km'.format(vy.min(),vy.max()))

        # Loop over faces and create a triangular patch consisting of coordinate tuples
        self.numpatch = faces.shape[0]
        for i in range(self.numpatch):
            # Get the indices of the vertices
            v1, v2, v3 = faces[i,:]
            # Get the coordinates
            x1, y1, lon1, lat1, z1 = vx[v1], vy[v1], vertices[v1,0], vertices[v1,1], vz[v1]
            x2, y2, lon2, lat2, z2 = vx[v2], vy[v2], vertices[v2,0], vertices[v2,1], vz[v2]
            x3, y3, lon3, lat3, z3 = vx[v3], vy[v3], vertices[v3,0], vertices[v3,1], vz[v3]
            # Make the coordinate tuples
            p1 = [x1, y1, z1]; pll1 = [lon1, lat1, z1]
            p2 = [x2, y2, z2]; pll2 = [lon2, lat2, z2]
            p3 = [x3, y3, z3]; pll3 = [lon3, lat3, z3]
            # Store the patch
            self.patch.append(np.array([p1, p2, p3]))
            self.patchll.append(np.array([pll1, pll2, pll3]))

        # Update the depth of the bottom of the fault
        if neg_depth:
            self.top   = np.max(vz)
            self.depth = np.min(vz)
        else:
            self.top   = np.min(vz)
            self.depth = np.max(vz)
        self.z_patches = np.linspace(self.depth, 0.0, 5)
        
        self.initializeslip()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writeGocadPatches(self, filename, utm=False):
        '''
        Write a triangulated Gocad surface file.

        Args:
            * filename  : output file name

        Kwargs:
            * utm       : Write in utm coordinates if True

        Returns:
            * None
        '''

        # Get the geographic vertices and connectivities from the Gocad file

        fid = open(filename, 'w')
        if utm:
            vertices = self.Vertices*1.0e3
        else:
            vertices = self.Vertices_ll
        for i in range(vertices.shape[0]):
            v = vertices[i]
            fid.write('VRTX {} {} {} {}\n'.format(i+1,v[0],v[1],v[2]))
        for i in range(self.Faces.shape[0]):
            vid = self.Faces[i,:]+1
            fid.write('TRGL {} {} {}\n'.format(vid[0],vid[1],vid[2]))
        fid.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getStrikes(self):
        '''
        Returns an array of strikes.
        '''

        # all done in one line
        return np.array([self.getpatchgeometry(p)[5] for p in self.patch])
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getDips(self):
        '''
        Returns an array of dips.
        '''

        # all done in one line
        return np.array([self.getpatchgeometry(p)[6] for p in self.patch])
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getDepths(self, center=True):
        '''
        Returns an array of depths.

        Kwargs:
            * center        : If True, returns the center of the patches
        '''

        # All done in one line
        return np.array([self.getpatchgeometry(p, center=True)[2] for p in self.patch]) 

    # ----------------------------------------------------------------------
    def writePatches2File(self, filename, add_slip=None, scale=1.0, stdh5=None, decim=1):
        '''
        Writes the patch corners in a file that can be used in psxyz.

        Args:
            * filename      : Name of the file.

        Kwargs:
            * add_slip      : Put the slip as a value for the color. 
                              Can be None, strikeslip, dipslip, total, coupling
            * scale         : Multiply the slip value by a factor.
            * patch         : Can be 'normal' or 'equiv'
            * stdh5         : Get the standard deviation from a h5 file
            * decim         : Decimate the h5 file

        Returns:
            * None
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch) and add_slip is not None:
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Write something
        print('Writing geometry to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # If an h5 file is specified, open it
        if stdh5 is not None:
            import h5py
            h5fid = h5py.File(stdh5, 'r')
            samples = h5fid['samples'].value[::decim,:]

        # Loop over the patches
        nPatches = len(self.patch)
        for pIndex in range(nPatches):

            # Select the string for the color
            string = '  '
            if add_slip is not None:
                if add_slip == 'coupling':
                    slp = self.coupling[pIndex]
                    string = '-Z{}'.format(slp)
                if add_slip == 'strikeslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex])
                    else:
                        slp = self.slip[pIndex,0]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip == 'dipslip':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex+nPatches])
                    else:
                        slp = self.slip[pIndex,1]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip == 'tensile':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex+2*nPatches])
                    else:
                        slp = self.slip[pIndex,2]*scale
                    string = '-Z{}'.format(slp)
                elif add_slip == 'total':
                    if stdh5 is not None:
                        slp = np.std(samples[:,pIndex]**2 + samples[:,pIndex+nPatches]**2)
                    else:
                        slp = np.sqrt(self.slip[pIndex,0]**2 + self.slip[pIndex,1]**2)*scale
                    string = '-Z{}'.format(slp)

            # Put the parameter number in the file as well if it exists
            parameter = ' '
            if hasattr(self,'index_parameter') and add_slip is not None:
                i = np.int64(self.index_parameter[pIndex,0])
                j = np.int64(self.index_parameter[pIndex,1])
                k = np.int64(self.index_parameter[pIndex,2])
                parameter = '# {} {} {} '.format(i,j,k)

            # Put the slip value
            if add_slip is not None:
                if add_slip=='coupling':
                    slipstring = ' # {}'.format(self.coupling[pIndex])
                else:
                    slipstring = ' # {} {} {} '.format(self.slip[pIndex,0],
                                               self.slip[pIndex,1], self.slip[pIndex,2])

            # Write the string to file
            if add_slip is None:
                fout.write('> {} {} \n'.format(string, parameter))
            else:
                fout.write('> {} {} {}  \n'.format(string,parameter,slipstring))

            # Write the 3 patch corners (the order is to be GMT friendly)
            p = self.patchll[pIndex]
            pp = p[0]; fout.write('{} {} {} \n'.format(np.round(pp[0], decimals=4), 
                                                       np.round(pp[1], decimals=4), 
                                                       np.round(pp[2], decimals=4)))
            pp = p[1]; fout.write('{} {} {} \n'.format(np.round(pp[0], decimals=4), 
                                                       np.round(pp[1], decimals=4), 
                                                       np.round(pp[2], decimals=4)))
            pp = p[2]; fout.write('{} {} {} \n'.format(np.round(pp[0], decimals=4), 
                                                       np.round(pp[1], decimals=4), 
                                                       np.round(pp[2], decimals=4)))

        # Close the file
        fout.close()

        # Close h5 file if it is open
        if stdh5 is not None:
            h5fid.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writeSlipDirection2File(self, filename, scale=1.0, factor=1.0,
                                neg_depth=False, ellipse=False, nsigma=1., reference_strike=None, threshold=0.0):
        '''
        Write a psxyz compatible file to draw lines starting from the center 
        of each patch, indicating the direction of slip. Scale can be a real 
        number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
    
        Args:
            * filename      : Name of the output file
    
        Kwargs:
            * scale         : Scale of the line
            * factor        : Multiply slip by a factor
            * neg_depth     : if True, depth is a negative number
            * ellipse       : Write the ellipse
            * nsigma        : Nxsigma for the ellipse
            * reference_strike : Reference strike direction in degrees. If the patch strike differs by 180 degrees, adjust rake by 180 degrees.
            * threshold     : Threshold value for sca before multiplying by factor.
    
        Returns:
            * None
        '''
    
        # Compute the slip direction
        self.computeSlipDirection(scale=scale, factor=factor, ellipse=ellipse, nsigma=nsigma, neg_depth=neg_depth, reference_strike=reference_strike, threshold=threshold)
    
        # Write something
        print('Writing slip direction to file {}'.format(filename))
    
        # Open the file
        fout = open(filename, 'w')
    
        # Loop over the patches
        for p, above_threshold in zip(self.slipdirection, self.slipdirection_above_threshold):
            if not above_threshold:
                continue
    
            # Write the > sign to the file
            fout.write('> \n')
    
            # Get the center of the patch
            xc, yc, zc = p[0]
            lonc, latc = self.xy2ll(xc, yc)
            if neg_depth:
                zc = -1.0*zc
            fout.write('{} {} {} \n'.format(lonc, latc, zc))
    
            # Get the end of the vector
            xc, yc, zc = p[1]
            lonc, latc = self.xy2ll(xc, yc)
            if neg_depth:
                zc = -1.0*zc
            fout.write('{} {} {} \n'.format(lonc, latc, zc))
    
        # Close file
        fout.close()
    
        if ellipse:
            # Open the file
            fout = open('ellipse_'+filename, 'w')
    
            # Loop over the patches
            for e in self.ellipse:
    
                # Get ellipse points
                ex, ey, ez = e[:,0],e[:,1],e[:,2]
    
                # Depth
                if neg_depth:
                    ez = -1.0 * ez
    
                # Conversion to geographical coordinates
                lone,late = self.xy2ll(ex, ey)
    
                # Write the > sign to the file
                fout.write('> \n')
    
                for lon,lat,z in zip(lone,late,ez):
                    fout.write('{} {} {} \n'.format(lon, lat, -1.*z))
            # Close file
            fout.close()
    
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writeSlipCenter2File(self, filename, add_slip=None, scale=1.0, neg_depth=False):
                                
        '''
        Write a psxyz, surface or greenspline compatible file with the center 
        of each patch and magnitude slip. Scale can be a real 
        number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'

        Args:
            * filename      : Name of the output file

        Kwargs:
            * add_slip      : Put the slip as a value at the center 
                              Can be None, strikeslip, dipslip, total, coupling
            * scale         : Multiply the slip value by a factor.
            * neg_depth     : if True, depth is a negative nmber
 
        Returns:
            * None
        '''

        # Write something
        print('Writing slip at patch centers to file {}'.format(filename))

        # Open the file
        fout = open(filename, 'w')

        # Loop over the patches
        if self.N_slip == None:
            self.N_slip = len(self.patch)
        for pIndex in range(self.N_slip):
            # Get the center of the patch
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(pIndex, center=True)
            
           # Get the slip value to be added
            if add_slip is not None:
                if add_slip == 'coupling':
                    slp = self.coupling[pIndex]
                if add_slip == 'strikeslip':
                    slp = self.slip[pIndex,0]*scale
                elif add_slip == 'dipslip':
                    slp = self.slip[pIndex,1]*scale
                elif add_slip == 'tensile':
                    slp = self.slip[pIndex,2]*scale
                elif add_slip == 'total':
                    slp = np.sqrt(self.slip[pIndex,0]**2 + self.slip[pIndex,1]**2)*scale
 
            # project center of the patch to lat-long
            lonc, latc = self.xy2ll(xc, yc)
            if neg_depth:
                zc = -1.0*zc
                
            fout.write('{} {} {} {}\n'.format(lonc, latc, zc, slp))

        # Close file
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def writeFourEdges2File(self, filename=None, dirname='.', top_tolerance=0.1, 
                            bottom_tolerance=0.1, method='hybrid', merge_threshold=0.02):
        '''
        Write the four edges of the patches to a file.

        Kwargs:
            * filename      : Name of the output file
            * dirname       : Directory where the files will be written
            * top_tolerance : Tolerance for the top edge
            * bottom_tolerance : Tolerance for the bottom edge

        Returns:
            * None
        '''
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        self.find_fault_fouredge_vertices(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance,
                                           refind=True, method=method, merge_threshold=merge_threshold)

        # Write edges to four edges file
        edge_names = ['top', 'bottom', 'left', 'right']
        four_edges = self.edge_vertices
        if filename is not None:
            basename = filename.split('.')[0]
            extname = filename.split('.')[1]
            for i, edge in enumerate(four_edges):
                edge_file = '{}_{}.{}'.format(basename, edge_names[i], extname)
                edge_file = os.path.join(dirname, edge_file)
                with open(edge_file, 'w') as fout:
                    edge_points = four_edges[edge]
                    lon, lat = self.xy2ll(edge_points[:,0], edge_points[:,1])
                    for i in range(len(lon)):
                        fout.write('{} {} {}\n'.format(lon[i], lat[i], edge_points[i,2]))
        else:
            for i, edge in enumerate(four_edges):
                edge_file = '{}_{}.{}'.format(self.name, edge_names[i], 'gmt')
                edge_file = os.path.join(dirname, edge_file)
                with open(edge_file, 'w') as fout:
                    edge_points = four_edges[edge]
                    lon, lat = self.xy2ll(edge_points[:,0], edge_points[:,1])
                    for i in range(len(lon)):
                        fout.write('{} {} {}\n'.format(lon[i], lat[i], edge_points[i,2]))

        # All Done
        return four_edges
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def interpolate_curve_at_depth(self, target_depth, variable_axis='auto', ascending=True, 
                                  depth_tolerance=0.1, output_file=None, verbose=True,
                                  interpolation_method='linear', num_points='auto',
                                  top_tolerance=0.01, bottom_tolerance=0.01, 
                                  method='hybrid', merge_threshold=0.5):
        """
        Interpolate a curve at specified depth using triangular mesh edge vertices
        
        This method first finds the four edge vertices of the fault, then interpolates
        a curve at the target depth based on the left and right boundary vertices.
        
        Parameters:
        -----------
        target_depth : float
            Target depth for curve generation (positive value, in km)
        variable_axis : str, optional
            Independent variable axis ('x', 'y', or 'auto'). Default: 'auto'
            - 'x': Use X-coordinate as independent variable
            - 'y': Use Y-coordinate as independent variable  
            - 'auto': Automatically choose best axis based on edge span
        ascending : bool, optional
            Sort order for the independent variable (default: True)
        depth_tolerance : float, optional
            Tolerance for interpolating at target depth (default: 0.1 km)
        output_file : str, optional
            Output GMT format file path (default: None, no file output)
        verbose : bool, optional
            Enable detailed output messages (default: True)
        interpolation_method : str, optional
            Interpolation method for griddata ('linear', 'nearest', 'cubic'). Default: 'linear'
        num_points : int or str, optional
            Number of points to generate in the curve (default: 'auto')
            If 'auto', interpolates between top and bottom edge point counts
        top_tolerance : float, optional
            Tolerance for finding top edge vertices (default: 0.01 km)
        bottom_tolerance : float, optional
            Tolerance for finding bottom edge vertices (default: 0.01 km)
        method : str, optional
            Method for edge reconstruction (default: 'hybrid')
        merge_threshold : float, optional
            Threshold for merging duplicate vertices (default: 0.5)
            
        Returns:
        --------
        curve_lonlat : ndarray
            Interpolated curve points (N, 3) in [longitude, latitude, depth] format
        curve_info : dict
            Dictionary containing interpolation information
        """
        from scipy.interpolate import griddata
        
        if verbose:
            print(f"Interpolating curve at depth {target_depth:.2f} km for triangular fault {self.name}")
        
        target_depth = abs(target_depth)  # Ensure positive depth
        
        # First, find the four edge vertices if not already found
        if not hasattr(self, 'edge_vertices') or not hasattr(self, 'edge_vertex_indices'):
            if verbose:
                print("Finding fault four edge vertices...")
            self.find_fault_fouredge_vertices(
                top_tolerance=top_tolerance, 
                bottom_tolerance=bottom_tolerance,
                refind=True, 
                method=method, 
                merge_threshold=merge_threshold
            )
        
        # Get the four edge vertices
        left_vertices = self.edge_vertices['left']
        right_vertices = self.edge_vertices['right']
        top_vertices = self.edge_vertices['top']
        bottom_vertices = self.edge_vertices['bottom']
        
        if verbose:
            print(f"Found edge vertices:")
            print(f"  Left edge: {len(left_vertices)} vertices")
            print(f"  Right edge: {len(right_vertices)} vertices")
            print(f"  Top edge: {len(top_vertices)} vertices")
            print(f"  Bottom edge: {len(bottom_vertices)} vertices")
        
        # Determine variable axis based on edge span
        if variable_axis == 'auto':
            x_range_top = np.max(top_vertices[:, 0]) - np.min(top_vertices[:, 0])
            x_range_bottom = np.max(bottom_vertices[:, 0]) - np.min(bottom_vertices[:, 0])
            y_range_top = np.max(top_vertices[:, 1]) - np.min(top_vertices[:, 1])
            y_range_bottom = np.max(bottom_vertices[:, 1]) - np.min(bottom_vertices[:, 1])

            x_range_avg = (x_range_top + x_range_bottom) / 2
            y_range_avg = (y_range_top + y_range_bottom) / 2

            variable_axis = 'x' if x_range_avg >= y_range_avg else 'y'

            if verbose:
                print(f"Auto-selected variable axis: {variable_axis}")
                print(f"Average X span: {x_range_avg:.2f} km, Average Y span: {y_range_avg:.2f} km")

        axis_idx = 0 if variable_axis.lower() == 'x' else 1
        other_idx = 1 if variable_axis.lower() == 'x' else 0
        
        # Determine number of points based on top and bottom edge counts
        if num_points == 'auto':
            top_count = len(top_vertices)
            bottom_count = len(bottom_vertices)
            
            # Get depths for interpolation
            top_depth = np.mean(np.abs(top_vertices[:, 2]))
            bottom_depth = np.mean(np.abs(bottom_vertices[:, 2]))
            
            # Linear interpolation of point count based on depth
            if bottom_depth > top_depth:
                depth_factor = (target_depth - top_depth) / (bottom_depth - top_depth)
                depth_factor = max(0, min(1, depth_factor))  # Clamp to [0, 1]
                num_points = int(top_count + depth_factor * (bottom_count - top_count))
            else:
                num_points = top_count
            
            num_points = max(10, num_points)  # Minimum 10 points
            
            if verbose:
                print(f"Interpolated point count: {num_points} (top: {top_count}, bottom: {bottom_count})")
        
        # Find boundary coordinates at target depth by interpolating left and right edges
        # Interpolate left boundary
        left_depths = np.abs(left_vertices[:, 2])
        if len(left_vertices) > 1:
            left_boundary_coord = np.interp(target_depth, left_depths, left_vertices[:, axis_idx])
            left_other_coord = np.interp(target_depth, left_depths, left_vertices[:, other_idx])
        else:
            left_boundary_coord = left_vertices[0, axis_idx]
            left_other_coord = left_vertices[0, other_idx]
        
        # Interpolate right boundary
        right_depths = np.abs(right_vertices[:, 2])
        if len(right_vertices) > 1:
            right_boundary_coord = np.interp(target_depth, right_depths, right_vertices[:, axis_idx])
            right_other_coord = np.interp(target_depth, right_depths, right_vertices[:, other_idx])
        else:
            right_boundary_coord = right_vertices[0, axis_idx]
            right_other_coord = right_vertices[0, other_idx]
        
        if verbose:
            coord_name = 'X' if variable_axis.lower() == 'x' else 'Y'
            print(f"Boundary {coord_name} coordinates at depth {target_depth:.2f} km:")
            print(f"  Left boundary: {left_boundary_coord:.3f}")
            print(f"  Right boundary: {right_boundary_coord:.3f}")
        
        # Generate interpolation points along the independent axis
        if ascending:
            target_coords = np.linspace(left_boundary_coord, right_boundary_coord, num_points)
        else:
            target_coords = np.linspace(right_boundary_coord, left_boundary_coord, num_points)
        
        # Collect all vertices for interpolation
        all_vertices = np.vstack([left_vertices, right_vertices, top_vertices, bottom_vertices])
        # Remove duplicates based on coordinates
        unique_vertices = []
        tolerance = merge_threshold
        for vertex in all_vertices:
            is_duplicate = False
            for unique_vertex in unique_vertices:
                if np.linalg.norm(vertex - unique_vertex) < tolerance:
                    is_duplicate = True
                    break
            if not is_duplicate:
                unique_vertices.append(vertex)
        
        all_vertices = np.array(unique_vertices)
        
        if verbose:
            print(f"Using {len(all_vertices)} unique vertices for interpolation")
        
        # Perform 2D interpolation to get the other coordinate
        points = np.column_stack([all_vertices[:, axis_idx], all_vertices[:, 2]])  # (variable_axis, depth)
        values = all_vertices[:, other_idx]  # other coordinate values
        
        # Create target points for interpolation
        target_points = np.column_stack([target_coords, np.full(num_points, target_depth)])
        
        # Interpolate the other coordinate
        try:
            target_other_coords = griddata(points, values, target_points, method=interpolation_method)
            
            # Handle NaN values by using nearest neighbor interpolation as fallback
            nan_mask = np.isnan(target_other_coords)
            if np.any(nan_mask):
                if verbose:
                    print(f"Found {np.sum(nan_mask)} NaN values, using nearest neighbor fallback")
                target_other_coords[nan_mask] = griddata(points, values, target_points[nan_mask], method='nearest')
            
        except Exception as e:
            if verbose:
                print(f"Interpolation failed: {e}, using linear interpolation between boundaries")
            # Fallback: linear interpolation between left and right boundaries
            target_other_coords = np.linspace(left_other_coord, right_other_coord, num_points)
        
        # Create curve points
        if variable_axis.lower() == 'x':
            curve_points = np.column_stack([target_coords, target_other_coords, np.full(num_points, target_depth)])
        else:
            curve_points = np.column_stack([target_other_coords, target_coords, np.full(num_points, target_depth)])
        
        # Remove any remaining NaN values
        valid_mask = ~np.isnan(curve_points).any(axis=1)
        curve_points = curve_points[valid_mask]
        
        if len(curve_points) == 0:
            raise ValueError("Interpolation failed - no valid points generated")
        
        if verbose:
            print(f"Generated {len(curve_points)} valid curve points")
            print(f"Depth range: {curve_points[:, 2].min():.3f} - {curve_points[:, 2].max():.3f} km")
        
        # Convert to longitude/latitude coordinates
        x_coords = curve_points[:, 0]
        y_coords = curve_points[:, 1]
        depths = curve_points[:, 2]
        
        lon_coords, lat_coords = self.xy2ll(x_coords, y_coords)
        curve_lonlat = np.column_stack([lon_coords, lat_coords, depths])
        
        if verbose:
            print(f"Coordinate conversion completed")
            print(f"Longitude range: {lon_coords.min():.6f} to {lon_coords.max():.6f}")
            print(f"Latitude range: {lat_coords.min():.6f} to {lat_coords.max():.6f}")
        
        # Create curve info
        curve_info = {
            'method': f'edge_based_{interpolation_method}',
            'edge_vertices_used': {
                'left': len(left_vertices),
                'right': len(right_vertices), 
                'top': len(top_vertices),
                'bottom': len(bottom_vertices)
            },
            'variable_axis': variable_axis,
            'point_count': len(curve_points),
            'depth_range': [curve_points[:, 2].min(), curve_points[:, 2].max()],
            'target_depth': target_depth,
            'interpolation_method': interpolation_method,
            'coordinate_system': 'longitude_latitude',
            'fault_name': self.name,
            'boundary_coords': {
                'left': left_boundary_coord,
                'right': right_boundary_coord
            },
            'lon_range': [lon_coords.min(), lon_coords.max()],
            'lat_range': [lat_coords.min(), lat_coords.max()]
        }
        
        # Output to GMT file if requested
        if output_file is not None:
            self._write_curve_lonlat_to_gmt(curve_lonlat, output_file, curve_info, verbose)
        
        return curve_lonlat, curve_info
    
    def _write_curve_lonlat_to_gmt(self, curve_lonlat, output_file, curve_info, verbose):
        """
        Write longitude/latitude curve to GMT format file for triangular patches
        """
        import os
        
        try:
            # Create output directory if it doesn't exist
            output_dir = os.path.dirname(output_file)
            if output_dir and not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            with open(output_file, 'w') as f:
                # Write comprehensive header
                f.write(f"# Interpolated triangular mesh curve for {self.name}\n")
                f.write(f"# Target depth: {curve_info['target_depth']:.2f} km\n")
                f.write(f"# Actual depth range: {curve_info['depth_range'][0]:.3f} - {curve_info['depth_range'][1]:.3f} km\n")
                f.write(f"# Interpolation method: {curve_info['method']}\n")
                # f.write(f"# Source vertices: {curve_info['source_vertices']}\n")
                f.write(f"# Variable axis: {curve_info['variable_axis']}\n")
                f.write(f"# Point count: {curve_info['point_count']}\n")
                # f.write(f"# Depth tolerance: {curve_info['depth_tolerance']:.3f} km\n")
                f.write(f"# Longitude range: {curve_info['lon_range'][0]:.6f} to {curve_info['lon_range'][1]:.6f}\n")
                f.write(f"# Latitude range: {curve_info['lat_range'][0]:.6f} to {curve_info['lat_range'][1]:.6f}\n")
                f.write(f"# Format: Longitude Latitude Depth\n")
                f.write(f"# Coordinate system: WGS84\n")
                f.write(f"# Mesh type: Triangular patches\n")
                f.write(f">\n")  # GMT segment separator
                
                # Write data points
                for point in curve_lonlat:
                    f.write(f"{point[0]:.8f} {point[1]:.8f} {point[2]:.3f}\n")
            
            if verbose:
                print(f"Triangular mesh curve written to GMT file: {output_file}")
                print(f"File contains {len(curve_lonlat)} points in longitude/latitude format")
                
        except Exception as e:
            print(f"Error writing GMT file: {e}")
            raise
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getEllipse(self, patch, ellipseCenter=None, Npoints=10, factor=1.0,
                   nsigma=1.):
        '''
        Compute the ellipse error given Cm for a given patch

        Args:
            * patch : Which patch to consider

        Kwargs:
            * center  : center of the ellipse
            * Npoints : number of points on the ellipse
            * factor  : scaling factor
            * nsigma  : will design a nsigma*sigma error ellipse

        Returns:
            * Ellipse   : Array containing the ellipse
        '''

        # Get Cm
        Cm = np.diag(self.Cm[patch,:2])
        Cm[0,1] = Cm[1,0] = self.Cm[patch,2]

        # Get strike and dip
        xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(patch, center=True)
        dip *= np.pi/180.
        strike *= np.pi/180.
        if ellipseCenter!=None:
            xc,yc,zc = ellipseCenter

        # Compute eigenvalues/eigenvectors
        D,V = np.linalg.eig(Cm)
        v1 = V[:,0]
        a = nsigma*np.sqrt(np.abs(D[0]))
        b = nsigma*np.sqrt(np.abs(D[1]))
        phi = np.arctan2(v1[1],v1[0])
        theta = np.linspace(0,2*np.pi,Npoints);

        # The ellipse in x and y coordinates
        Ex = a * np.cos(theta) * factor
        Ey = b * np.sin(theta) * factor

        # Correlation Rotation
        R  = np.array([[np.cos(phi), -np.sin(phi)],
                       [np.sin(phi), np.cos(phi)]])
        RE = np.dot(R,np.array([Ex,Ey]))

        # Strike/Dip rotation
        ME = np.array([RE[0,:], RE[1,:] * np.cos(dip), RE[1,:]*np.sin(dip)])
        R  = np.array([[np.sin(strike), -np.cos(strike), 0.0],
                       [np.cos(strike), np.sin(strike), 0.0],
                       [0.0, 0.0, 1.]])
        RE = np.dot(R,ME).T

        # Translation on Fault
        RE[:,0] += xc
        RE[:,1] += yc
        RE[:,2] += zc

        # All done
        return RE
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def computeSlipDirection(self, scale=1.0, factor=1.0, ellipse=False, nsigma=1., neg_depth=False, reference_strike=None, threshold=0.0):
        '''
        Computes the segment indicating the slip direction.
    
        Kwargs:
            * scale            : can be a real number or a string in 'total', 'strikeslip', 'dipslip' or 'tensile'
            * factor           : Multiply by a factor
            * ellipse          : Compute the ellipse
            * nsigma           : How many times sigma for the ellipse
            * reference_strike : Reference strike direction in degrees. If the patch strike differs by 180 degrees, adjust rake by 180 degrees.
            * threshold        : Threshold value for sca before multiplying by factor.
    
        Returns:
            * None
        '''
    
        # Create the array
        self.slipdirection = []
        self.slipdirection_above_threshold = []
    
        # Check Cm if ellipse
        if ellipse:
            self.ellipse = []
            assert((self.Cm != None).all()), 'Provide Cm values'
    
        # Loop over the patches
        if self.N_slip is None:
            self.N_slip = len(self.patch)
        for p in range(self.N_slip):
            # Get some geometry
            xc, yc, zc, width, length, strike, dip = self.getpatchgeometry(p, center=True)
            
            # Adjust strike and rake if necessary
            if reference_strike is not None:
                reference_strike_rad = np.radians(reference_strike)
                strike_vector = np.array([np.cos(strike), np.sin(strike)])
                reference_vector = np.array([np.cos(reference_strike_rad), np.sin(reference_strike_rad)])
                dot_product = np.dot(strike_vector, reference_vector)
                if dot_product < 0:
                    rake_adjustment = np.pi
                else:
                    rake_adjustment = 0.0
            else:
                rake_adjustment = 0.0
    
            # Get the slip vector
            slip = self.slip[p, :]
            rake = np.arctan2(slip[1], slip[0]) + rake_adjustment
    
            # Compute the vector
            x = (np.sin(strike) * np.cos(rake) - np.cos(strike) * np.cos(dip) * np.sin(rake))
            y = (np.cos(strike) * np.cos(rake) + np.sin(strike) * np.cos(dip) * np.sin(rake))
            if neg_depth:    # do we need to flip depths? EJF 2020/10/18
                z = 1.0 * np.sin(dip) * np.sin(rake)
            else:
                z = -1.0 * np.sin(dip) * np.sin(rake)
    
            # Scale these
            if isinstance(scale, float):
                sca = scale
            elif isinstance(scale, str):
                if scale == 'total':
                    sca = np.sqrt(slip[0]**2 + slip[1]**2 + slip[2]**2)
                elif scale == 'strikeslip':
                    sca = np.abs(slip[0])
                elif scale == 'dipslip':
                    sca = np.abs(slip[1])
                elif scale == 'tensile':
                    sca = np.abs(slip[2])
                else:
                    print('Unknown Slip Direction in computeSlipDirection')
                    sys.exit(1)
            
            # Check if sca exceeds the threshold
            above_threshold = sca > threshold
            self.slipdirection_above_threshold.append(above_threshold)
    
            sca *= factor
            x *= sca
            y *= sca
            z *= sca
    
            # update point
            xe = xc + x
            ye = yc + y
            ze = zc + z
    
            # Append ellipse
            if ellipse:
                self.ellipse.append(self.getEllipse(p, ellipseCenter=[xe, ye, ze], factor=factor, nsigma=nsigma))
    
            # Append slip direction
            self.slipdirection.append([[xc, yc, zc], [xe, ye, ze]])
    
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletepatch(self, patch, checkVertices=True, checkSlip=False):
        '''
        Deletes a patch.

        Args:
            * patch     : index of the patch to remove.

        Kwargs:
            * checkVertices : Make sure vertice array corresponds to patch corners
            * checkSlip     : Check that slip vector corresponds to patch corners

        Returns:
            * None
        '''

        # Save vertices
        vids = copy.deepcopy(self.Faces[patch])

        # Remove the patch
        del self.patch[patch]
        del self.patchll[patch]
        self.Faces = np.delete(self.Faces, patch, axis=0)
        
        # Check if vertices are to be removed
        v2del = []
        if checkVertices:
            for v in vids:
                if v not in self.Faces.flatten().tolist():
                    v2del.append(v)

        # Remove if needed
        if len(v2del)>0:
            self.deletevertices(v2del, checkPatch=False)

        # Clean slip vector
        if self.slip is not None and checkSlip:
            self.slip = np.delete(self.slip, patch, axis=0)
            self.N_slip = len(self.slip)
            if hasattr(self, 'numpatch'):
                self.numpatch -= 1
            else:
                self.numpatch = len(self.patch)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletevertices(self, iVertices, checkPatch=True, checkSlip=True):
        '''
        Deletes some vertices. If some patches are composed of these vertices 
        and checkPatch is True, deletes the patches.

        Args:
            * iVertices     : List of vertices to delete.

        Kwargs:
            * checkPatch    : Check and delete if patches are concerned.
            * checkSlip     : Check and delete if slip terms are concerned.

        Returns:
            * None
        ''' 

        # If some patches are concerned
        if checkPatch:
            # Check
            iPatch = []
            for iV in iVertices:
                i, j = np.where(self.Faces==iV)
                if len(i.tolist())>0:
                    iPatch.append(np.unique(i))
            if len(iPatch)>0:
                # Delete
                iPatch = np.unique(np.concatenate(iPatch)).tolist()
                self.deletepatches(iPatch, checkVertices=False, checkSlip=checkSlip)

        # Modify the vertex numbers in Faces
        newFaces = copy.deepcopy(self.Faces)
        for v in iVertices:
            i,j = np.where(self.Faces>v)
            newFaces[i,j] -= 1
        self.Faces = newFaces
         
        # Do the deletion
        self.Vertices = np.delete(self.Vertices, iVertices, axis=0)
        self.Vertices_ll = np.delete(self.Vertices_ll, iVertices, axis=0)
        
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletevertex(self, iVertex, checkPatch=True, checkSlip=True):
        '''
        Delete a Vertex. If some patches are composed of this vertex and 
        checkPatch is True, deletes the patches.

        Args:
            * iVertex       : index of the vertex to delete

        Kwargs:
            * checkPatch    : Check and delete if patches are concerned.
            * checkSlip     : Check and delete is slip is concerned.

        Returns:
            * None
        '''

        # Delete only one vertex
        self.deletevertices([iVertex], checkPatch, checkSlip=checkSlip)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def deletepatches(self, tutu, checkVertices=True, checkSlip=True):
        '''
        Deletes a list of patches.

        Args:
            * tutu      : List of indices

        Kwargs:
            * checkVertices : Check and delete if patches are concerned.
            * checkSlip     : Check and delete is slip is concerned.

        Returns:
            * None
        '''

        while len(tutu)>0:

            # Get index to delete
            i = tutu.pop()

            # delete it
            self.deletepatch(i, checkVertices=checkVertices, checkSlip=checkSlip)

            # Upgrade list
            for u in range(len(tutu)):
                if tutu[u]>i:
                    tutu[u] -= 1

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def refineMesh(self):
        '''
        Cuts all the patches in 4, based on the mid-point of each triangle and 
        builds a new fault from that.

        Returns:
            * None
        '''

        # Iterate over the fault patches
        newpatches = []
        for patch in self.patch:
            triangles = self.splitPatch(patch)
            for triangle in triangles:
                newpatches.append(triangle)

        # Delete all the patches 
        del self.patch
        del self.patchll
        del self.Vertices
        del self.Vertices_ll
        del self.Faces
        self.patch = None
        self.N_slip = None

        # Add the new patches
        self.addpatches(newpatches)

        # Update the depth of the bottom of the fault
        self.top   = np.min(self.Vertices[:,2])
        self.depth = np.max(self.Vertices[:,2])
        self.z_patches = np.linspace(self.depth, 0.0, 5)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def addpatches(self, patches):
        '''
        Adds patches to the list.

        Args:
            * patches     : List of patch geometries

        Returns:
            * None
        '''

        # Iterate
        for patch in patches:
            self.addpatch(patch)

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def addpatch(self, patch, slip=[0, 0, 0]):
        '''
        Adds a patch to the list.

        Args:
            * patch     : Geometry of the patch to add (km, not lon lat)

        Kwargs:
            * slip      : List of the strike, dip and tensile slip.

        Returns:
            * None
        '''

        # Check if the list of patch exists
        if self.patch is None:
            self.patch = []

        # Check that patch is an array
        if type(patch) is list:
            patch = np.array(patch)
        assert type(patch) is np.ndarray, 'addPatch: Patch has to be a numpy array'

        # append the patch
        if type(self.patch) is np.ndarray:
            if self.patch.size==0:
                self.patch = patch[None, :, :]
            else:
                self.patch = np.vstack((self.patch, patch[None, :, :]))
        else:
            self.patch.append(patch.tolist())

        # modify the slip
        # if self.N_slip!=None and self.N_slip==len(self.patch):
        sh = self.slip.shape
        nl = sh[0] + 1
        nc = 3
        tmp = np.zeros((nl, nc))
        if nl > 1:                      # Case where slip is empty
            tmp[:nl-1,:] = self.slip
        tmp[-1,:] = slip
        self.slip = tmp
        self.N_slip = nl

        # Create Vertices and Faces if not there
        if not hasattr(self, 'Vertices'):
            self.Vertices = np.array([patch[0], patch[1], patch[2]])
            self.Faces = np.array([[0, 1, 2]])

        # Check if vertices are already there
        vids = []
        for p in patch:
            ii = np.flatnonzero(np.array([(p.tolist()==v).all() for v in self.Vertices]))
            if len(ii)==0:
                self.Vertices = np.insert(self.Vertices, self.Vertices.shape[0],
                        p, axis=0)
                vids.append(self.Vertices.shape[0]-1)
            else:
                vids.append(ii[0])
        self.Faces = np.insert(self.Faces, self.Faces.shape[0], vids, axis=0)

        # Vertices2ll
        self.patch = np.array(self.patch)
        self.vertices2ll()
        self.patch2ll()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def homogeneizeStrike(self, direction=1, sign=1.):
        '''
        Rotates the vertices in {Faces} so that the normals are all pointing in
        a common direction. The {direction} is checked by the axis (0=x, 1=y, 2=z).

        Kwargs: 
            * direction : Direction of the normal to check. 
            * sign      : +1 or -1
        '''

        # Check
        assert type(sign) in (float, int), 'sign must be a float or int: {}'.format(sign)
        assert float(sign)!=0., 'sign must be different from 0'

        # Compute normals
        normals = np.array([self.getpatchgeometry(p, retNormal=True) for p in self.patch])

        # Find the normals that are not following the rule
        inormals = np.flatnonzero(np.sign(normals[:,direction])==np.sign(sign))

        # For these patches, get the patch, flip 2 summits
        for ip in inormals:
            p1, p2, p3 = self.patch[ip]
            self.patch[ip] = np.array([p1, p3, p2])
            f1, f2, f3 = self.Faces[ip,:]
            self.Faces[ip,0] = f1
            self.Faces[ip,1] = f3
            self.Faces[ip,2] = f2

        # Recompute patches 2 lonlat
        self.patch2ll()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def replacePatch(self, patch, iPatch):
        '''
        Replaces one patch by the given geometry.

        Args:
            * patch     : Patch geometry.
            * iPatch    : index of the patch to replace.

        Returns:    
            * None
        '''

        # Replace
        if type(patch) is list:
            patch = np.array(patch)
        assert type(patch) is np.ndarray, 'replacePatch: Patch must be an array'
        self.patch[iPatch] = patch

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def pointRotation3D(self, iPatch, iPoint, theta, p_axis1, p_axis2):
        '''
        Rotate a point with an arbitrary axis (fault tip)
        Used in rotatePatch
        
        Args:
            * iPatch: index of the patch to be rotated
            * iPoint: index of the patch corner (point) to be rotated
            * theta : angle of rotation in degrees
            * p_axis1 : first point of axis (ex: one side of a fault)
            * p_axis2 : second point to define the axis (ex: the other side of a fault)
            
        Returns:
            * rotated point
        Reference: 'Rotate A Point About An Arbitrary Axis (3D)' - Paul Bourke 
        '''
        def to_radians(angle):
            return np.divide(np.dot(angle, np.pi), 180.0)
    
        def to_degrees(angle):
            return np.divide(np.dot(angle, 180.0), np.pi)
        
        point = self.patch[iPatch][iPoint]
        
        # Translate so axis is at origin    
        p = point - p_axis1
    
        N = p_axis2 - p_axis1
        Nm = np.sqrt(N[0]**2 + N[1]**2 + N[2]**2)
        
        # Rotation axis unit vector
        n = [N[0]/Nm, N[1]/Nm, N[2]/Nm]
    
        # Matrix common factors     
        c = np.cos(to_radians(theta))
        t = 1 - np.cos(to_radians(theta))
        s = np.sin(to_radians(theta))
        X = n[0]
        Y = n[1]
        Z = n[2]
    
        # Matrix 'M'
        d11 = t*X**2 + c
        d12 = t*X*Y - s*Z
        d13 = t*X*Z + s*Y
        d21 = t*X*Y + s*Z
        d22 = t*Y**2 + c
        d23 = t*Y*Z - s*X
        d31 = t*X*Z - s*Y
        d32 = t*Y*Z + s*X
        d33 = t*Z**2 + c
    
        #            |p.x|
        # Matrix 'M'*|p.y|
        #            |p.z|
        q = np.empty((3))
        q[0] = d11*p[0] + d12*p[1] + d13*p[2]
        q[1] = d21*p[0] + d22*p[1] + d23*p[2]
        q[2]= d31*p[0] + d32*p[1] + d33*p[2]
        
        # Translate axis and rotated point back to original location    
        return np.array(q + p_axis1)
    # ----------------------------------------------------------------------
        
    # ----------------------------------------------------------------------
    def rotatePatch(self, iPatch , theta, p_axis1, p_axis2, verbose=False):
        '''
        Rotate a patch with an arbitrary axis (fault tip)
        Used by fault class uncertainties
        
        Args:
            * iPatch: index of the patch to be rotated
            * theta : angle of rotation in degrees
            * p_axis1 : first point of axis (ex: one side of a fault)
            * p_axis2 : second point to define the axis (ex: the other side of a fault)
            
        Returns:
            * rotated patch
        '''
        if verbose:
            print('Rotating patch {} '.format(iPatch))
        
        # Calculate rotated patch
        rotated_patch = [self.pointRotation3D(iPatch,0, theta, p_axis1, p_axis2),
                         self.pointRotation3D(iPatch,1, theta, p_axis1, p_axis2),
                         self.pointRotation3D(iPatch,2, theta, p_axis1, p_axis2)]
        
        patch = rotated_patch
        
        # Replace
        self.patch[iPatch] = np.array(patch)

        # Build the ll patch
        lon1, lat1 = self.xy2ll(patch[0][0], patch[0][1])
        z1 = patch[0][2]
        lon2, lat2 = self.xy2ll(patch[1][0], patch[1][1])
        z2 = patch[1][2]
        lon3, lat3 = self.xy2ll(patch[2][0], patch[2][1])
        z3 = patch[2][2]

        # append the ll patch
        patchll = [ [lon1, lat1, z1],
                    [lon2, lat2, z2],
                    [lon3, lat3, z3] ]

        # Replace
        self.patchll[iPatch] = np.array(patchll)
        return 
    # ----------------------------------------------------------------------    

    # ----------------------------------------------------------------------
    # Translate a patch
    def translatePatch(self, iPatch , tr_vector):
        '''
        Translate a patch
        Used by class uncertainties
        
        Args:
            * iPatch: index of the patch to be rotated
            * tr_vector: array, translation vector in 3D
            
        Returns:
            * None
        '''        
        # Calculate rotated patch
        tr_p1 = np.array( [ self.patch[iPatch][0][0]+tr_vector[0], 
                          self.patch[iPatch][0][1]+tr_vector[1], 
                          self.patch[iPatch][0][2]+tr_vector[2]])
        tr_p2 = np.array( [self.patch[iPatch][1][0]+tr_vector[0], 
                          self.patch[iPatch][1][1]+tr_vector[1], 
                          self.patch[iPatch][1][2]+tr_vector[2]])
        tr_p3 = np.array( [self.patch[iPatch][2][0]+tr_vector[0], 
                          self.patch[iPatch][2][1]+tr_vector[1], 
                          self.patch[iPatch][2][2]+tr_vector[2]])
        
        tr_patch=[tr_p1, tr_p2, tr_p3]
                                             
        # Replace
        self.patch[iPatch] = tr_patch

        # Build the ll patch
        lon1, lat1 = self.xy2ll(tr_patch[0][0], tr_patch[0][1])
        z1 = tr_patch[0][2]
        lon2, lat2 = self.xy2ll(tr_patch[1][0], tr_patch[1][1])
        z2 = tr_patch[1][2]
        lon3, lat3 = self.xy2ll(tr_patch[2][0], tr_patch[2][1])
        z3 = tr_patch[2][2]

        # append the ll patch
        patchll = [ [lon1, lat1, z1],
                    [lon2, lat2, z2],
                    [lon3, lat3, z3] ]

        # Replace
        self.patchll[iPatch] = np.array(patchll)
        return 
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getpatchgeometry(self, patch, center=False, retNormal=False, checkindex=True):
        '''
        Returns the patch geometry as needed for triangleDisp.

        Args:
            * patch         : index of the wanted patch or patch

        Kwargs:
            * center        : if true, returns the coordinates of the center of the patch. if False, returns the first corner
            * checkindex    : Checks the index of the patch
            * retNormal     : If True gives, also the normal vector to the patch

        Returns:
            * x, y, z, width, length, strike, dip, (normal)
        '''

        # Get the patch
        u = None
        if type(patch) in (int, np.int64, np.int32):
            u = patch
        else:
            if checkindex:
                u = self.getindex(patch)
        if u is not None:
            patch = self.patch[u]

        # Get the center of the patch
        x1, x2, x3 = self.getcenter(patch)

        # Get the vertices of the patch (indexes are flipped to get depth along z axis)
        verts = copy.deepcopy(patch)
        p1, p2, p3 = [np.array([lst[1],lst[0],lst[2]]) for lst in verts]

        # Get a dummy width and height
        width = np.linalg.norm(p1 - p2)
        length = np.linalg.norm(p3 - p1)

        # Get the patch normal
        normal = np.cross(p2 - p1, p3 - p1)

        # If fault is vertical, force normal to be horizontal
        if self.vertical:
            normal[2] = 0.

        # Normalize
        normal /= np.linalg.norm(normal)

        # Enforce clockwise circulation
        if np.round(normal[2],decimals=2) < 0.:
            normal *= -1.0
            p2, p3 = p3, p2

        # Force strike between 0 and 90 or between 270 and 360
        #if normal[1] > 0:
        #     normal *= -1

        # Get the strike vector and strike angle
        strike = np.arctan2(-normal[0], normal[1]) - np.pi
        if strike<0.:
            strike += 2*np.pi

        # Set the dip vector
        dip = np.arccos(normal[2])

        if retNormal:
            return x1, x2, x3, width, length, strike, dip, normal
        else:
            return x1, x2, x3, width, length, strike, dip
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distanceVertexToVertex(self, vertex1, vertex2, distance='center', lim=None):
        '''
        Measures the distance between two vertexes.

        Args:
            * vertex1   : first patch or its index
            * vertex2   : second patch or its index

        Kwargs:
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].
            * distance  : Useless argument only here for compatibility reasons

        Returns:
            * distance  : float
        '''

        if distance == 'center':

            # Get the centers
            x1, y1, z1 = vertex1
            x2, y2, z2 = vertex2

            # Compute the distance
            dis = np.sqrt((x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis > lim[0]:
                    dis = lim[1]

        else:
            raise NotImplementedError('only distance=center is implemented')

        # All done
        return dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distanceMatrix(self, distance='center', lim=None):
        '''
        Returns a matrix of the distances between patches.

        Kwargs:
            * distance  : distance estimation mode

                 - center : distance between the centers of the patches.
                 - no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].

        Returns:
            * distances : Array of floats
        '''

        # Assert 
        assert distance == 'center', 'No other method implemented than center'

        # Check
        if self.N_slip is None:
            self.N_slip = self.slip.shape[0]

        # Loop
        Distances = np.zeros((self.N_slip, self.N_slip))
        for i in range(self.N_slip):
            p1 = self.patch[i]
            for j in range(self.N_slip):
                if j == i:
                    continue
                p2 = self.patch[j]
                Distances[i,j] = self.distancePatchToPatch(p1, p2, distance='center', lim=lim)

        # All done
        return Distances
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distancesMatrix(self, distance='center', lim=None):
        '''
        Returns two matrices of the distances between patches.
        One for the horizontal dimensions, the other for verticals

        Kwargs:
            * distance  : distance estimation mode

                 - center : distance between the centers of the patches.
                 - no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].

        Returns:
            * distances : Array of floats
        '''

        # Assert 
        assert distance=='center', 'No other method implemented than center'

        # Check
        if self.N_slip==None:
            self.N_slip = self.slip.shape[0]

        # Loop
        HDistances = np.zeros((self.N_slip, self.N_slip))
        VDistances = np.zeros((self.N_slip, self.N_slip))
        for i in range(self.N_slip):
            p1 = self.patch[i]
            c1 = self.getcenter(p1)
            for j in range(self.N_slip):
                if j == i:
                    continue
                p2 = self.patch[j]
                c2 = self.getcenter(p2)
                HDistances[i,j] = np.sqrt( (c1[0]-c2[0])**2 + (c1[1]-c2[1])**2 )
                HDistances[j,i] = HDistances[i,j]
                VDistances[i,j] = np.sqrt( (c1[2]-c2[2])**2 )
                VDistances[j,i] = VDistances[i,j]

        # All done
        return HDistances, VDistances
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def distancePatchToPatch(self, patch1, patch2, distance='center', lim=None):
        '''
        Measures the distance between two patches.

        Args:
            * patch1    : first patch or its index
            * patch2    : second patch or its index

        Kwargs:
            * distance  : distance estimation mode

                    - center : distance between the centers of the patches.
                    - no other method is implemented for now.
            * lim       : if not None, list of two float, the first one is the distance above which d=lim[1].

        Returns:
            * distace   : float
        '''

        if distance == 'center':

            # Get the centers
            x1, y1, z1 = self.getcenter(patch1)
            x2, y2, z2 = self.getcenter(patch2)

            # Compute the distance
            dis = np.sqrt((x1 -x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)

            # Check
            if lim is not None:
                if dis > lim[0]:
                    dis = lim[1]

        else:
            raise NotImplementedError('only distance=center is implemented')

        # All done
        return dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def slip2dis(self, data, patch, slip=None):
        '''
        Computes the surface displacement for a given patch at the data location
        using a homogeneous half-space.

        Args:
            * data          : data object from gps or insar.
            * patch         : number of the patch that slips

        Kwargs:
            * slip          : if a number is given, that is the amount of slip along strike. If three numbers are given, that is the amount of slip along strike, along dip and opening. if None, values from self.slip are taken.

        Returns:
            * ss_dis        : Surface displacements due to strike slip
            * ds_dis        : Surface displacements due to dip slip
            * ts_dis        : Surface displacements due to tensile opening
        '''

        # Set the slip values
        if slip is None:
            SLP = [self.slip[patch,0], self.slip[patch,1], self.slip[patch,2]]
        elif slip.__class__ is float:
            SLP = [slip, 0.0, 0.0]
        elif slip.__class__ is list:
            SLP = slip

        # Get patch vertices
        vertices = list(self.patch[patch])

        # Get data position
        x = data.x
        y = data.y
        z = np.zeros_like(x)

        # Get strike slip displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, SLP[0], 0.0, 0.0)
        ss_dis = np.column_stack((ux, uy, uz))

        # Get dip slip displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, 0.0, SLP[1], 0.0)
        ds_dis = np.column_stack((ux, uy, uz))

        # Get opening displacements
        ux, uy, uz = tdisp.displacement(x, y, z, vertices, 0.0, 0.0, SLP[2])
        op_dis = np.column_stack((ux, uy, uz))

        # All done
        return ss_dis, ds_dis, op_dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildAdjacencyMap(self, verbose=True):
        '''
        For each triangle, find the indices of the adjacent (edgewise) triangles.

        Kwargs:
            * verbose

        Returns:
            * None
        '''

        from .edge_utils.mesh_edge_finder  import find_adjacent_triangles

        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the adjacency map for all patches")

        # cache the  vertex indices from the faces
        vertex_indices = self.Faces.astype(np.int_)
        # build the adjacency map
        self.adjacencyMap = find_adjacent_triangles(vertex_indices)
        # 将不规则列表转换为固定长度的数组
        self.adjacencyMap_array = np.array([x + [-1]*(3-len(x)) for x in self.adjacencyMap], dtype=np.int_)
        if verbose:
            print('')
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildLaplacian_csi(self, verbose=True, method=None, irregular=False):
        '''
        Build a discrete Laplacian smoothing matrix.

        Kwargs:
            * verbose       : Speak to me
            * method        : Not used, here for consistency purposes
            * irregular     : Not used, here for consistency purposes
        
        Returns:
            * Laplacian     : 2D array
        '''
        
        if self.adjacencyMap is None or len(self.adjacencyMap) != len(self.patch):
            self.buildAdjacencyMap(verbose=verbose)

        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the Laplacian matrix based on CSI")

        # Pre-compute patch centers
        centers = self.getcenters()

        # Cache the vertices and faces arrays
        vertices, faces = self.Vertices, self.Faces

        # Allocate array for Laplace operator
        npatch = len(self.patch)
        D = np.zeros((npatch,npatch))

        # Loop over patches
        for i in range(npatch):

            if verbose:
                sys.stdout.write('%i / %i\r' % (i, npatch))
                sys.stdout.flush()

            # Center for current patch
            refCenter = np.array(centers[i])

            # Compute Laplacian using adjacent triangles
            hvals = []
            adjacents = self.adjacencyMap[i]
            for index in adjacents:
                pcenter = np.array(centers[index])
                dist = np.linalg.norm(pcenter - refCenter)
                hvals.append(dist)
            if len(hvals) == 3:
                h12, h13, h14 = hvals
                D[i,adjacents[0]] = -h13*h14
                D[i,adjacents[1]] = -h12*h14
                D[i,adjacents[2]] = -h12*h13
                sumProd = h13*h14 + h12*h14 + h12*h13
            elif len(hvals) == 2:
                h12, h13 = hvals
                # Make a virtual patch
                h14 = max(h12, h13)
                D[i,adjacents[0]] = -h13*h14
                D[i,adjacents[1]] = -h12*h14
                sumProd = h13*h14 + h12*h14 + h12*h13
            # Added by kfhe, at 10/16/2021 to avoiding the case where sumProd have been not defined
            elif len(hvals) == 1:
                h12, = hvals
                h13, h14 = h12, h12
                D[i, adjacents[0]] = [-h13*h14]
                sumProd = h13*h14 + h12*h14 + h12*h13
            D[i,i] = sumProd

        if verbose:
            print('')
        D = D / np.max(np.abs(np.diag(D)))
        return D
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def find_boundary_and_corner_triangles(self, top_tolerance: float, bottom_tolerance: float):
        '''
        Find the triangles on the top, bottom, left and right boundaries of a mesh, as well as the corner triangles.
        Left: In North, Right: In South if the left/right can not be determined from west/east

        Args:
            vertex_indices: A numpy array of shape (n, 3) containing the vertex indices for each triangle in the mesh.
            vertex_coordinates: A numpy array of shape (m, 3) containing the coordinates of each vertex in the mesh.
            top_tolerance: A float specifying the tolerance for determining the top boundary of the mesh.
            bottom_tolerance: A float specifying the tolerance for determining the bottom boundary of the mesh.
            remove_corner: A boolean specifying whether to remove the corner triangles from the boundary triangles.
        '''
        from .edge_utils.mesh_edge_finder import find_boundary_and_corner_triangles

        # Cache the vertices and faces arrays
        vertex_indices = self.Faces.astype(np.int_)
        vertex_coordinates = self.Vertices
        # Find the boundary and corner triangles
        boundary_triangles, corner_triangles = find_boundary_and_corner_triangles(vertex_indices, vertex_coordinates, top_tolerance, bottom_tolerance)

        self.edge_dict = {'top': boundary_triangles['top'],
                            'bottom': boundary_triangles['bottom'],
                            'left': boundary_triangles['left'],
                            'right': boundary_triangles['right']
                            }
        
        self.corner_dict = {'left_top': corner_triangles['top_left'],
                            'right_top': corner_triangles['top_right'],
                            'left_bottom': corner_triangles['bottom_left'],
                            'right_bottom': corner_triangles['bottom_right']
                            }

        # All done
        return

    def find_fault_edge_vertices(self, top_tolerance=0.1, bottom_tolerance=0.1, refind=False):
        '''
        Left: In West, Right: In East
        '''
        import copy
        # find boundary and corner triangles indexes in fault.Faces
        if not hasattr(self, 'edge_dict') or refind:
            self.find_boundary_and_corner_triangles(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance)
        edge, corner = self.edge_dict,  self.corner_dict

        def get_edge_and_inds(edge, edge_key, corner_key1, corner_key2):
            # Get edge triangle indexes with corner triangle indexes added if exist
            edge = copy.deepcopy(edge[edge_key])
            if corner[corner_key1] is not None:
                edge.append(corner[corner_key1])
            if corner[corner_key2] is not None:
                edge.append(corner[corner_key2])
            # Get the unique edge triangle indexes in all trianles. edge is index of self.Faces
            edge = np.unique(edge)
            # Get the unique points indexes in edge triangles, pnt_inds is index of self.Vertices
            pnts = self.Faces[edge]
            pnt_inds = np.unique(np.sort(pnts.flatten()))
            return edge, pnt_inds

        right_edge, rpnt_inds = get_edge_and_inds(edge, 'right', 'right_top', 'right_bottom')
        left_edge, lpnt_inds = get_edge_and_inds(edge, 'left', 'left_top', 'left_bottom')
        top_edge, tpnt_inds = get_edge_and_inds(edge, 'top', 'left_top', 'right_top')
        bottom_edge, bpnt_inds = get_edge_and_inds(edge, 'bottom', 'left_bottom', 'right_bottom')

        self.edge_triangles_indices = {
            'left': left_edge, 
            'right': right_edge, 
            'top': top_edge, 
            'bottom': bottom_edge}
        self.edge_triangle_vertex_indices = {
            'left': lpnt_inds, 
            'right': rpnt_inds, 
            'top': tpnt_inds, 
            'bottom': bpnt_inds}
        # All Done
        return
    
    def find_fault_fouredge_vertices(self, top_tolerance=0.1, bottom_tolerance=0.1, 
                                     refind=False, method='hybrid', merge_threshold=0.02):
        from .edge_utils.mesh_edge_finder import find_left_or_right_edgeline_points
        if not hasattr(self, 'edge_triangles_indices') or refind:
            self.find_fault_edge_vertices(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, refind=refind)

        left_inds, left_pnts = find_left_or_right_edgeline_points(self.edge_triangles_indices['left'], self.Faces, self.Vertices, side='left')
        right_inds, right_pnts = find_left_or_right_edgeline_points(self.edge_triangles_indices['right'], self.Faces, self.Vertices, side='right')
    
        top_inds = self.edge_triangle_vertex_indices['top']
        bottom_inds = self.edge_triangle_vertex_indices['bottom']
        top_pnts = self.Vertices[top_inds]
        bottom_pnts = self.Vertices[bottom_inds]
        top_depth = np.min(top_pnts[:, -1])
        bottom_depth = np.max(bottom_pnts[:, -1])
    
        top_flag = np.where(np.abs(top_pnts[:, -1] - top_depth) <= top_tolerance)[0]
        top_inds = top_inds[top_flag]
        top_pnts = top_pnts[top_flag]
        top_sortinds = np.argsort(top_pnts[:, 0])
        top_inds = top_inds[top_sortinds]
        top_pnts = top_pnts[top_sortinds]
    
        bottom_flag = np.where(np.abs(bottom_pnts[:, -1] - bottom_depth) <= bottom_tolerance)[0]
        bottom_inds = bottom_inds[bottom_flag]
        bottom_pnts = bottom_pnts[bottom_flag]
        bottom_sortinds = np.argsort(bottom_pnts[:, 0])
        bottom_inds = bottom_inds[bottom_sortinds]
        bottom_pnts = bottom_pnts[bottom_sortinds]
        
        self.edge_vertex_indices = {
            'top': top_inds,
            'bottom': bottom_inds,
            'left': left_inds, 
            'right': right_inds
        }
        self.edge_vertices = {
            'top': top_pnts,
            'bottom': bottom_pnts,
            'left': left_pnts, 
            'right': right_pnts
        }

        top_edge, top_inds = self.find_ordered_edge_vertices('top', top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, return_indices=True,
                                                              method=method, merge_threshold=merge_threshold)
        bottom_edge, bottom_inds = self.find_ordered_edge_vertices('bottom', top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, return_indices=True,
                                                              method=method, merge_threshold=merge_threshold)

        self.edge_vertex_indices['top'] = top_inds
        self.edge_vertex_indices['bottom'] = bottom_inds
        self.edge_vertices['top'] = top_edge
        self.edge_vertices['bottom'] = bottom_edge
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildLaplacian_mudpy(self, verbose=True, method=None, irregular=False, bounds=None, corner=True, topscale=0.01, bottomscale=0.01):
        '''
        Build a discrete Laplacian smoothing matrix.

        Kwargs:
            * verbose       : Speak to me
            * method        : Not used, here for consistency purposes
            * irregular     : Not used, here for consistency purposes
            * corner        : Not consider the case of corner if False
            * bounds        : ['free'/'locked',]*4, default: ['free',]*4
                              for ['top', 'bottom', 'left', 'right'], repestively
        
        Returns:
            * Laplacian     : 2D array
        
        Reference:
            * Jiang et al., 2013, GJI
            * Wang et al., 2017, RS
            * Melgar et al., 2014, PhD
            * Zhou et al., 2014, GJI
            * Kefeng He et al., 2022, SRL
        Comments: 
            * Added by kfhe at 10/16/2021
        '''
        # Find the boundary and corner triangles if not found
        if not hasattr(self, 'edge_dict') or self.edge_dict is None or self.corner_dict is None:
            self.find_boundary_and_corner_triangles(top_tolerance=topscale, bottom_tolerance=bottomscale)
        
        # Build the adjacency map if not built
        if self.adjacencyMap is None or len(self.adjacencyMap) != len(self.patch):
            self.buildAdjacencyMap(verbose=verbose)
            self.find_boundary_and_corner_triangles(top_tolerance=topscale, bottom_tolerance=bottomscale)
        
        edge_dict = self.edge_dict
        corner_dict = self.corner_dict
        
        if bounds is None:
            bounds = ['free',]*4

        # saved format is [top, bottom, left, right] with True/False
        bounds_mark = np.array([bound.lower() == 'free' for bound in bounds])
        # saved format is list with edge index which is free
        free_edge_tris = [edge for bound, flag in zip(bounds, ['top', 'bottom', 'left', 'right']) if bound.lower() == 'free' for edge in edge_dict[flag]]
        # saved format is {corner_index: corner_type}
        free_corner_dict = {corner_dict[flag]: bounds_mark[idx].sum() for flag, idx in zip(['left_top', 'right_top', 'left_bottom', 'right_bottom'], [[0, 2], [0, 3], [1, 2], [1, 3]]) if corner_dict[flag] is not None}

        if verbose:
            print("------------------------------------------")
            print("------------------------------------------")
            print("Building the Laplacian matrix based on Mudpy")

        # Pre-compute patch centers
        centers = np.array(self.getcenters())
        # 预计算所有可能需要的值
        npatch = len(self.patch)
        hvals_all = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=-1)
        free_edge_tris_set = set(free_edge_tris)
        free_corner_dict_keys_set = set(free_corner_dict.keys())

        # 初始化D
        D = np.zeros((npatch,npatch))

        # 对于每个patch，计算其邻接patch的数量
        adjacents_counts = np.array([len(adjacents) for adjacents in self.adjacencyMap])

        # 对于邻接patch数量为3的patch，使用向量化操作进行计算
        mask = adjacents_counts == 3
        adjacents = self.adjacencyMap_array[mask]
        hvals = np.take_along_axis(hvals_all[mask], adjacents, axis=1)
        h12, h13, h14 = hvals.T
        # 创建一个广播的索引数组
        rows = np.arange(D.shape[0])[:, None]
        # 使用广播的索引数组来赋值
        D[rows[mask], adjacents] = -np.array([h13*h14, h12*h14, h12*h13]).T
        sumProd = h13*h14 + h12*h14 + h12*h13
        D[mask, mask] = sumProd

        # 对于邻接patch数量为2的patch，使用向量化操作进行计算
        mask = adjacents_counts == 2
        adjacents = self.adjacencyMap_array[mask][:, :2]
        hvals = np.take_along_axis(hvals_all[mask], adjacents, axis=1)
        h12, h13 = hvals.T
        h14 = np.maximum(h12, h13)
        sumProd = h13*h14 + h12*h14 + h12*h13
        scale = np.where(np.isin(mask.nonzero()[0], list(free_edge_tris_set)), sumProd/(h13*h14 + h12*h14), 1.0)
        # 创建一个广播的索引数组
        rows = np.arange(D.shape[0])[:, None]
        # 使用广播的索引数组来赋值
        D[rows[mask], adjacents] = (-np.array([h13*h14, h12*h14])*scale).T
        D[mask, mask] = sumProd

        # 对于邻接patch数量为1的patch，使用向量化操作进行计算
        mask = adjacents_counts == 1
        adjacents = self.adjacencyMap_array[mask][:, :1]
        hvals = np.take_along_axis(hvals_all[mask], adjacents, axis=1)
        h12 = hvals[:, 0]
        h13 = h14 = h12
        sumProd = h13*h14 + h12*h14 + h12*h13
        scale_cor = np.array([free_corner_dict[i] for i in mask.nonzero()[0] if i in free_corner_dict_keys_set])
        scale = np.where(scale_cor > 0, np.where(scale_cor == 1, sumProd/(h13*h14) * 2./3., sumProd/(h13*h14)), 1.0)
        # 创建一个广播的索引数组
        rows = np.arange(D.shape[0])[:, None]
        # 使用广播的索引数组来赋值
        D[rows[mask], adjacents] = (-h13*h14*scale)[:, np.newaxis]
        D[mask, mask] = sumProd

        D = D / np.max(np.abs(np.diag(D)))
        if verbose:
            print('')
        return D
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def buildLaplacian(self, verbose=False, method=None, irregular=False, bounds=None, corner=True, topscale=0.01, bottomscale=0.01):
        '''
        Build normalized Laplacian smoothing array.
        This routine is not designed for unevenly paved faults.
        It does not account for the variations in size of the patches.

        Kwargs:
            * verbose       : speak to me
            * method        : Useless argument only here for compatibility reason
            * irregular     : Not used, here for consistency purposes
        '''
        import sys
        if method in ('csi', 'CSI', 'Csi', None):
            D = self.buildLaplacian_csi(verbose=verbose, irregular=irregular)
        elif method in ('Mudpy', 'MUDPY', 'mudpy', 'mud', 'Diego'):
            if bounds is None:
                print('Bounds must be given for the method of Mudpy!!!')
                sys.exit(1)
            D = self.buildLaplacian_mudpy(verbose=verbose, irregular=irregular, bounds=bounds, corner=corner, topscale=topscale, bottomscale=bottomscale)
        return D
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def compute_slip_gradient(self, slip='total'):
        """
        Compute the gradient of slip values for each subfault.
        
        Parameters:
        slip (str): Type of slip to compute gradient for ('total', 'strikeslip', 'dipslip').
        
        Returns:
        np.ndarray: Gradient of slip values for each subfault.
        """
        from scipy.spatial import KDTree
    
        # Select the appropriate slip values
        if slip == 'total':
            slip_values = np.sqrt(self.slip[:, 0]**2 + self.slip[:, 1]**2 + self.slip[:, 2]**2)
        elif slip == 'strikeslip':
            slip_values = self.slip[:, 0]
        elif slip == 'dipslip':
            slip_values = self.slip[:, 1]
        else:
            raise ValueError("Invalid slip type. Choose from 'total', 'strikeslip', or 'dipslip'.")
    
        coordinates = np.mean(self.patch, axis=1)
        tree = KDTree(coordinates)
        gradients = np.zeros_like(slip_values)
        
        # Get the four edges of the fault
        if not hasattr(self, 'edge_triangles_indices'):
            self.find_fault_edge_vertices()
        edge_indices = self.edge_triangles_indices
        edge_indices = np.empty(0, dtype=int)
        for key in ['top', 'bottom', 'left', 'right']:
            if key in self.edge_triangles_indices:
                edge_indices = np.append(edge_indices, self.edge_triangles_indices[key])
        edge_indices = np.unique(edge_indices)

        for i in range(len(slip_values)):
            if i in edge_indices:
                k = 3  # Only consider two neighbors
            else:
                k = 4  # Consider three neighbors
            
            distances, indices = tree.query(coordinates[i], k=k)  # k includes the point itself
            indices = indices[1:]  # Exclude the point itself
            
            # Calculate the gradient as the mean of the absolute differences
            gradients[i] = np.mean(np.abs(slip_values[i] - slip_values[indices]) / distances[1:])
        
        return gradients
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getcenter(self, p):
        '''
        Get the center of one triangular patch.

        Args:
            * p     : Patch geometry.

        Returns:
            * x,y,z : floats 
        '''

        # Get center
        if type(p) is int:
            p1, p2, p3 = self.patch[p]
        else:
            p1, p2, p3 = p

        # Compute the center
        x = (p1[0] + p2[0] + p3[0]) / 3.0
        y = (p1[1] + p2[1] + p3[1]) / 3.0
        z = (p1[2] + p2[2] + p3[2]) / 3.0

        # All done
        return x,y,z
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def computetotalslip(self):
        '''
        Computes the total slip and stores it self.totalslip
        '''

        # Computes the total slip
        self.totalslip = np.sqrt(self.slip[:,0]**2 + self.slip[:,1]**2 \
                + self.slip[:,2]**2)

        # All done
        return
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def getcenters(self, coordinates='xy'):
        '''
        Get the center of the patches.

        Kwargs:
            * coordinates   : 'xy' or 'll'

        Returns:
            * centers:  list of triplets
        '''

        assert coordinates in ('xy', 'll'), 'coordinates must be xy or ll: currently set to {}'.format('xy')

        # Vectorized operation to get centers
        centers = np.array([self.getcenter(p) for p in self.patch])

        # All done
        if coordinates == 'xy':
            return centers
        else:
            lon, lat = self.xy2ll(centers[:, 0], centers[:, 1])
            return np.column_stack([lon, lat, centers[:, 2]])
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def surfacesimulation(self, box=None, disk=None, err=None, npoints=10, 
                          lonlat=None, slipVec=None):
        '''
        Takes the slip vector and computes the surface displacement that 
        corresponds on a regular grid.

        Kwargs:
            * box       : A list of [minlon, maxlon, minlat, maxlat].
            * disk      : list of [xcenter, ycenter, radius, n]
            * lonlat    : Arrays of lat and lon. [lon, lat]
            * err       : Compute random errors and scale them by {err}
            * slipVec   : Replace slip by what is in slipVec
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # create a fake gps object
        self.sim = gpsclass('simulation', utmzone=self.utmzone, lon0=self.lon0, lat0=self.lat0)

        # Create a lon lat grid
        if lonlat is None:
            if (box is None) and (disk is None) :
                lon = np.linspace(self.lon.min(), self.lon.max(), npoints)
                lat = np.linspace(self.lat.min(), self.lat.max(), npoints)
                lon, lat = np.meshgrid(lon,lat)
                lon = lon.flatten()
                lat = lat.flatten()
            elif (box is not None):
                lon = np.linspace(box[0], box[1], npoints)
                lat = np.linspace(box[2], box[3], npoints)
                lon, lat = np.meshgrid(lon,lat)
                lon = lon.flatten()
                lat = lat.flatten()
            elif (disk is not None):
                lon = []; lat = []
                xd, yd = self.ll2xy(disk[0], disk[1])
                xmin = xd-disk[2]; xmax = xd+disk[2]; ymin = yd-disk[2]; ymax = yd+disk[2]
                ampx = (xmax-xmin)
                ampy = (ymax-ymin)
                n = 0
                while n<disk[3]:
                    x, y = np.random.rand(2)
                    x *= ampx; x -= ampx/2.; x += xd
                    y *= ampy; y -= ampy/2.; y += yd
                    if ((x-xd)**2 + (y-yd)**2) <= (disk[2]**2):
                        lo, la = self.xy2ll(x,y)
                        lon.append(lo); lat.append(la)
                        n += 1
                lon = np.array(lon); lat = np.array(lat)
        else:
            lon = np.array(lonlat[0])
            lat = np.array(lonlat[1])

        # Clean it
        if (lon.max()>360.) or (lon.min()<-180.0) or (lat.max()>90.) or (lat.min()<-90):
            self.sim.x = lon
            self.sim.y = lat
            lon, lat = self.sim.xy2ll(lon, lat)
            self.sim.lon, self.sim.lat = lon, lat
        else:
            self.sim.lon = lon
            self.sim.lat = lat
            # put these in x y utm coordinates
            self.sim.x, self.sim.y = self.sim.ll2xy(lon ,lat)

        # Initialize the vel_enu array
        self.sim.vel_enu = np.zeros((lon.size, 3))

        # Create the station name array
        self.sim.station = []
        for i in range(len(self.sim.x)):
            name = '{:04d}'.format(i)
            self.sim.station.append(name)
        self.sim.station = np.array(self.sim.station)

        # Create an error array
        if err is not None:
            self.sim.err_enu = []
            for i in range(len(self.sim.x)):
                x,y,z = np.random.rand(3)
                x *= err
                y *= err
                z *= err
                self.sim.err_enu.append([x,y,z])
            self.sim.err_enu = np.array(self.sim.err_enu)

        # import stuff
        import sys

        # Load the slip values if provided
        if slipVec is not None:
            nPatches = len(self.patch)
            print(nPatches, slipVec.shape)
            assert slipVec.shape == (nPatches,3), 'mismatch in shape for input slip vector'
            self.slip = slipVec

        # Loop over the patches
        for p in range(len(self.patch)):
            sys.stdout.write('\r Patch {} / {} '.format(p+1,len(self.patch)))
            sys.stdout.flush()
            # Get the surface displacement due to the slip on this patch
            ss, ds, op = self.slip2dis(self.sim, p)
            # Sum these to get the synthetics
            self.sim.vel_enu += ss
            self.sim.vel_enu += ds
            self.sim.vel_enu += op

        sys.stdout.write('\n')
        sys.stdout.flush()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def compute_surface_displacement(
        self, box=None, disk=None, npoints=10, lonlat=None, profile=None,
        slipVec=None, nu=0.25, data=None, method='cutde', **kwargs
    ):
        """
        Compute surface displacement for triangular fault using selected method ('cutde' or 'edcmp').
        Supports arbitrary observation points (box, disk, lonlat, profile, or data object) and user-defined slip.
    
        Parameters
        ----------
        box : list, optional
            [minlon, maxlon, minlat, maxlat], grid sampling.
        disk : list, optional
            [lon_center, lat_center, radius_km, n], random sampling in disk.
        npoints : int, optional
            Number of points per axis (for grid).
        lonlat : tuple of arrays, optional
            (lon, lat) arrays for custom sampling.
        profile : dict, optional
            {'start': (lon1, lat1), 'end': (lon2, lat2), 'n': N}, sample along profile.
        slipVec : np.ndarray, optional
            (n_patch, 3) slip for each patch [strike, dip, tensile].
        nu : float, optional
            Poisson's ratio.
        data : object, optional
            Custom data object with .x, .y, .z attributes (z optional, default 0).
        method : str, optional
            'cutde' or 'edcmp', selects calculation backend.
    
        Returns
        -------
        obs_pts : np.ndarray
            Observation points (N, 3).
        disp_total : np.ndarray
            Surface displacement at each observation point (N, 3).
        """
    
        obs_pts = self._prepare_observation_points(
            box=box, disk=disk, npoints=npoints, lonlat=lonlat, profile=profile, data=data
        )
    
        # Prepare slip vector
        if slipVec is not None:
            slips = slipVec
        else:
            slips = self.slip
    
        # Prepare source triangles
        src_tris = self.Vertices[self.Faces, :]
        src_tris = np.copy(src_tris)
        src_tris[:, :, -1] *= -1  # cutde requires z-up
    
        if method == 'cutde':
            from cutde.halfspace import disp, disp_free, disp_matrix
            if obs_pts.shape[0] < 10000:
                disp_total = disp_free(obs_pts, src_tris, slips, nu)
            else:
                # The output disp_mat is a (N_OBS_PTS, 3, N_SRC_TRIS, 3) array. 
                disp_mat = disp_matrix(obs_pts, src_tris, nu)
                disp_total = np.einsum('ijkl,kl->ij', disp_mat, slips)
        elif method == 'edcmp':
            # You can implement edcmp backend here, e.g.:
            disp_total = self._edcmp_surface_displacement(obs_pts, src_tris, slips, nu, **kwargs)
        else:
            raise ValueError(f"Unknown method: {method}")
    
        return obs_pts, disp_total
    
    def _prepare_observation_points(self, box=None, disk=None, npoints=10, lonlat=None, profile=None, data=None):
        """
        Prepare observation points for surface displacement calculation.
        Returns (N, 3) array.
        """
        import numpy as np
        if data is not None and hasattr(data, 'x') and hasattr(data, 'y'):
            x = np.asarray(data.x)
            y = np.asarray(data.y)
            if hasattr(data, 'z'):
                z = np.asarray(data.z)
            else:
                z = np.zeros_like(x)
            obs_pts = np.column_stack([x, y, z])
        elif lonlat is not None:
            lon = np.array(lonlat[0])
            lat = np.array(lonlat[1])
            x, y = self.ll2xy(lon, lat)
            obs_pts = np.column_stack([x, y, np.zeros_like(x)])
        elif box is not None:
            lon = np.linspace(box[0], box[1], npoints)
            lat = np.linspace(box[2], box[3], npoints)
            lon, lat = np.meshgrid(lon, lat)
            lon = lon.flatten()
            lat = lat.flatten()
            x, y = self.ll2xy(lon, lat)
            obs_pts = np.column_stack([x, y, np.zeros_like(x)])
        elif disk is not None:
            lon, lat = [], []
            from random import uniform
            xc, yc = disk[0], disk[1]
            r = disk[2]
            n = disk[3]
            while len(lon) < n:
                theta = uniform(0, 2*np.pi)
                rad = r * np.sqrt(uniform(0, 1))
                dx = rad * np.cos(theta)
                dy = rad * np.sin(theta)
                x, y = self.ll2xy(xc, yc)
                x_new, y_new = x + dx, y + dy
                lon_new, lat_new = self.xy2ll(x_new, y_new)
                lon.append(lon_new)
                lat.append(lat_new)
            lon = np.array(lon)
            lat = np.array(lat)
            x, y = self.ll2xy(lon, lat)
            obs_pts = np.column_stack([x, y, np.zeros_like(x)])
        elif profile is not None:
            lon1, lat1 = profile['start']
            lon2, lat2 = profile['end']
            n = profile['n']
            lon = np.linspace(lon1, lon2, n)
            lat = np.linspace(lat1, lat2, n)
            x, y = self.ll2xy(lon, lat)
            obs_pts = np.column_stack([x, y, np.zeros_like(x)])
        else:
            lon = np.linspace(self.lon.min(), self.lon.max(), npoints)
            lat = np.linspace(self.lat.min(), self.lat.max(), npoints)
            lon, lat = np.meshgrid(lon, lat)
            lon = lon.flatten()
            lat = lat.flatten()
            x, y = self.ll2xy(lon, lat)
            obs_pts = np.column_stack([x, y, np.zeros_like(x)])
        return np.ascontiguousarray(obs_pts)
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def cumdistance(self, discretized=False):
        '''
        Computes the distance between the first point of the fault and every other
        point, when you walk along the fault.

        Kwargs:
            * discretized           : if True, use the discretized fault trace

        Returns:
            * cum                   : Array of floats
        '''

        # Get the x and y positions
        if discretized:
            x = self.xi
            y = self.yi
        else:
            x = self.xf
            y = self.yf

        # initialize
        dis = np.zeros((x.shape[0]))

        # Loop
        for i in np.arange(1,x.shape[0]):
            d = np.sqrt((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)
            dis[i] = dis[i-1] + d

        # all done
        return dis
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def AverageAlongStrikeOffsets(self, name, insars, filename, 
                                        discretized=True, smooth=None):
        '''
        !Untested in a looong time...!

        If the profiles have the lon lat vectors as the fault,
        This routines averages it and write it to an output file.
        '''

        if discretized:
            lon = self.loni
            lat = self.lati
        else:
            lon = self.lon
            lat = self.lat

        # Check if good
        for sar in insars:
            dlon = sar.AlongStrikeOffsets[name]['lon']
            dlat = sar.AlongStrikeOffsets[name]['lat']
            assert (dlon==lon).all(), '{} dataset rejected'.format(sar.name)
            assert (dlat==lat).all(), '{} dataset rejected'.format(sar.name)

        # Get distance
        x = insars[0].AlongStrikeOffsets[name]['distance']

        # Initialize lists
        D = []; AV = []; AZ = []; LO = []; LA = []

        # Loop on the distance
        for i in range(len(x)):

            # initialize average
            av = 0.0
            ni = 0.0

            # Get values
            for sar in insars:
                o = sar.AlongStrikeOffsets[name]['offset'][i]
                if np.isfinite(o):
                    av += o
                    ni += 1.0

            # if not only nan
            if ni>0:
                d = x[i]
                av /= ni
                az = insars[0].AlongStrikeOffsets[name]['azimuth'][i]
                lo = lon[i]
                la = lat[i]
            else:
                d = np.nan
                av = np.nan
                az = np.nan
                lo = lon[i]
                la = lat[i]

            # Append
            D.append(d)
            AV.append(av)
            AZ.append(az)
            LO.append(lo)
            LA.append(la)


        # smooth?
        if smooth is not None:
            # Arrays
            D = np.array(D); AV = np.array(AV); AZ = np.array(AZ); LO = np.array(LO); LA = np.array(LA)
            # Get the non nans
            u = np.flatnonzero(np.isfinite(AV))
            # Gaussian Smoothing
            dd = np.abs(D[u][:,None] - D[u][None,:])
            dd = np.exp(-0.5*dd*dd/(smooth*smooth))
            norm = np.sum(dd, axis=1)
            dd = dd/norm[:,None]
            AV[u] = np.dot(dd,AV[u])
            # List
            D = D.tolist(); AV = AV.tolist(); AZ = AZ.tolist(); LO = LO.tolist(); LA = LA.tolist()

        # Open file and write header
        fout = open(filename, 'w')
        fout.write('# Distance (km) || Offset || Azimuth (rad) || Lon || Lat \n')

        # Write to file
        for i in range(len(D)):
            d = D[i]; av = AV[i]; az = AZ[i]; lo = LO[i]; la = LA[i]
            fout.write('{} {} {} {} {} \n'.format(d,av,az,lo,la))

        # Close the file
        fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def ExtractAlongStrikeVariationsOnDiscretizedFault(self, depth=0.5, filename=None, discret=0.5):
        '''
        ! Untested in a looong time !
        
        Extracts the Along Strike variations of the slip at a given depth, resampled along the discretized fault trace.

        Kwargs:
            * depth       : Depth at which we extract the along strike variations of slip.
            * discret     : Discretization length
            * filename    : Saves to a file.

        Returns:
            * None
        '''

        # Import things we need
        import scipy.spatial.distance as scidis

        # Dictionary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # Creates the list where we store things
        # [lon, lat, strike-slip, dip-slip, tensile, distance, xi, yi]
        Var = []

        # Open the output file if needed
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Distance to origin (km) | Position (x,y) (km)\n')

        # Discretize the fault
        if discret is not None:
            self.discretize(every=discret, tol=discret/10., fracstep=discret/12.)
        nd = self.xi.shape[0]

        # Compute the cumulative distance along the fault
        dis = self.cumdistance(discretized=True)

        # Get the patches concerned by the depths asked
        dPatches = []
        sPatches = []
        for p in self.patch:
            # Check depth
            if ((p[0,2]<=depth) and (p[2,2]>=depth)):
                # Get patch
                sPatches.append(self.getslip(p))
                # Put it in dis
                xc, yc = self.getcenter(p)[:2]
                d = scidis.cdist([[xc, yc]], [[self.xi[i], self.yi[i]] for i in range(self.xi.shape[0])])[0]
                imin1 = d.argmin()
                dmin1 = d[imin1]
                d[imin1] = 99999999.
                imin2 = d.argmin()
                dmin2 = d[imin2]
                dtot=dmin1+dmin2
                # Put it along the fault
                xcd = (self.xi[imin1]*dmin1 + self.xi[imin2]*dmin2)/dtot
                ycd = (self.yi[imin1]*dmin1 + self.yi[imin2]*dmin2)/dtot
                # Distance
                if dmin1<dmin2:
                    jm = imin1
                else:
                    jm = imin2
                dPatches.append(dis[jm] + np.sqrt( (xcd-self.xi[jm])**2 + (ycd-self.yi[jm])**2) )

        # Create the interpolator
        ssint = sciint.interp1d(dPatches, [sPatches[i][0] for i in range(len(sPatches))], kind='linear', bounds_error=False)
        dsint = sciint.interp1d(dPatches, [sPatches[i][1] for i in range(len(sPatches))], kind='linear', bounds_error=False)
        tsint = sciint.interp1d(dPatches, [sPatches[i][2] for i in range(len(sPatches))], kind='linear', bounds_error=False)

        # Interpolate
        for i in range(self.xi.shape[0]):
            x = self.xi[i]
            y = self.yi[i]
            lon = self.loni[i]
            lat = self.lati[i]
            d = dis[i]
            ss = ssint(d)
            ds = dsint(d)
            ts = tsint(d)
            Var.append([lon, lat, ss, ds, ts, d, x, y])
            # Write things if asked
            if filename is not None:
                fout.write('{} {} {} {} {} {} {} {} \n'.format(lon, lat, ss, ds, ts, d, x, y))

        # Store it in AlongStrike
        self.AlongStrike['Depth {}'.format(depth)] = np.array(Var)

        # Close fi needed
        if filename is not None:
            fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def ExtractAlongStrikeVariations(self, depth=0.5, origin=None, filename=None, orientation=0.0):
        '''
        Extract the Along Strike Variations of the creep at a given depth

        Kwargs:
            * depth   : Depth at which we extract the along strike variations of slip.
            * origin  : Computes a distance from origin. Give [lon, lat].
            * filename: Saves to a file.
            * orientation: defines the direction of positive distances.

        Returns:
            * None
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Dictionary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # Creates the List where we will store things
        # For each patch, it will be [lon, lat, strike-slip, dip-slip, tensile, distance]
        Var = []

        # Creates the orientation vector
        Dir = np.array([np.cos(orientation*np.pi/180.), np.sin(orientation*np.pi/180.)])

        # initialize the origin
        x0 = 0
        y0 = 0
        if origin is not None:
            x0, y0 = self.ll2xy(origin[0], origin[1])

        # open the output file
        if filename is not None:
            fout = open(filename, 'w')
            fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Patch Area (km2) | Distance to origin (km) \n')

        # compute area, if not done yet
        if not hasattr(self,'area'):
            self.computeArea()

        # Loop over the patches
        for p in self.patch:

            # Get depth range
            dmin = np.min([p[i,2] for i in range(4)])
            dmax = np.max([p[i,2] for i in range(4)])

            # If good depth, keep it
            if ((depth>=dmin) & (depth<=dmax)):

                # Get index
                io = self.getindex(p)

                # Get the slip and area
                slip = self.slip[io,:]
                area = self.area[io]

                # Get patch center
                xc, yc, zc = self.getcenter(p)
                lonc, latc = self.xy2ll(xc, yc)

                # Computes the horizontal distance
                vec = np.array([x0-xc, y0-yc])
                sign = np.sign( np.dot(Dir,vec) )
                dist = sign * np.sqrt( (xc-x0)**2 + (yc-y0)**2 )

                # Assemble
                o = [lonc, latc, slip[0], slip[1], slip[2], area, dist]

                # write output
                if filename is not None:
                    fout.write('{} {} {} {} {} {} \n'.format(lonc, latc, slip[0], slip[1], slip[2], area, dist))

                # append
                Var.append(o)

        # Close the file
        if filename is not None:
            fout.close()

        # Stores it
        self.AlongStrike['Depth {}'.format(depth)] = np.array(Var)

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def ExtractAlongStrikeAllDepths(self, filename=None, discret=0.5):
        '''
        Extracts the Along Strike Variations of the creep at all depths for 
        the discretized fault trace.

        Kwargs:
            * filename      : Name of the output file
            * discret       : Fault discretization

        Returns:
            * None
        '''

        # Dictionnary to store these guys
        if not hasattr(self, 'AlongStrike'):
            self.AlongStrike = {}

        # If filename provided, create it
        if filename is not None:
            fout = open(filename, 'w')

        # Create the list of depths
        depths = np.unique(np.array([[self.patch[i][u,2] for u in range(4)] for i in range(len(self.patch))]).flatten())
        depths = depths[:-1] + (depths[1:] - depths[:-1])/2.

        # Discretize the fault
        self.discretize(every=discret, tol=discret/10., fracstep=discret/12.)

        # For a list of depths, iterate
        for d in depths.tolist():

            # Get the values
            self.ExtractAlongStrikeVariationsOnDiscretizedFault(depth=d, filename=None, discret=None)

            # If filename, write to it
            if filename is not None:
                fout.write('> # Depth = {} \n'.format(d))
                fout.write('# Lon | Lat | Strike-Slip | Dip-Slip | Tensile | Distance to origin (km) | x, y \n')
                Var = self.AlongStrike['Depth {}'.format(d)]
                for i in range(Var.shape[0]):
                    lon = Var[i,0]
                    lat = Var[i,1]
                    ss = Var[i,2]
                    ds = Var[i,3]
                    ts = Var[i,4]
                    dist = Var[i,5]
                    x = Var[i,6]
                    y = Var[i,7]
                    fout.write('{} {} {} {} {} {} {} \n'.format(lon, lat, ss, ds, ts, area, dist, x, y))

        # Close file if done
        if filename is not None:
            fout.close()

        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def plot(self, figure=134, slip='total', equiv=False, show=True, 
             norm=None, linewidth=1.0, plot_on_2d=True, 
             colorbar=True, cbaxis=[0.1, 0.2, 0.1, 0.02], cborientation='horizontal', cblabel='', 
             drawCoastlines=True, expand=0.2, savefig=False, scalebar=None, figsize=(None, None),
             cmap='jet', edgecolor='slip', ftype='eps', dpi=600, bbox_inches=None, suffix='',
             remove_direction_labels=False, cbticks=None, cblinewidth=1, cbfontsize=10, cb_label_side='opposite', map_cbaxis=None):
        '''
        Plot the available elements of the fault.
        
        Kwargs:
            * figure        : Number of the figure.
            * slip          : What slip to plot
            * equiv         : useless. For consitency between fault objects
            * show          : Show me
            * Norm          : colorbar min and max values
            * linewidth     : Line width in points
            * plot_on_2d    : Plot on a map as well
            * drawCoastline : Self-explanatory argument...
            * expand        : Expand the map by {expand} degree around the edges
                              of the fault.
            * savefig       : Save figures as eps.
            * scalebar      : Length of a scalebar (float, default is None)
            * remove_direction_labels : If True, remove E, N, S, W from axis labels (default is False)
            * cbticks       : List of ticks to set on the colorbar
            * cblinewidth   : Width of the colorbar label border and tick lines
            * cbfontsize    : Font size of the colorbar label, default is 10
            * cb_label_side : Position of the label relative to the ticks ('opposite' or 'same'), default is 'opposite'
            * map_cbaxis    : Axis for the colorbar on the map plot, default is None
        
        Returns:
            * None
        '''
    
        # Get lons lats
        lon = np.unique(np.array([p[:,0] for p in self.patchll]))
        #lon[lon<0.] += 360.
        lat = np.unique(np.array([p[:,1] for p in self.patchll]))
        lonmin = lon.min()-expand
        lonmax = lon.max()+expand
        latmin = lat.min()-expand
        latmax = lat.max()+expand
    
        # Create a figure
        fig = geoplot(figure=figure, lonmin=lonmin, lonmax=lonmax, 
                                         latmin=latmin, latmax=latmax, figsize=figsize,
                                         remove_direction_labels=remove_direction_labels
                                         )
    
        # Draw the coastlines
        if drawCoastlines:
            fig.drawCoastlines(parallels=None, meridians=None, drawOnFault=True)
    
        # Draw the fault
        fig.faultpatches(self, slip=slip, norm=norm, colorbar=colorbar, cbaxis=cbaxis, cborientation=cborientation, cblabel=cblabel, 
                         plot_on_2d=plot_on_2d, linewidth=linewidth, cmap=cmap, edgecolor=edgecolor,
                         cbticks=cbticks, cblinewidth=cblinewidth, cbfontsize=cbfontsize, cb_label_side=cb_label_side, map_cbaxis=map_cbaxis)
    
        # Savefigs?
        if savefig:
            prefix = self.name.replace(' ','_')
            suffix = f'_{suffix}' if suffix != '' else ''
            saveFig = ['fault']
            if plot_on_2d:
                saveFig.append('map')
            fig.savefig(prefix+'{0}_{1}'.format(suffix, slip), ftype=ftype, dpi=dpi, bbox_inches=bbox_inches, saveFig=saveFig)
    
        self.slipfig = fig
        # show
        if show:
            showFig = ['fault']
            if plot_on_2d:
                showFig.append('map')
            fig.show(showFig=showFig)
    
        # All done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def plotMayavi(self, neg_depth=True, value_to_plot='total', colormap='jet',
                   reverseSign=False):
        '''
        ! OBSOLETE BUT KEPT HERE TO BE TESTED IN THE FUTURE !
        Plot 3D representation of fault using MayaVi.

        Args:
            * neg_depth     : Flag to specify if patch depths are negative or positive
            * value_to_plot : What to plot on patches
            * colormap      : Colormap for patches
            * reverseSign   : Flag to reverse sign of value_to_plot

        ! OBSOLETE BUT KEPT HERE TO BE TESTED IN THE FUTURE !
        '''
        try:
            from mayavi import mlab
        except ImportError:
            print('mayavi module not installed. skipping plotting...')
            return

        # Sign factor for negative depths
        negFactor = -1.0
        if neg_depth:
            negFactor = 1.0

        # Sign for values
        valueSign = 1.0
        if reverseSign:
            valueSign = -1.0

        # Plot the wireframe %%% modified by kfh, 10/11/2020 np.copy
        x, y, z = self.Vertices[:,0], self.Vertices[:,1], np.copy(self.Vertices[:,2])
        z *= negFactor
        mesh = mlab.triangular_mesh(x, y, z, self.Faces, representation='wireframe',
                                    opacity=0.6, color=(0.0,0.0,0.0))

        # Compute the scalar value to color the patches
        if value_to_plot == 'total':
            self.computetotalslip()
            plotval = self.totalslip
        elif value_to_plot == 'strikeslip':
            plotval = self.slip[:,0]
        elif value_to_plot == 'dipslip':
            plotval = self.slip[:,1]
        elif value_to_plot == 'tensile':
            plotval = self.slip[:,2]
        elif value_to_plot == 'index':
            plotval = np.linspace(0, len(self.patch)-1, len(self.patch))
        else:
            assert False, 'unsupported value_to_plot'

        # Assign the scalar data to a source dataset
        cell_data = mesh.mlab_source.dataset.cell_data
        cell_data.scalars = valueSign * plotval
        cell_data.scalars.name = 'Cell data'
        cell_data.update()

        # Make a new mesh with the scalar data applied to patches
        mesh2 = mlab.pipeline.set_active_attribute(mesh, cell_scalars='Cell data')
        surface = mlab.pipeline.surface(mesh2, colormap=colormap)

        mlab.colorbar(surface)
        mlab.show()

        return
    # ----------------------------------------------------------------------


    # ----------------------------------------------------------------------
    def mapFault2Fault(self, Map, fault):
        '''
        User provides a Mapping function np.array((len(self.patch), len(fault.patch)))
        and a fault and the slip from the argument
        fault is mapped into self.slip.
        
        Function just does
        self.slip[:,0] = np.dot(Map,fault.slip)
        '''

        # Check size
        if self.N_slip!=None and self.N_slip!=len(self.patch):
            raise NotImplementedError('Only works for len(slip)==len(patch)')

        # Get the number of patches
        nPatches = len(self.patch)
        nPatchesExt = len(fault.patch)

        # Assert the Mapping function is correct
        assert(Map.shape==(nPatches,nPatchesExt)), 'Mapping function has the wrong size...'

        # Map the slip
        self.slip[:,0] = np.dot(Map, fault.slip[:,0])
        self.slip[:,1] = np.dot(Map, fault.slip[:,1])
        self.slip[:,2] = np.dot(Map, fault.slip[:,2])

        # all done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def mapUnder2Above(self, deepfault):
        '''
        This routine is very very particular. It only works with 2 vertical faults.
        It Builds the mapping function from one fault to another, when these are vertical.
        These two faults must have the same surface trace. If the deep fault has more than one raw of patches,
        it might go wrong and give some unexpected results.

        Args:
            * deepfault     : Deep section of the fault.
        '''

        # Assert faults are compatible
        assert ( (self.lon==deepfault.lon).all() and (self.lat==deepfault.lat).all()), 'Surface traces are different...'

        # Check that all patches are verticals
        dips = np.array([self.getpatchgeometry(i)[-1]*180./np.pi for i in range(len(self.patch))])
        assert((dips == 90.).all()), 'Not viable for non-vertical patches, fault {}....'.format(self.name)
        deepdips = np.array([deepfault.getpatchgeometry(i)[-1]*180./np.pi for i in range(len(deepfault.patch))])
        assert((deepdips == 90.).all()), 'Not viable for non-vertical patches, fault {}...'.format(deepfault.name)

        # Get the number of patches
        nPatches = len(self.patch)
        nDeepPatches = len(deepfault.patch)

        # Create the map from under to above
        Map = np.zeros((nPatches, nDeepPatches))

        # Discretize the surface trace quite finely
        self.discretize(every=0.5, tol=0.05, fracstep=0.02)

        # Compute the cumulative distance along the fault
        dis = self.cumdistance(discretized=True)

        # Compute the cumulative distance between the beginning of the fault and the corners of the patches
        distance = []
        for p in self.patch:
            D = []
            # for each corner
            for c in p:
                # Get x,y
                x = c[0]
                y = c[1]
                # Get the index of the nearest xi value under x
                i = np.flatnonzero(x>=self.xi)[-1]
                # Get the corresponding distance along the fault
                d = dis[i] + np.sqrt( (x-self.xi[i])**2 + (y-self.yi[i])**2 )
                # Append
                D.append(d)
            # Array unique
            D = np.unique(np.array(D))
            # append
            distance.append(D)

        # Do the same for the deep patches
        deepdistance = []
        for p in deepfault.patch:
            D = []
            for c in p:
                x = c[0]
                y = c[1]
                i = np.flatnonzero(x>=self.xi)
                if len(i)>0:
                    i = i[-1]
                    d = dis[i] + np.sqrt( (x-self.xi[i])**2 + (y-self.yi[i])**2 )
                else:
                    d = 99999999.
                D.append(d)
            D = np.unique(np.array(D))
            deepdistance.append(D)

        # Numpy arrays
        distance = np.array(distance)
        deepdistance = np.array(deepdistance)

        # Loop over the patches to find out which are over which
        for p in range(len(self.patch)):

            # Get the patch distances
            d1 = distance[p,0]
            d2 = distance[p,1]

            # Get the index for the points
            i1 = np.intersect1d(np.flatnonzero((d1>=deepdistance[:,0])), np.flatnonzero((d1<deepdistance[:,1])))[0]
            i2 = np.intersect1d(np.flatnonzero((d2>deepdistance[:,0])), np.flatnonzero((d2<=deepdistance[:,1])))[0]

            # two cases possible:
            if i1==i2:              # The shallow patch is fully inside the deep patch
                Map[p,i1] = 1.0     # All the slip comes from this patch
            else:                   # The shallow patch is on top of several patches
                # two cases again
                if np.abs(i2-i1)==1:       # It covers the boundary between 2 patches
                    delta1 = np.abs(d1-deepdistance[i1][1])
                    delta2 = np.abs(d2-deepdistance[i2][0])
                    total = delta1 + delta2
                    delta1 /= total
                    delta2 /= total
                    Map[p,i1] = delta1
                    Map[p,i2] = delta2
                else:                       # It is larger than the boundary between 2 patches and covers several deep patches
                    delta = []
                    delta.append(np.abs(d1-deepdistance[i1][1]))
                    for i in range(i1+1,i2):
                        delta.append(np.abs(deepdistance[i][1]-deepdistance[i][0]))
                    delta.append(np.abs(d2-deepdistance[i2][0]))
                    delta = np.array(delta)
                    total = np.sum(delta)
                    delta = delta/total
                    for i in range(i1,i2+1):
                        Map[p,i] = delta

        # All done
        return Map
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def getSubSourcesFault(self, verbose=True):
        '''
        Returns a TriangularPatches fault object with each triangle
        corresponding to the subsources used for plotting.
    
        Kwargs:
            * verbose       : Talk to me (default: True)

        Returns:
            * fault         : Returns a triangularpatches instance
        '''

        # Import What is needed
        from .EDKSmp import dropSourcesInPatches as Patches2Sources

        # Drop the sources in the patches and get the corresponding fault
        Ids, xs, ys, zs, strike, dip, Areas, fault = Patches2Sources(self, 
                                                verbose=verbose,
                                                returnSplittedPatches=True)
        self.plotSources = [Ids, xs, ys, zs, strike, dip, Areas]

        # Interpolate the slip on each subsource
        fault.initializeslip()
        fault.slip[:,0] = self._getSlipOnSubSources(Ids, xs, ys, zs, self.slip[:,0])
        fault.slip[:,1] = self._getSlipOnSubSources(Ids, xs, ys, zs, self.slip[:,1])
        fault.slip[:,2] = self._getSlipOnSubSources(Ids, xs, ys, zs, self.slip[:,2])

        # All done
        return fault

    def find_ordered_edge_vertices(self, edge='top', depth=None, buffer_depth=0.1, 
                                   top_tolerance=0.1, bottom_tolerance=0.1, refind=True,
                                   return_indices=False, merge_threshold=0.5, method='hybrid'):
        """
        Find the ordered edge vertices from the edge triangles.
    
        Parameters:
        -----------
        edge : str, optional
            The edge to find vertices for ('top' or 'bottom'). Default is 'top'.
        depth : float, optional
            The depth to use for the edge. If None, uses self.top for 'top' edge and self.depth for 'bottom' edge.
        buffer_depth : float, optional
            The buffer depth to include points within the edge. Default is 0.1.
        top_tolerance : float, optional
            The tolerance for the top edge. Default is 0.1.
        bottom_tolerance : float, optional
            The tolerance for the bottom edge. Default is 0.1.
        refind : bool, optional
            Whether to refind the edge vertices. Default is True.
        return_indices : bool, optional
            Whether to return the indices of the ordered vertices. Default is False.
        merge_threshold : float, optional
            The threshold for merging vertices. Default is 0.5.
        method : str, optional
            The method to use for reconstruction. Default is 'hybrid'.

        Returns:
        --------
        ordered_vertices : list
            List of ordered edge vertices.
        """
        self.find_fault_edge_vertices(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, refind=refind)

        if edge not in ['top', 'bottom']:
            raise ValueError("Invalid value for edge. It should be 'top' or 'bottom'.")
    
        if depth is None:
            self.top = np.min(self.Vertices[:, 2])
            self.depth = np.max(self.Vertices[:, 2])
            depth = self.top if edge == 'top' else self.depth
    
        edge_faces = self.Faces[self.edge_triangles_indices[edge]]
        vertices = self.Vertices[edge_faces]
    
        # Select points within depth and buffer depth
        edge_points_mask = np.abs(vertices[:, :, 2] - depth) <= buffer_depth
        edge_points_index = set(edge_faces[edge_points_mask])
    
        # Create a dictionary to count occurrences of each vertex
        vertex_count = {}
        for face in edge_faces:
            for vertex in face:
                if vertex in edge_points_index:
                    if vertex in vertex_count:
                        vertex_count[vertex] += 1
                    else:
                        vertex_count[vertex] = 1

        # Count how many vertices have count=1
        count_ones = sum(1 for count in vertex_count.values() if count == 1)

        # If more than 2 vertices have count=1, use CurveReconstructor
        if count_ones > 2:
            # print(f"Found {count_ones} vertices with count=1, using CurveReconstructor...")
            
            from .edge_utils.fault_edge_reconstruction import CurveReconstructor
            
            vertex_indices = list(vertex_count.keys())
            vertex_positions = self.Vertices[vertex_indices]
            
            reconstructor = CurveReconstructor(vertex_positions, merge_threshold=merge_threshold)
            _, _, original_path = reconstructor.reconstruct(method=method, optimize=False)
            
            # Map back to original vertex indices
            ordered_vertices = [vertex_indices[i] for i in original_path]
            ordered_edge_points = self.Vertices[ordered_vertices]

            if return_indices:
                return ordered_edge_points, ordered_vertices
            else:
                return ordered_edge_points

        # Find the starting vertex (vertex with only one occurrence)
        start_vertex = None
        for vertex, count in vertex_count.items():
            if count == 1:
                start_vertex = vertex
                break
    
        if start_vertex is None:
            raise ValueError("No starting vertex found. Check the edge triangles.")
    
        # Order the vertices
        ordered_vertices = [start_vertex]
        current_vertex = start_vertex
        while len(ordered_vertices) < len(vertex_count):
            found_next_vertex = False
            for face in edge_faces:
                if current_vertex in face:
                    for vertex in face:
                        if vertex != current_vertex and vertex not in ordered_vertices and vertex in vertex_count:
                            ordered_vertices.append(vertex)
                            current_vertex = vertex
                            found_next_vertex = True
                            break
                    if found_next_vertex:
                        break
            if not found_next_vertex:
                raise ValueError("Cannot find the next vertex. Check the edge triangles.")
    
        # Map ordered vertices to edge points
        ordered_edge_points = self.Vertices[ordered_vertices]

        if return_indices:
            return ordered_edge_points, ordered_vertices
        else:
            return ordered_edge_points

    def getfaultEdgeTriangles_and_EdgeLines(self, top_tolerance=0.1, bottom_tolerance=0.1, 
                                            refind=False, method='hybrid', merge_threshold=0.5):
        '''
        Left: In West, Right: In East
        Left: In North, Right: In South if the left/right can not be determined from west/east

        Top edge vertices are ordered from left to right.
        Bottom edge vertices are ordered from left to right.
        Left edge vertices are ordered from top to bottom.
        Right edge vertices are ordered from top to bottom.
        '''
        # find boundary and corner triangles indexes in fault.Faces
        fault = self
        if not hasattr(fault, 'edge_triangles_indices') or refind:
            fault.find_fault_edge_vertices(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance, refind=True)
    
        if not hasattr(fault, 'edge_vertex_indices') or refind:
            fault.find_fault_fouredge_vertices(top_tolerance=top_tolerance, bottom_tolerance=bottom_tolerance,
                                               refind=True, method=method, merge_threshold=merge_threshold)

        def sort_edge_triangles(edge):
            edge_tri_sort = []
            for a_ind, b_ind in np.repeat(self.edge_vertex_indices[edge], 2)[1:-1].reshape(-1, 2):
                for tri_ind in self.edge_triangles_indices[edge]:
                    tri_vert_inds = fault.Faces[tri_ind]
                    if a_ind in tri_vert_inds and b_ind in tri_vert_inds:
                        edge_tri_sort.append(tri_ind)
                        break
            return np.array(edge_tri_sort, dtype=int)
    
        self.edge_triangles_indices['top'] = sort_edge_triangles('top')
        self.edge_triangles_indices['bottom'] = sort_edge_triangles('bottom')
    
        def sort_edge_by_center_z(edge):
            center_z = np.mean(fault.Vertices[fault.Faces[self.edge_triangles_indices[edge]], -1], axis=1)
            sort_order = np.argsort(center_z)
            self.edge_triangles_indices[edge] = self.edge_triangles_indices[edge][sort_order]
    
        sort_edge_by_center_z('left')
        sort_edge_by_center_z('right')

        # sort top and bottom from left to right
        if self.edge_vertex_indices['top'][0] not in self.edge_vertex_indices['left']:
            self.edge_vertex_indices['top'] = self.edge_vertex_indices['top'][::-1]
            self.edge_vertices['top'] = self.edge_vertices['top'][::-1]
            self.edge_triangles_indices['top'] = self.edge_triangles_indices['top'][::-1]
        
        if self.edge_vertex_indices['bottom'][0] not in self.edge_vertex_indices['left']:
            self.edge_vertex_indices['bottom'] = self.edge_vertex_indices['bottom'][::-1]
            self.edge_vertices['bottom'] = self.edge_vertices['bottom'][::-1]
            self.edge_triangles_indices['bottom'] = self.edge_triangles_indices['bottom'][::-1]
    
        # All Done
        return
    # ----------------------------------------------------------------------

    # ----------------------------------------------------------------------
    def findAsperities(self, function, slip='strikeslip', verbose=True):
        '''
        Finds the number, size and location of asperities that are identified by the 
        given function.

        Args:
            * function          : Function that takes an array the size of the number of patches and returns an array of bolean the same size. Trues are within the asperity.

        Kwargs:
            * slip              : Which slip vector do you want to apply the function to
            * verbose           : Talk to me?

        Returns:
            * Asperities
        '''

        # Assert 
        assert self.patchType == 'triangle', 'Not implemented for Triangular tents'

        # Update the map
        def _checkUpdate(check, iTriangle, modifier, fault):
            # Get the 3 surrounding triangles
            Adjacents = fault.adjacencyMap[iTriangle]
            # Check if they are in the asperity
            modify = [iTriangle]
            for adjacent in Adjacents:
                if check[adjacent]==1.: modify.append(adjacent)
            # Modify the map
            for mod in modify: check[mod] = modifier
            # Return the triangles surrounding
            modify.remove(iTriangle)
            return modify

        # Get the array to test
        if slip == 'strikeslip':
            values = self.slip[:,0]
        elif slip == 'dipslip':
            values = self.slip[:,1]
        elif slip == 'tensile':
            values = self.slip[:,2]
        elif slip == 'coupling':
            values = self.coupling
        else:
            print('findAsperities: Unknown type slip vector...')

        # Get the bolean array
        test = function(values).astype(float)

        # Build the adjacency Map
        if self.adjacencyMap is None:
            self.buildAdjacencyMap(verbose=verbose)

        # Number of the first asperity
        i = 1

        # We iterate until no triangle has been classified in an asperity
        # 0 means the triangle is not in an asperity
        # 1 means the triangle is in an asperity
        # 2 or more means the triangle is in an asperity and has been classified
        while len(np.flatnonzero(test==1.))>0:

            # Pick a triangle inside an asperity
            iTriangle = np.flatnonzero(test==1.)[0]
 
            # This is asperity i
            i += 1

            # Talk to me
            if verbose:
                print('Dealing with asperity #{}'.format(i))

            # Build the list of new triangles to check
            toCheck = _checkUpdate(test, iTriangle, i, self)

            # While toCheck has stuff to check, check them
            nT = 0
            while len(toCheck)>0:
                # Get triangle to check
                iCheck = toCheck.pop()
                if verbose:
                    nT += 1
                    sys.stdout.write('\r Triangles: {}'.format(nT))
                    sys.stdout.flush()
                # Check it
                toCheck += _checkUpdate(test, iCheck, i, self)

        # Normally, all the 1. have been replaced by 2., 3., etc
        
        # Find the unique numbers in test
        Counters = np.unique(test).tolist()
        Counters.remove(0.)
    
        # Get the asperities
        Asperities = []
        for counter in Counters:
            Asperities.append(np.flatnonzero(test==counter))

        # All done
        return Asperities
    # ----------------------------------------------------------------------

#EOF
