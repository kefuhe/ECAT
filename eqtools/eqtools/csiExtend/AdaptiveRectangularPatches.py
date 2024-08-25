import numpy as np
from csi.planarfault import planarfault


class AdaptiveRectangularPatches(planarfault):
    """
    """
    def __init__(self, name, utmzone=None, ellps='WGS84', verbose=True, lon0=None, lat0=None):
        """
        Initialize the class.
        """
        super(AdaptiveRectangularPatches, self).__init__(name, 
                                                         utmzone=utmzone, 
                                                         ellps=ellps, 
                                                         verbose=verbose, 
                                                         lon0=lon0, 
                                                         lat0=lat0)
        
        # All Done
        return
    
    def set_coords(self, coords, lonlat=True, coord_type='top'):
        """
        Sets the coordinates.

        Parameters:
        coords: Coordinates, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.
        coord_type: The type of coordinates, which can be 'top', 'bottom', or 'layer'.

        Note:
        This method configures the coordinates for a specific boundary or layer within the model. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter. The coord_type parameter specifies the boundary or layer these coordinates apply to.
        """
        if coords is None or len(coords) == 0:
            raise ValueError(f"{coord_type}_coords cannot be None or empty.")
        if coord_type == 'layer':
            if not isinstance(coords, list):
                raise ValueError("For 'layer', coords should be a list of 2D arrays.")
            self.layers_ll = [None] * len(coords)
            self.layers = [None] * len(coords)
            for i, layer_coords in enumerate(coords):
                if lonlat:
                    self.layers_ll[i] = layer_coords
                    lon, lat = layer_coords[:, 0], layer_coords[:, 1]
                    x, y = self.ll2xy(lon, lat)
                    self.layers[i] = np.vstack((x, y, layer_coords[:, 2])).T
                else:
                    self.layers[i] = layer_coords
                    x, y, z = layer_coords[:, 0], layer_coords[:, 1], layer_coords[:, 2] # unit: km
                    lon, lat = self.xy2ll(x, y)
                    self.layers_ll[i] = np.vstack((lon, lat, z)).T
        else:
            if lonlat:
                setattr(self, f"{coord_type}_coords_ll", coords)
                lon, lat = coords[:, 0], coords[:, 1]
                x, y = self.ll2xy(lon, lat)
                setattr(self, f"{coord_type}_coords", np.vstack((x, y, coords[:, 2])).T)
            else:
                setattr(self, f"{coord_type}_coords", coords)
                x, y, z = coords[:, 0], coords[:, 1], coords[:, 2] # unit: km
                lon, lat = self.xy2ll(x, y)
                setattr(self, f"{coord_type}_coords_ll", np.vstack((lon, lat, z)).T)

    def set_top_coords(self, top_coords, lonlat=True):
        """
        Sets the top coordinates.

        Parameters:
        top_coords: The coordinates of the top, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in top_coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.

        Note:
        This method configures the top boundary of a model by specifying its coordinates. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter.
        """
        self.set_coords(top_coords, lonlat, 'top')

    def set_bottom_coords(self, bottom_coords, lonlat=True):
        """
        Sets the bottom coordinates.

        Parameters:
        bottom_coords: The coordinates of the bottom, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in bottom_coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.

        Note:
        This method configures the bottom boundary of a model by specifying its coordinates. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter.
        """
        self.set_coords(bottom_coords, lonlat, 'bottom')

    def set_layer_coords(self, layer_coords, lonlat=True):
        """
        Sets the layer coordinates.

        Parameters:
        layer_coords: The coordinates of the layer, a two-dimensional array where each row represents the coordinates of a point.
        lonlat: If True, indicates that the coordinates in layer_coords are in longitude and latitude. Otherwise, the coordinates are in UTM format.

        Note:
        This method configures the coordinates of a specific layer within a model. The coordinate system (longitude/latitude or UTM) is determined by the lonlat parameter.
        """
        self.set_coords(layer_coords, lonlat, 'layer')

    def generate_new_solution(self, clon=None, clat=None, cdepth=None,
                              strike=None, dip=None, length=None, width=None, 
                              top=None, depth=None):
        """
        Generate the top and bottom coordinates of the fault trace from the nonlinear solution.

        Parameters:
        clon (float): The longitude of the center point of the top line.
        clat (float): The latitude of the center point of the top line.
        cdepth (float): The depth of the center point of the top line.
        strike (float): The strike angle of the fault patch.
        dip (float): The dip angle of the fault patch.
        length (float): The length of the fault patch.
        width (float): The width of the fault patch.
        top (float): The top depth of the fault patch.
        depth (float): The bottom depth of the fault patch.

        Returns:
        top_coords: The top coordinates of the fault patch.
        bottom_coords: The bottom coordinates of the fault patch.
        """
        from numpy import deg2rad, sin, cos, tan

        if any(param is None for param in [clon, clat, cdepth, strike, dip, length]):
            raise ValueError("Please provide all the required parameters.")
        
        # Convert the strike and dip angles to radians
        str_rad = deg2rad(90 - strike)
        dip_rad = deg2rad(dip)
        half_length = length / 2.0
        # top or self.top, at least one of them should be provided
        if top is None:
            assert hasattr(self, 'top') and self.top is not None, "Please provide the top depth of the fault patch."
            top = self.top

        # width or depth/self.depth, at least one of them should be provided； if they are both provided, depth/self.depth will be used
        if width is None:
            assert depth is not None or (hasattr(self, 'depth') and self.depth is not None), "Please provide the width or depth of the fault patch."
            depth = depth if depth is not None else self.depth
            width = (depth - top) / sin(dip_rad)
        else:
            if depth is None:
                depth = self.depth if hasattr(self, 'depth') and self.depth is not None else width * sin(dip_rad) + top

        self.depth = depth
        
        # Calculate the top two end points of the fault patch
        cx, cy = self.ll2xy(clon, clat)
        cxy_trans = (cx + 1.j*cy) * np.exp(1.j*-str_rad)
        cxy_trans_neg = cxy_trans - half_length
        cxy_trans_pos = cxy_trans + half_length

        # top
        top_offset = (cdepth - top) / tan(dip_rad)
        cxy_top_trans_neg = cxy_trans_neg + top_offset*1.j
        cxy_top_trans_pos = cxy_trans_pos + top_offset*1.j
        cxy_top_neg = cxy_top_trans_neg * np.exp(1.j*str_rad)
        cxy_top_pos = cxy_top_trans_pos * np.exp(1.j*str_rad)
        top_coords = np.array([[cxy_top_neg.real, cxy_top_neg.imag, top], [cxy_top_pos.real, cxy_top_pos.imag, top]])
        self.set_top_coords(top_coords, lonlat=False)
        # bottom
        bottom_offset = (cdepth - depth) / tan(dip_rad)
        cxy_bottom_trans_neg = cxy_trans_neg + bottom_offset*1.j
        cxy_bottom_trans_pos = cxy_trans_pos + bottom_offset*1.j
        cxy_bottom_neg = cxy_bottom_trans_neg * np.exp(1.j*str_rad)
        cxy_bottom_pos = cxy_bottom_trans_pos * np.exp(1.j*str_rad)
        bottom_coords = np.array([[cxy_bottom_neg.real, cxy_bottom_neg.imag, depth], [cxy_bottom_pos.real, cxy_bottom_pos.imag, depth]])
        self.set_bottom_coords(bottom_coords, lonlat=False)

        top_coords = self.top_coords
        bottom_coords = self.bottom_coords
        cx_new, cy_new = np.mean(top_coords[:, 0]), np.mean(top_coords[:, 1])
        clon_new, clat_new = self.xy2ll(cx_new, cy_new)
        cdepth_new = np.mean(top_coords[:, 2])
        width_new = np.linalg.norm(bottom_coords - top_coords, axis=1).mean()

        # All Done
        return clon_new, clat_new, cdepth_new, strike, dip, length, width_new
    
    def buildPatches_from_nonlinear_soln(self, clon=None, clat=None, cdepth=None,
                                         strike=None, dip=None, length=None, width=None, 
                                         top=None, depth=None, n_strike=None, n_dip=None, verbose=True):
        """
        Build patches from the nonlinear solution.

        Parameters:
        clon (float): The longitude of the center point of the top line.
        clat (float): The latitude of the center point of the top line.
        cdepth (float): The depth of the center point of the top line.
        strike (float): The strike angle of the fault patch.
        dip (float): The dip angle of the fault patch.
        length (float): The length of the fault patch.
        width (float): The width of the fault patch.
        top (float): The top depth of the fault patch.
        depth (float): The bottom depth of the fault patch.
        n_strike (int): The number of patches along the strike direction.
        n_dip (int): The number of patches along the dip direction.

        Returns:
        patches: A list of patches.
        """
        if any(param is None for param in [clon, clat, cdepth, strike, dip, length, n_strike, n_dip]):
            raise ValueError("Please provide all the required parameters.")
        
        clon_new, clat_new, cdepth_new, strike, dip, length, width_new = self.generate_new_solution(clon, clat, cdepth, strike, dip, length, width, top, depth)
        self.buildPatches(clon_new, clat_new, cdepth_new, strike, dip, length, width_new, n_strike, n_dip, verbose=verbose)
        
        # All Done
        return
    
    def compute_patch_areas(self):
        '''
        Compute the area of all patches and store them in {area}

        added by Kefeng he at 2024/08/23
        '''
        self.area = np.array([self.patchArea(p) for p in self.equivpatch])
        return self.area