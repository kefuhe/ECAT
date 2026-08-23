from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import os
from .geometry_utils import FaultGeometry, ReferencePoint


class BaseSlipConverter(ABC):
    """
    Abstract base class for slip distribution data converters.
    
    This class provides common functionality for:
    - GMT format output
    - 3D visualization
    - Statistical analysis
    - Coordinate transformations
    
    Subclasses need to implement the data reading method and define format-specific parameters.
    """
    
    def __init__(self, input_file):
        """
        Initialize the base slip converter.
        
        Args:
            input_file (str): Path to the input slip data file
        """
        self.input_file = input_file
        self.slip_data = None
        self.fault_geometry = None
        
        # Format-specific attributes to be set by subclasses
        self.reference_point = None
        self.depth_unit = None
        self.length_unit = None
        self.width_unit = None
        self.converter_type = None
        
    @abstractmethod
    def read_slip_file(self):
        """
        Read slip distribution file. Must be implemented by subclasses.
        
        Should populate self.slip_data with a dictionary containing:
        - 'longitude': array of longitude values (degrees)
        - 'latitude': array of latitude values (degrees)
        - 'depth': array of depth values (positive downward)
        - 'length': array of fault patch lengths
        - 'width': array of fault patch widths
        - 'strike': array of strike angles (degrees)
        - 'dip': array of dip angles (degrees)
        - 'strike_slip': array of strike slip values
        - 'dip_slip': array of dip slip values
        - 'total_slip': array of total slip magnitudes
        
        Notes:
            - Depth should be in the unit specified by self.depth_unit
            - Length and width should be in the units specified by self.length_unit and self.width_unit
            - Strike and dip should be in degrees
            - Slip values should be in meters
            - slip direction convention:
                - Strike slip: positive for left-lateral (sinistral)
                - Dip slip: positive for reverse (upward) motion

        Returns:
            bool: True if successful, False otherwise
        """
        pass
    
    @abstractmethod
    def _setup_projection(self):
        """
        Setup coordinate system and fault geometry calculator.
        Must be implemented by subclasses.
        """
        pass
    
    def calculate_patch_corners(self, lon, lat, depth, length, width, strike, dip, pos_s=0.0, pos_d=0.0):
        """
        Calculate the four corner points of a rectangular fault patch.
        
        Args:
            lon (float): Longitude of fault reference point (degrees)
            lat (float): Latitude of fault reference point (degrees)
            depth (float): Depth of fault reference point (positive downward)
            length (float): Fault length along strike
            width (float): Fault width along dip
            strike (float): Strike angle (degrees)
            dip (float): Dip angle (degrees)
            pos_s (float): Position along strike relative to reference point (default: 0.0)
            pos_d (float): Position along dip relative to reference point (default: 0.0)
            
        Returns:
            list: Four corner points as [(lon1, lat1, depth1), ...]
        """
        if self.fault_geometry is None:
            raise ValueError("Fault geometry not initialized. Call read_slip_file() first.")
        
        return self.fault_geometry.calculate_fault_corners(
            lon, lat, depth, length, width, strike, dip,
            pos_s=pos_s, pos_d=pos_d,
            reference_point=self.reference_point,
            depth_unit=self.depth_unit,
            length_unit=self.length_unit,
            width_unit=self.width_unit,
            converter_type=self.converter_type
        )
    
    def write_gmt_file(self, output_file=None, custom_slip_components=None):
        """
        Convert slip data to GMT format file.
        
        Args:
            output_file (str): Output GMT file path
            custom_slip_components (dict): Custom slip components to include
                                         e.g., {'opening': True, 'tensile': False} 
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.slip_data is None:
            print("Error: No data loaded. Call read_slip_file() first.")
            return False
        
        if output_file is None:
            output_file = f"{self.__class__.__name__.lower()}_slip.gmt"
        
        try:
            with open(output_file, 'w') as fout:
                n_patches = len(self.slip_data['longitude'])
                
                for i in range(n_patches):
                    # Get patch parameters
                    lon = self.slip_data['longitude'][i]
                    lat = self.slip_data['latitude'][i]
                    depth = self.slip_data['depth'][i]
                    length = self.slip_data['length'][i]
                    width = self.slip_data['width'][i]
                    strike = self.slip_data['strike'][i]
                    dip = self.slip_data['dip'][i]
                    
                    # Slip values
                    strike_slip = self.slip_data['strike_slip'][i]
                    dip_slip = self.slip_data['dip_slip'][i]
                    total_slip = self.slip_data['total_slip'][i]

                    # pos_s pos_d
                    if 'pos_s' in self.slip_data and 'pos_d' in self.slip_data:
                        pos_s = self.slip_data['pos_s'][i]
                        pos_d = self.slip_data['pos_d'][i]
                    else:
                        pos_s = 0.0
                        pos_d = 0.0

                    # Calculate corner points
                    corners = self.calculate_patch_corners(
                        lon, lat, depth, length, width, strike, dip,
                        pos_s=pos_s, pos_d=pos_d
                    )
                    
                    # Write GMT segment header with slip information
                    if custom_slip_components and 'opening' in custom_slip_components:
                        if custom_slip_components['opening'] and 'opening' in self.slip_data:
                            opening = self.slip_data['opening'][i]
                            fout.write(f"> -Z{total_slip:.6f} # {strike_slip:.6f} {dip_slip:.6f} {opening:.6f}\n")
                        else:
                            fout.write(f"> -Z{total_slip:.6f} # {strike_slip:.6f} {dip_slip:.6f} 0.0\n")
                    else:
                        fout.write(f"> -Z{total_slip:.6f} # {strike_slip:.6f} {dip_slip:.6f} 0.0\n")
                    
                    # Write corner coordinates
                    for lon_corner, lat_corner, depth_corner in corners:
                        # Convert depth to km if needed
                        if self.depth_unit == 'm':
                            depth_km = depth_corner / 1000.0
                        else:
                            depth_km = depth_corner
                        fout.write(f"{lon_corner:.6f} {lat_corner:.6f} {depth_km:.3f}\n")
            
            print(f"GMT conversion completed: {output_file}")
            return True
            
        except Exception as e:
            print(f"Error writing GMT file: {e}")
            return False
    
    def plot_3d(self, save_fig=False, output_file=None):
        """
        Create 3D visualization of fault slip distribution.
        
        Args:
            save_fig (bool): Whether to save the figure
            output_file (str): Output figure file path
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.slip_data is None:
            print("Error: No data loaded. Call read_slip_file() first.")
            return False
        
        if output_file is None:
            output_file = f"{self.__class__.__name__.lower()}_slip_3d.png"
        
        try:
            fig = plt.figure(figsize=(12, 10))
            ax = fig.add_subplot(111, projection='3d')
            
            max_slip = np.max(self.slip_data['total_slip'])
            min_slip = np.min(self.slip_data['total_slip'])
            all_lons, all_lats, all_depths = [], [], []
            
            n_patches = len(self.slip_data['longitude'])
            print(f"Plotting {n_patches} fault patches...")
            
            for i in range(n_patches):
                # Get patch parameters
                lon = self.slip_data['longitude'][i]
                lat = self.slip_data['latitude'][i]
                depth = self.slip_data['depth'][i]
                length = self.slip_data['length'][i]
                width = self.slip_data['width'][i]
                strike = self.slip_data['strike'][i]
                dip = self.slip_data['dip'][i]
                total_slip = self.slip_data['total_slip'][i]
                
                # pos_s pos_d
                if 'pos_s' in self.slip_data and 'pos_d' in self.slip_data:
                    pos_s = self.slip_data['pos_s'][i]
                    pos_d = self.slip_data['pos_d'][i]
                else:
                    pos_s = 0.0
                    pos_d = 0.0

                # Calculate corner points
                corners = self.calculate_patch_corners(
                    lon, lat, depth, length, width, strike, dip,
                    pos_s=pos_s, pos_d=pos_d
                )
                
                # Extract coordinates for plotting
                lons, lats, depths = zip(*corners)
                
                # Convert depth to km for plotting
                if self.depth_unit == 'm':
                    depths = [d/1000.0 for d in depths] # abs(d)
                else:
                    depths = [d for d in depths]
                
                all_lons.extend(lons)
                all_lats.extend(lats)
                all_depths.extend(depths)
                
                # Create 3D polygon
                vertices = list(zip(lons, lats, depths))
                poly = Poly3DCollection([vertices], alpha=0.8, edgecolor='k', linewidth=0.3)
                
                # Set color based on slip magnitude
                if max_slip > min_slip:
                    color_intensity = (total_slip - min_slip) / (max_slip - min_slip)
                else:
                    color_intensity = 0.5
                poly.set_facecolor(plt.cm.jet(color_intensity))
                ax.add_collection3d(poly)
            
            # Set labels and limits
            ax.set_xlabel('Longitude (°)')
            ax.set_ylabel('Latitude (°)')
            ax.set_zlabel('Depth (km)')
            
            if all_lons and all_lats and all_depths:
                ax.set_xlim(min(all_lons), max(all_lons))
                ax.set_ylim(min(all_lats), max(all_lats))
                ax.set_zlim(max(all_depths), min(all_depths))  # Reverse z-axis
            
            # Add colorbar
            mappable = plt.cm.ScalarMappable(cmap='jet')
            mappable.set_array(self.slip_data['total_slip'])
            cbar = plt.colorbar(mappable, ax=ax, label='Slip (m)', shrink=0.8)
            
            plt.title(f'{self.__class__.__name__} 3D Fault Slip Distribution\nMax slip: {max_slip:.2f} m')
            plt.tight_layout()
            
            if save_fig:
                plt.savefig(output_file, dpi=300, bbox_inches='tight')
                print(f"Figure saved as {output_file}")
            
            plt.show()
            return True
            
        except Exception as e:
            print(f"Error creating 3D plot: {e}")
            return False
    
    def get_slip_stats(self):
        """
        Get basic statistics of the slip distribution.
        
        Returns:
            dict: Dictionary containing slip statistics
        """
        if self.slip_data is None:
            print("Error: No data loaded. Call read_slip_file() first.")
            return None
        
        total_slip = self.slip_data['total_slip']
        
        # Calculate depth range from actual corner points
        all_depths = []
        for i in range(len(self.slip_data['longitude'])):
            corners = self.calculate_patch_corners(
                self.slip_data['longitude'][i], self.slip_data['latitude'][i],
                self.slip_data['depth'][i], self.slip_data['length'][i],
                self.slip_data['width'][i], self.slip_data['strike'][i],
                self.slip_data['dip'][i]
            )
            depths = [abs(corner[2]) for corner in corners]
            all_depths.extend(depths)
        
        # Convert to km if needed
        if self.depth_unit == 'm':
            all_depths = [d/1000.0 for d in all_depths]
        
        dep_min = min(all_depths)
        dep_max = max(all_depths)
        
        stats = {
            'num_patches': len(total_slip),
            'max_slip': np.max(total_slip),
            'min_slip': np.min(total_slip),
            'mean_slip': np.mean(total_slip),
            'depth_range': f"{dep_min:.2f} - {dep_max:.2f} km",
            'converter_type': self.__class__.__name__
        }
        
        return stats

    def calculate_moment_magnitude(self, gmt_file=None, type='rect'):
        print("\nCalculating moment magnitude...")
        from csi.faultpostproc import faultpostproc as faultpp
        lon0 = np.mean(self.slip_data['longitude'])
        lat0 = np.mean(self.slip_data['latitude'])
        utmzone= None
        if type == 'rect':
            from csi.RectangularPatches import RectangularPatches
            fault = RectangularPatches('Fault', lon0=lon0, lat0=lat0)
            gmt_file = gmt_file or f"{self.__class__.__name__.lower()}_slip.gmt"
            fault.readPatchesFromFile(gmt_file, readpatchindex=False)
        else:
            from ..csiExtend.BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches as TriangularPatches
            fault = TriangularPatches('Tri', lon0=lon0, lat0=lat0)
            gmt_file = gmt_file or f"{self.__class__.__name__.lower()}_slip.gmt"
            fault.readPatchesFromFile(gmt_file, readpatchindex=False)

        # Compute the triangle areas, moments, moment tensor, and magnitude
        fault.compute_patch_areas()
        fault_processor = faultpp('Rect', fault, 3.0e10, lon0=lon0, lat0=lat0, utmzone=utmzone, verbose=False)
        fault_processor.computeMoments()
        fault_processor.computeMomentTensor()
        fault_processor.computeMagnitude()

        # Print the moment magnitude
        self.tripproc = fault_processor
        print(f'Seismic moment and magnitude of the fault from {gmt_file} is {fault_processor.Mo:.2e} Nm and {fault_processor.Mw:.2f}, respectively.')
    
    def convert_all(self, gmt_file=None, fig_file=None, fault_type='rect'):
        """
        Complete conversion workflow: read data, write GMT file, and create visualization.
        
        Args:
            gmt_file (str): Output GMT file path
            fig_file (str): Output figure file path
            
        Returns:
            bool: True if all operations successful, False otherwise
        """
        print(f"Reading slip data from: {self.input_file}")
        if not self.read_slip_file():
            return False
        
        print(f"\nSlip distribution statistics:")
        stats = self.get_slip_stats()
        if stats:
            for key, value in stats.items():
                print(f"  {key}: {value}")
        
        print(f"\nConverting to GMT format...")
        if not self.write_gmt_file(gmt_file):
            return False
        
        self.calculate_moment_magnitude(gmt_file=gmt_file, type=fault_type)
        
        print(f"\nCreating 3D visualization...")
        if not self.plot_3d(save_fig=True, output_file=fig_file):
            return False
        
        print(f"\nConversion completed successfully!")
        return True