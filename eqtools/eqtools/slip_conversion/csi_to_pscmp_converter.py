"""
CSI to PSCMP format converter

This module provides conversion from CSI RectangularPatches to PSCMP format
with better code organization and integration with the slip conversion framework.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List
from .base_converter import BaseSlipConverter
from .geometry_utils import ReferencePoint, FaultGeometry
from pyproj import Proj


class CSIToPSCMPConverter(BaseSlipConverter):
    """
    CSI RectangularPatches to PSCMP format converter.
    
    Converts CSI fault objects to PSCMP .inp format with proper handling
    of coordinate systems and slip conventions.
    """
    
    def __init__(self, csi_fault=None, slip_file=None):
        """
        Initialize converter with CSI fault object.
        
        Parameters:
        -----------
        csi_fault : csi.RectangularPatches or None
            CSI fault object to convert
        slip_file : str or None
            For compatibility with base class
        """
        super().__init__(slip_file)
        
        # Set converter-specific parameters
        self.reference_point = ReferencePoint.CENTER
        self.depth_unit = 'km'
        self.length_unit = 'km'
        self.width_unit = 'km'
        self.converter_type = 'pyproj'
        
        # CSI-specific attributes
        self.csi_fault = csi_fault
        self.df = None
        self.proj = None
        self.utm_zone = None
        self.fault_geometry = None
        
        # PSCMP format template
        self.pscmp_patch_format = (
            '{0:4d} {1:9.4f} {2:9.4f} {3:6.1f} {4:6.1f} {5:6.1f} {6:6.1f} '
            '{7:6.1f} {8:3d} {9:3d} {10:6.1f}\n'
            '    {11:9.4f} {12:9.4f} {13:9.3f} {14:9.3f} {15:9.3f}'
        )
    
    def _setup_projection(self):
        """
        Setup UTM projection and fault geometry calculator.
        Implementation of abstract method from BaseSlipConverter.
        """
        if self.df is not None and len(self.df) > 0:
            lat_ref, lon_ref = self.df.iloc[0]['o_lat'], self.df.iloc[0]['o_lon']
            self.utm_zone = int((lon_ref + 180) // 6) + 1
            self.proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
            self.fault_geometry = FaultGeometry(self.proj)
            
            print(f"Using UTM zone: {self.utm_zone}")
            print(f"Reference point: ({lat_ref:.4f}°, {lon_ref:.4f}°)")
        elif self.csi_fault is not None:
            # Use CSI fault's coordinate system if available
            if hasattr(self.csi_fault, 'utmzone') and self.csi_fault.utmzone is not None:
                self.utm_zone = self.csi_fault.utmzone
                self.proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
                self.fault_geometry = FaultGeometry(self.proj)
                print(f"Using CSI fault's UTM zone: {self.utm_zone}")
            else:
                # Estimate UTM zone from first patch
                if hasattr(self.csi_fault, 'patch') and len(self.csi_fault.patch) > 0:
                    first_patch = self.csi_fault.patch[0]
                    if hasattr(first_patch, 'lon'):
                        lon_ref = first_patch.lon
                        self.utm_zone = int((lon_ref + 180) // 6) + 1
                        self.proj = Proj(proj='utm', zone=self.utm_zone, ellps='WGS84')
                        self.fault_geometry = FaultGeometry(self.proj)
                        print(f"Estimated UTM zone: {self.utm_zone}")
    
    def load_csi_fault(self, csi_fault):
        """
        Load CSI fault object.
        
        Parameters:
        -----------
        csi_fault : csi.RectangularPatches
            CSI rectangular fault object
        """
        self.csi_fault = csi_fault
        self._convert_csi_to_dataframe()
        self._setup_slip_data()
        self._setup_projection()
    
    def read_slip_file(self):
        """
        Implementation of base class method.
        For CSI converter, data is loaded via load_csi_fault().
        """
        if self.csi_fault is None:
            raise ValueError("No CSI fault loaded. Use load_csi_fault() first.")
        
        self._convert_csi_to_dataframe()
        self._setup_slip_data()
        self._setup_projection()
        return True
    
    def _convert_csi_to_dataframe(self):
        """
        Convert CSI fault object to standardized DataFrame format.
        """
        if self.csi_fault is None:
            raise ValueError("No CSI fault object loaded")
        
        n_patches = self.csi_fault.slip.shape[0]
        fault_data = []
        
        for i in range(n_patches):
            # Get patch geometry from CSI
            patch_data = self._extract_patch_geometry(i)
            fault_data.append(patch_data)
        
        self.df = pd.DataFrame(fault_data)
        print(f"Converted {len(self.df)} CSI patches to standard format")
    
    def _extract_patch_geometry(self, patch_idx: int, use_center: bool = False) -> dict:
        """
        Extract geometry and slip data for a single patch.
        
        Parameters:
        -----------
        patch_idx : int
            Patch index
        use_center : bool
            If True, use patch center as reference point
            
        Returns:
        --------
        dict
            Patch data dictionary
        """
        # Get patch geometry from CSI
        x, y, z, width, length, strike_rad, dip_rad = self.csi_fault.getpatchgeometry(
            patch_idx, center=use_center
        )
        
        # Convert to degrees
        strike_deg = np.rad2deg(strike_rad)
        dip_deg = np.rad2deg(dip_rad)
        
        # Convert coordinates to lat/lon
        lon, lat = self.csi_fault.xy2ll(x, y)
        
        # Get slip values (CSI convention)
        slip_strike, slip_dip, opening = self.csi_fault.slip[patch_idx, :]
        
        # Calculate position within patch
        if use_center:
            pos_s, pos_d = 0.0, 0.0
        else:
            pos_s, pos_d = length / 2.0, width / 2.0
        
        # Convert slip from CSI to PSCMP convention
        # CSI: positive dip-slip is reverse (up-dip)
        # PSCMP: positive dip-slip is normal (down-dip)
        slip_downdip = -slip_dip
        
        return {
            'fault_id': patch_idx + 1,
            'patch_id': f"{patch_idx + 1}_1",
            'o_lat': float(lat),
            'o_lon': float(lon),
            'o_depth': float(z),
            'length': float(length),
            'width': float(width),
            'patch_length': float(length),
            'patch_width': float(width),
            'strike': float(strike_deg),
            'dip': float(dip_deg),
            'slip_strike': float(slip_strike),
            'slip_downdip': float(slip_downdip),
            'opening': float(opening),
            'pos_s': float(pos_s),
            'pos_d': float(pos_d),
            'np_st': 1,
            'np_di': 1,
            'start_time': 0.0
        }
    
    def _setup_slip_data(self):
        """
        Setup slip_data dictionary for base class compatibility.
        """
        if self.df is None:
            raise ValueError("No data converted. Call _convert_csi_to_dataframe() first.")
        
        self.slip_data = {
            'longitude': self.df['o_lon'].values,
            'latitude': self.df['o_lat'].values,
            'depth': self.df['o_depth'].values,
            'length': self.df['patch_length'].values,
            'width': self.df['patch_width'].values,
            'strike': self.df['strike'].values,
            'dip': self.df['dip'].values,
            'strike_slip': self.df['slip_strike'].values,
            'dip_slip': -self.df['slip_downdip'].values,  # Convert back to CSI convention for base class
            'total_slip': np.sqrt(self.df['slip_strike'].values**2 + self.df['slip_downdip'].values**2),
            'pos_s': self.df['pos_s'].values,
            'pos_d': self.df['pos_d'].values,
            'opening': self.df['opening'].values
        }
    
    def convert_single_patch(self, patch_idx: int, use_center: bool = False, 
                           start_time: float = 0.0, custom_slip: Optional[List] = None) -> str:
        """
        Convert a single CSI patch to PSCMP format string.
        
        Parameters:
        -----------
        patch_idx : int
            Index of patch to convert
        use_center : bool
            Use patch center as reference point
        start_time : float
            Start time for PSCMP
        custom_slip : list or None
            Custom slip values [strike_slip, dip_slip, opening]
            
        Returns:
        --------
        str
            PSCMP format string for the patch
        """
        if self.csi_fault is None:
            raise ValueError("No CSI fault loaded")
        
        # Extract patch data
        patch_data = self._extract_patch_geometry(patch_idx, use_center)
        
        # Use custom slip if provided
        if custom_slip is not None:
            custom_slip = np.asarray(custom_slip)
            patch_data['slip_strike'] = float(custom_slip[0])
            patch_data['slip_downdip'] = float(-custom_slip[1])  # Convert to PSCMP convention
            patch_data['opening'] = float(custom_slip[2]) if len(custom_slip) > 2 else 0.0
        
        # Update start time
        patch_data['start_time'] = float(start_time)
        
        # Format PSCMP string
        pscmp_string = self.pscmp_patch_format.format(
            patch_data['fault_id'],     # n
            patch_data['o_lat'],        # O_lat
            patch_data['o_lon'],        # O_lon
            patch_data['o_depth'],      # O_depth
            patch_data['length'],       # length
            patch_data['width'],        # width
            patch_data['strike'],       # strike
            patch_data['dip'],          # dip
            patch_data['np_st'],        # np_st
            patch_data['np_di'],        # np_di
            patch_data['start_time'],   # start_time
            patch_data['pos_s'],        # pos_s
            patch_data['pos_d'],        # pos_d
            patch_data['slip_strike'],  # slip_strike
            patch_data['slip_downdip'], # slip_downdip
            patch_data['opening']       # opening
        )
        
        return pscmp_string
    
    def convert_to_pscmp_format(self, output_file: Optional[str] = None, 
                              use_center: bool = False, start_time: float = 0.0,
                              custom_slip: Optional[Union[List, np.ndarray]] = None) -> List[str]:
        """
        Convert all CSI patches to PSCMP format.
        
        Parameters:
        -----------
        output_file : str or None
            Output filename. If None, returns string list
        use_center : bool
            Use patch centers as reference points
        start_time : float
            Start time for all patches
        custom_slip : array-like or None
            Custom slip values. Shape: (n_patches, 3) or (1, 3) or (3,)
            
        Returns:
        --------
        list
            List of PSCMP format strings
        """
        if self.csi_fault is None:
            raise ValueError("No CSI fault loaded")
        
        n_patches = self.csi_fault.slip.shape[0]
        
        # Handle custom slip
        if custom_slip is not None:
            custom_slip = np.asarray(custom_slip)
            if custom_slip.ndim == 1:
                custom_slip = np.tile(custom_slip, (n_patches, 1))
            elif custom_slip.shape[0] == 1:
                custom_slip = np.tile(custom_slip, (n_patches, 1))
        
        # Convert all patches
        pscmp_strings = []
        for i in range(n_patches):
            slip = custom_slip[i] if custom_slip is not None else None
            pscmp_string = self.convert_single_patch(
                i, use_center=use_center, start_time=start_time, custom_slip=slip
            )
            pscmp_strings.append(pscmp_string)
        
        # Save to file if requested
        if output_file is not None:
            self._save_pscmp_patches(pscmp_strings, output_file)
        
        return pscmp_strings
    
    def _save_pscmp_patches(self, pscmp_strings: List[str], output_file: str):
        """
        Save PSCMP format strings to file.
        
        Parameters:
        -----------
        pscmp_strings : list
            List of PSCMP format strings
        output_file : str
            Output filename
        """
        with open(output_file, 'w') as f:
            f.write("# PSCMP fault patches converted from CSI\n")
            f.write("# Format: n O_lat O_lon O_depth length width strike dip np_st np_di start_time\n")
            f.write("#         pos_s pos_d slip_strike slip_downdip opening\n")
            f.write("#\n")
            for pscmp_string in pscmp_strings:
                f.write(pscmp_string + '\n')
        
        print(f"PSCMP patches saved to: {output_file}")
    
    def save_pscmp_inp_format(self, output_file: str):
        """
        Save fault data in PSCMP .inp format with proper header.
        
        Parameters:
        -----------
        output_file : str
            Output .inp filename
        """
        if self.df is None:
            raise ValueError("No data loaded. Call load_csi_fault() first.")
        
        pscmp_strings = self.convert_to_pscmp_format()
        
        with open(output_file, 'w') as f:
            f.write("#===============================================================================\n")
            f.write("# n_faults\n")
            f.write("#-------------------------------------------------------------------------------\n")
            f.write(f"  {len(self.df)}\n")
            f.write("#-------------------------------------------------------------------------------\n")
            f.write("# n   O_lat   O_lon   O_depth length  width strike dip   np_st np_di start_time\n")
            f.write("# [-] [deg]   [deg]   [km]    [km]     [km] [deg]  [deg] [-]   [-]   [day]\n")
            f.write("#     pos_s   pos_d   slp_stk slp_ddip open\n")
            f.write("#     [km]    [km]    [m]     [m]      [m]\n")
            f.write("#-------------------------------------------------------------------------------\n")
            
            for pscmp_string in pscmp_strings:
                f.write(pscmp_string + '\n')
        
        print(f"PSCMP .inp format saved: {output_file}")
    
    def create_pscmp_config(self, template_name: str = 'simple', **kwargs):
        """
        Create PSCMP configuration from CSI fault.
        
        Parameters:
        -----------
        template_name : str
            PSCMP configuration template
        **kwargs : dict
            Additional PSCMP parameters
            
        Returns:
        --------
        PSCMPConfig
            Configured PSCMP object
        """
        if self.slip_data is None:
            self.read_slip_file()
        
        from ..csiExtend.psgrn_pscmp.pscmp_config import PSCMPConfig
        config = PSCMPConfig.from_csi_converter(self, template_name, **kwargs)
        
        return config


def csi_to_pscmp(csi_fault, output_file: str = 'pscmp_patches.inp', 
                use_center: bool = False, start_time: float = 0.0,
                custom_slip: Optional[Union[List, np.ndarray]] = None):
    """
    Convenience function to convert CSI fault to PSCMP format.
    
    Parameters:
    -----------
    csi_fault : csi.RectangularPatches
        CSI fault object
    output_file : str
        Output filename
    use_center : bool
        Use patch centers as reference points
    start_time : float
        Start time for all patches
    custom_slip : array-like or None
        Custom slip values
        
    Returns:
    --------
    CSIToPSCMPConverter
        Converter object for further operations
    """
    converter = CSIToPSCMPConverter()
    converter.load_csi_fault(csi_fault)
    converter.convert_to_pscmp_format(
        output_file=output_file,
        use_center=use_center,
        start_time=start_time,
        custom_slip=custom_slip
    )
    
    return converter


# Example usage and testing
if __name__ == "__main__":
    # This would be used for testing with actual CSI objects
    pass


# Example usage and testing
if __name__ == "__main__":
    # 示例1：基本转换
    from csi import RectangularPatches
    from eqtools.slip_conversion.csi_to_pscmp_converter import CSIToPSCMPConverter, csi_to_pscmp
    
    # 假设你有一个CSI fault对象
    csi_fault = RectangularPatches('my_fault')
    # ... 设置fault几何和滑动 ...
    
    # 方法1：使用便捷函数
    converter = csi_to_pscmp(
        csi_fault, 
        output_file='my_fault_pscmp.inp',
        use_center=True,
        custom_slip=[1.0, 0.0, 0.0]  # 纯走滑
    )
    
    # 方法2：使用转换器类
    converter = CSIToPSCMPConverter()
    converter.load_csi_fault(csi_fault)
    
    # 转换单个patch
    patch_string = converter.convert_single_patch(0, use_center=True)
    print(patch_string)
    
    # 转换所有patches
    pscmp_strings = converter.convert_to_pscmp_format(
        output_file='all_patches.inp',
        use_center=False
    )
    
    # 创建完整的PSCMP配置
    pscmp_config = converter.create_pscmp_config(
        template_name='insar',
        output_dir='./pscmp_results/'
    )
    pscmp_config.write_config_file('complete_pscmp.dat')
    
    # 示例3：与现有转换器框架集成
    # 可以像其他转换器一样使用
    converter.save_gmt_format('fault_patches.gmt')
    converter.save_csv_format('fault_data.csv')