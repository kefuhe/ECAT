from .readDirectProjection2csisar import DirectProjectionSarReader
from .sar_conventions import GmtsarConfig


class GmtsarReader(DirectProjectionSarReader):
    """
    Reader for GMTSAR-style NetCDF/GRD products with ENU projection grids.

    The reader follows the same ECAT SAR convention contract as the GAMMA and
    GeoTIFF readers. GMTSAR-specific code is limited to mode/preset mapping and
    file reading through the direct-projection base class.
    """

    config_cls = GmtsarConfig
    mode_presets = {
        "unwrapped_phase": "gmtsar_unwrapped_phase",
        "phase_los": "gmtsar_unwrapped_phase",
        "los": "gmtsar_los_displacement",
        "los_displacement": "gmtsar_los_displacement",
        "range": "gmtsar_range_offset",
        "range_offset": "gmtsar_range_offset",
        "az": "gmtsar_azimuth_offset",
        "azimuth": "gmtsar_azimuth_offset",
        "azimuth_offset": "gmtsar_azimuth_offset",
    }
