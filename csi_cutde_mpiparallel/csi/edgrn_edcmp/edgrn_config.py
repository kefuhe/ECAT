import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class EdgrnLayer:
    """
    Single earth model layer definition (EDGRN format)
    """
    depth: float   # [m]
    vp: float      # [m/s]
    vs: float      # [m/s]
    rho: float     # [kg/m^3]

@dataclass
class EDGRNParameters:
    """
    EDGRN parameter set
    """
    obs_depth: float = 0.0  # [m]
    n_distances: int = 201
    min_distance: float = 0.0  # [m]
    max_distance: float = 100000.0  # [m]
    n_depths: int = 40
    min_depth: float = 250.0   # [m]
    max_depth: float = 19750.0 # [m]
    srate: float = 12.0
    output_dir: str = './edgrnfcts/'
    grn_files: Tuple[str, str, str] = ('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    layers: List[EdgrnLayer] = field(default_factory=list)

    def __post_init__(self):
        self.output_dir = self._fix_dir_sep(self.output_dir)
        if not self.layers:
            self.layers = [EdgrnLayer(0.0, 5570.0, 3216.0, 2900.0)]

    @staticmethod
    def _fix_dir_sep(path: str) -> str:
        sep = os.sep
        path = path.replace('/', sep).replace('\\', sep)
        if not path.endswith(sep):
            path += sep
        return path

    def add_layer(self, depth: float, vp: float, vs: float, rho: float):
        self.layers.append(EdgrnLayer(depth, vp, vs, rho))
        self.layers.sort(key=lambda x: x.depth)

    def clear_layers(self):
        self.layers.clear()

    def load_layers_from_dataframe(self, df: pd.DataFrame):
        required = ['depth', 'vp', 'vs', 'rho']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        self.clear_layers()
        for _, row in df.iterrows():
            self.add_layer(float(row['depth']), float(row['vp']), float(row['vs']), float(row['rho']))

    def load_layers_from_file(self, filename: Union[str, Path], file_format: str = 'auto'):
        filepath = Path(filename)
        if file_format == 'auto':
            file_format = filepath.suffix.lower()
            if file_format in ['.xlsx', '.xls']:
                file_format = 'excel'
            elif file_format == '.csv':
                file_format = 'csv'
            elif file_format == '.json':
                file_format = 'json'
            elif file_format in ['.txt', '.dat']:
                file_format = 'txt'
            else:
                raise ValueError(f"Cannot auto-detect format for {filepath}")
        if file_format == 'csv':
            df = pd.read_csv(filename)
        elif file_format == 'excel':
            df = pd.read_excel(filename)
        elif file_format == 'json':
            df = pd.read_json(filename)
        elif file_format == 'txt':
            df = pd.read_csv(filename, delim_whitespace=True, comment='#')
        else:
            raise ValueError(f"Unsupported file format: {file_format}")
        self.load_layers_from_dataframe(df)

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([{
            'depth': lyr.depth,
            'vp': lyr.vp,
            'vs': lyr.vs,
            'rho': lyr.rho
        } for lyr in self.layers])

class EDGRNConfig:
    """
    EDGRN configuration file generator
    """
    def __init__(self, parameters: Optional[EDGRNParameters] = None):
        self.parameters = parameters or EDGRNParameters()

    @staticmethod
    def _fix_dir_sep(path: str) -> str:
        sep = os.sep
        path = path.replace('/', sep).replace('\\', sep)
        if not path.endswith(sep):
            path += sep
        return path

    def generate_config_string(self) -> str:
        # Always fix output_dir and grn_dir before output
        self.parameters.output_dir = self._fix_dir_sep(self.parameters.output_dir)
        if hasattr(self.parameters, 'grn_dir'):
            self.parameters.grn_dir = self._fix_dir_sep(self.parameters.grn_dir)
        lines = []
        lines.extend(self._generate_header())
        lines.extend(self._generate_obs_profile())
        lines.extend(self._generate_wavenumber())
        lines.extend(self._generate_outputs())
        lines.extend(self._generate_model())
        lines.append("#=======================end of input==========================================")
        return '\n'.join(lines)

    def write_config_file(self, filename: Union[str, Path] = "edgrn.inp", verbose=True):
        config_content = self.generate_config_string()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(config_content)
        if verbose:
            print(f"EDGRN configuration written to: {filename}")

    def _generate_header(self) -> List[str]:
        return [
            "#=============================================================================",
            "# This is the input file of FORTRAN77 program 'edgrn' for calculating",
            "# the Green's functions of a layered elastic half-space earth model. All",
            "# results will be stored in the given directory and provide the necessary",
            "# data base for the program 'edcmp' for computing elastic deformations",
            "# (3 displacement components, 6 strain/stress components and 2 vertical tilt",
            "# components) induced by a general dislocation source.",
            "#",
            "# For questions please contact with e-mail to 'wang@gfz-potsdam.de'.",
            "#",
            "# First implemented in May, 1997",
            "# Last modified: Nov, 2001",
            "#",
            "# by Rongjiang Wang",
            "# GeoForschungsZetrum Potsdam, Telegrafenberg, 14473 Potsdam, Germany",
            "#",
            "# For questions and suggestions please send e-mails to wang@gfz-potsdam.de",
            "#------------------------------------------------------------------------------",
            "#",
            "# PARAMETERS FOR THE OBSERVATION PROFILE",
            "# =======================================",
            "# 1. the uniform depth of the observation points [m]",
            "# 2. number of the equidistant radial distances (max. = nrmax in edgglobal.h),",
            "#    the start and end of the distances [m]",
            "# 3. number of the equidistant source depths (max. = nzsmax in edgglobal.h),",
            "#    the start and end of the source depths [m]",
            "#",
            "#    If possible, please choose the observation depth either significantly",
            "#    different from the source depths or identical with one of them.",
            "#",
            "#    The 2D distance and depth grids defined here should be necessarily large",
            "#    and dense enough for the discretisation of the real source-observation",
            "#    configuration to be considered later.",
            "#",
            "#    r1,r2 = minimum and maximum horizontal source-observation distances",
            "#    z1,z2 = minimum and maximum source depths",
            "#",
            "#------------------------------------------------------------------------------"
        ]

    def _generate_obs_profile(self) -> List[str]:
        p = self.parameters
        return [
            f"      {format(p.obs_depth, '.5e').replace('e', 'd')}                           |dble: obs_depth;",
            f" {p.n_distances:3d}  {format(p.min_distance, '.5e').replace('e', 'd')}   {format(p.max_distance, '.5e').replace('e', 'd')}              |int: nr; dble: r1, r2;",
            f"  {p.n_depths:2d}  {format(p.min_depth, '.5e').replace('e', 'd')}   {format(p.max_depth, '.5e').replace('e', 'd')}              |int: nzs; dble: zs1, zs2;",
            "#------------------------------------------------------------------------------",
            "#",
            "#	WAVENUMBER INTEGRATION PARAMETERS",
            "#	=================================",
            "# 1. sampling rate for wavenumber integration (the ratio between the Nyquist",
            "#    wavenumber and the really used wavenumber sample; the suggested value is",
            "#    10-128: the larger this value is chosen, the more accurate are the results",
            "#    but also the more computation time will be required)",
            "#------------------------------------------------------------------------------"
        ]

    def _generate_wavenumber(self) -> List[str]:
        p = self.parameters
        return [
            f" {p.srate:5.1f}                            |dble: srate;",
            "#------------------------------------------------------------------------------",
            "#",
            "#	OUTPUT FILES",
            "#	============",
            "#",
            "# 1. output directory, the three file names for fundamental Green's functions",
            "#    Note that all file or directory names should not be longer than 80",
            "#    characters. Directories must be ended by / (unix) or \\ (dos)!",
            "#------------------------------------------------------------------------------"
        ]

    def _generate_outputs(self) -> List[str]:
        p = self.parameters
        return [
            f" '{p.output_dir}'  '{p.grn_files[0]}'  '{p.grn_files[1]}'  '{p.grn_files[2]}'  |char: outputs,grnfile(3);",
            "#------------------------------------------------------------------------------",
            "#",
            "#	MULTILAYERED MODEL PARAMETERS",
            "#	=============================",
            "# The interfaces at which the elastic parameters are continuous, the surface",
            "# and the upper boundary of the half-space are all defined by a single data",
            "# line; The interfaces, at which the elastic parameters are discontinuous,",
            "# are all defined by two data lines. This input format would also be needed for",
            "# a graphic plot of the layered model.",
            "#",
            "# Layers which have different upper and lower parameter values, will be treated",
            "# as layers with a constant gradient, and will be discretised by a number of",
            "# homogeneous sublayers. Errors due to the discretisation are limited within",
            "# about 5%.",
            "#",
            "# 1. total number of the data lines (max. = lmax in edgglobal.h)",
            "# 2. table for the layered model",
            "#------------------------------------------------------------------------------"
        ]

    def _generate_model(self) -> List[str]:
        p = self.parameters
        lines = [
            f"   {len(p.layers):1d}                               |int: no_model_lines;",
            "#------------------------------------------------------------------------------",
            "#    no  depth[m]       vp[m/s]         vs[m/s]        ro[kg/m^3]",
            "#------------------------------------------------------------------------------"
        ]
        for i, lyr in enumerate(p.layers, 1):
            lines.append(
                f" {i:2d}  {format(lyr.depth, '.5e').replace('e', 'd'):>12}  {format(lyr.vp, '.6e').replace('e', 'd'):>12}  {format(lyr.vs, '.6e').replace('e', 'd'):>12}  {format(lyr.rho, '.6e').replace('e', 'd'):>12}"
            )
        return lines

# 示例用法
if __name__ == "__main__":
    params = EDGRNParameters()
    params.obs_depth = 0.0
    params.n_distances = 201
    params.min_distance = 0.0
    params.max_distance = 100000.0
    params.n_depths = 40
    params.min_depth = 250.0
    params.max_depth = 19750.0
    params.srate = 12.0
    params.output_dir = './'
    params.grn_files = ('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    params.layers = [
        EdgrnLayer(0.0, 5570.0, 3216.0, 2900.0)
    ]

    config = EDGRNConfig(params)
    config.write_config_file("edgrn.inp", verbose=True)