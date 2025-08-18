import os
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
from dataclasses import dataclass, field

@dataclass
class RectangularSource:
    """
    Definition of a single rectangular source (EDCMP format)
    """
    source_id: int
    slip: float
    xs: float
    ys: float
    zs: float
    length: float
    width: float
    strike: float
    dip: float
    rake: float

@dataclass
class EDCMPParameters:
    """
    EDCMP parameter set
    """
    # Observation point settings
    ixyr: int = 2
    observation_points: Union[List, Dict] = field(default_factory=dict)

    # Output settings
    output_dir: str = './'
    output_flags: Tuple[int, int, int, int] = (1, 0, 0, 0)
    output_files: Tuple[str, str, str, str] = (
        'edcmp.disp', 'edcmp.strn', 'edcmp.strss', 'edcmp.tilt'
    )

    # Source parameters
    sources: List[RectangularSource] = field(default_factory=list)

    # Earth model
    layered_model: bool = True
    grn_dir: str = './edgrnfcts/'
    grn_files: Tuple[str, str, str] = ('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    # Homogeneous half-space parameters
    zrec: float = 0.0
    lambda_: float = 3.0e10
    mu: float = 3.0e10

    def __post_init__(self):
        self.output_dir = self._fix_dir_sep(self.output_dir)
        self.grn_dir = self._fix_dir_sep(self.grn_dir)

    @staticmethod
    def _fix_dir_sep(path: str) -> str:
        sep = os.sep
        path = path.replace('/', sep).replace('\\', sep)
        if not path.endswith(sep):
            path += sep
        return path

    def set_rectangular_observation_array(self, nxr: int, xr1: float, xr2: float,
                                          nyr: int, yr1: float, yr2: float):
        """
        Set a rectangular 2D observation array.
        """
        self.ixyr = 2
        self.observation_points = {
            'nxr': nxr, 'xr1': xr1, 'xr2': xr2,
            'nyr': nyr, 'yr1': yr1, 'yr2': yr2
        }

    def set_profile_observation_array(self, nr: int, xr1: float, yr1: float, xr2: float, yr2: float):
        """
        Set a 1D profile observation array.
        """
        self.ixyr = 1
        self.observation_points = {
            'nr': nr,
            'xr1': xr1, 'yr1': yr1,
            'xr2': xr2, 'yr2': yr2
        }

    def set_irregular_observation_points(self, coordinates: List[Tuple[float, float]]):
        """
        Set irregular observation points.
        """
        self.ixyr = 0
        self.observation_points = {
            'nr': len(coordinates),
            'coordinates': coordinates
        }

    def add_source(self, source: RectangularSource):
        """
        Add a rectangular source to the source list.
        """
        self.sources.append(source)

class EDCMPConfig:
    """
    EDCMP configuration file generator, fully preserves template comments.
    """
    def __init__(self, parameters: Optional[EDCMPParameters] = None):
        """
        Initialize EDCMPConfig with optional EDCMPParameters.
        """
        self.parameters = parameters or EDCMPParameters()

    def _fix_dir_sep(self, path: str) -> str:
        """
        Fix directory separator for the current operating system.
        """
        sep = os.sep
        path = path.replace('/', sep).replace('\\', sep)
        if not path.endswith(sep):
            path += sep
        return path
    
    def generate_config_string(self) -> str:
        """
        Generate the full EDCMP configuration file content as a string.
        Ensures output_dir and grn_dir are always fixed for the current system.
        """
        self.parameters.output_dir = self._fix_dir_sep(self.parameters.output_dir)
        self.parameters.grn_dir = self._fix_dir_sep(self.parameters.grn_dir)
        lines = []
        lines.extend(self._generate_header())
        lines.extend(self._generate_observation_config())
        lines.extend(self._generate_output_config())
        lines.extend(self._generate_source_config())
        lines.extend(self._generate_earth_model_config())
        lines.append("#================================end of input===================================")
        return '\n'.join(lines)

    def write_config_file(self, filename: Union[str, Path] = "edcmp.inp", verbose=True):
        """
        Write the EDCMP configuration to a file.
        """
        config_content = self.generate_config_string()
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(config_content)
        if verbose:
            print(f"EDCMP configuration written to: {filename}")

    def _generate_header(self) -> List[str]:
        """
        Generate the header and template comments for the EDCMP configuration file.
        """
        return [
            "#===============================================================================",
            "# This is the input file of FORTRAN77 program \"edcmp\" for calculating",
            "# earthquakes' static deformations (3 displacement components, 6 strain/stress",
            "# components and 2 vertical tilt components) based on the dislocation theory.",
            "# The earth model used is either a homogeneous or multi-layered, isotropic and",
            "# elastic half-space. The earthquke source is represented by an arbitrary number",
            "# of rectangular dislocation planes.",
            "#",
            "# Note that the Cartesian coordinate system is used and the seismological",
            "# convention is adopted, that is, x is northward, y is eastward, and z is",
            "# downward.",
            "#",
            "# First implemented in Potsdam, Feb, 1999",
            "# Last modified: Potsdam, Nov, 2001",
            "#",
            "# by",
            "# Rongjiang Wang, Frank Roth, & Francisco Lorenzo",
            "# GeoForschungsZentrum Potsdam, Telegrafenberg, 14473 Potsdam, Germany",
            "#",
            "# For questions and suggestions please send e-mails to wang@gfz-potsdam.de",
            "#===============================================================================",
            "# OBSERVATION ARRAY",
            "# =================",
            "# 1. switch for irregular positions (0) or a 1D profile (1)",
            "#    or a rectangular 2D observation array (2): ixyr",
            "#",
            "#    IF (the switch = 0 for irregular observation positions) THEN",
            "#    ",
            "# 2. number of positions: nr",
            "# 3. coordinates of the observations: (xr(i),yr(i)),i=1,nr",
            "#",
            "#    ELSE IF (the switch = 1 for a 1D profile) THEN",
            "#",
            "# 2. number of position samples of the profile: nr",
            "# 3. the start and end positions: (xr1,yr1), (xr2,yr2)",
            "#",
            "#    ELSE IF (the switch = 2 for rectanglular 2D observation array) THEN",
            "#",
            "# 2. number of xr samples, start and end values [m]: nxr, xr1,xr2",
            "# 3. number of yr samples, start and end values [m]: nyr, yr1,yr2",
            "#",
            "#    Note that the total number of observation positions (nr or nxr*nyr)",
            "#    should be <= NRECMAX (see edcglobal.h)!",
            "#===============================================================================",
            "#  0",
            "#  6",
            "#   ( 0.0d+00,-10.0d+03), ( 0.0d+00,-3.0d+03), ( 0.0d+00, -1.5d+03),",
            "#   ( 0.0d+00,  1.5d+03), ( 0.0d+00, 3.0d+03), ( 0.0d+00, 10.0d+03)"
            "#",
            "#  1",
            "#  201",
            "#  (-50.0d+03,-15.0d+00), (50.0d+03,-15.0d+00)",
            "#",
            "#  2",
            "#  51  -35.00d+03   15.00d+03",
            "#  51  -25.00d+03   25.00d+03"
        ]

    def _generate_observation_config(self) -> List[str]:
        """
        Generate the observation array configuration section.
        """
        p = self.parameters
        lines = []
        if p.ixyr == 0:
            lines.append(f"  {p.ixyr}")
            obs = p.observation_points
            lines.append(f"  {obs['nr']}")
            for i in range(0, obs['nr'], 3):
                coord_line = ""
                for j in range(min(3, obs['nr'] - i)):
                    x, y = obs['coordinates'][i + j]
                    coord_line += f"   ({format(x, '.6e').replace('e', 'd')},{format(y, '.6e').replace('e', 'd')}),"
                lines.append(coord_line)
            lines[-1] = lines[-1].rstrip(',')
        elif p.ixyr == 1:
            lines.append(f"  {p.ixyr}")
            obs = p.observation_points
            lines.append(f"  {obs['nr']}")
            lines.append(
                f"  ({format(obs['xr1'], '.6e').replace('e', 'd')},{format(obs['yr1'], '.6e').replace('e', 'd')}), "
                f"({format(obs['xr2'], '.6e').replace('e', 'd')},{format(obs['yr2'], '.6e').replace('e', 'd')})"
            )
        elif p.ixyr == 2:
            lines.append(f"  {p.ixyr}")
            obs = p.observation_points
            lines.append(
                f" {obs['nxr']:2d}  {format(obs['xr1'], '.6e').replace('e', 'd')}   {format(obs['xr2'], '.6e').replace('e', 'd')}"
            )
            lines.append(
                f" {obs['nyr']:2d}  {format(obs['yr1'], '.6e').replace('e', 'd')}   {format(obs['yr2'], '.6e').replace('e', 'd')}"
            )
        return lines

    def _generate_output_config(self) -> List[str]:
        """
        Generate the output configuration section.
        """
        p = self.parameters
        return [
            "#===============================================================================",
            "# OUTPUTS",
            "# =======",
            "# 1. output directory in char format: outdir",
            "# 2. select the desired outputs (1/0 = yes/no)",
            "# 3. the file names in char format for displacement vector, strain tensor,",
            "#    stress tensor, vertical tilts, and los:",
            "#    dispfile, strainfile, stressfile, tiltfile, losfile",
            "#",
            "#    Note that all file or directory names should not be longer than 80",
            "#    characters. Directories must be ended by / (unix) or \\ (dos)!",
            "#===============================================================================",
            f"  '{p.output_dir}'",
            "    " + "              ".join(str(flag) for flag in p.output_flags),
            "  " + "   ".join(f"'{fname}'" for fname in p.output_files)
        ]

    def _generate_source_config(self) -> List[str]:
        """
        Generate the rectangular source configuration section.
        """
        p = self.parameters
        lines = [
            "#===============================================================================",
            "# RECTANGLAR DISLOCATION SOURCES",
            "# ===============================",
            "# 1. number of the source rectangles: ns (<= NSMAX in edcglobal.h)",
            "# 2. the 6 parameters for the 1. source rectangle:",
            "#    Slip [m],",
            "#    coordinates of the upper reference point for strike (xs, ys, zs) [m],",
            "#    length (strike direction) [m], and width (dip direction) [m],",
            "#    strike [deg], dip [deg], and rake [deg];",
            "# 3. ... for the 2. source ...",
            "# ...",
            "#                   N",
            "#                  /",
            "#                 /| strike",
            "#         Ref:-> @------------------------",
            "#                |\\        p .            \\ W",
            "#                :-\\      i .              \\ i",
            "#                |  \\    l .                \\ d",
            "#                :90 \\  S .                  \\ t",
            "#                |-dip\\  .                    \\ h",
            "#                :     \\. | rake               \\ ",
            "#                Z      -------------------------",
            "#                              L e n g t h",
            "#",
            "#    Note that if one of the parameters length and width = 0, then a line source",
            "#    will be considered and the dislocation parameter Slip has the unit m^2; if",
            "#    both length and width = 0, then a point source will be considered and the",
            "#    Slip has the unit m^3.",
            "#===============================================================================",
            f"  {len(p.sources)}",
            "#         coord. origin:",
            "#-------------------------------------------------------------------------------",
            "# no  Slip   xs        ys       zs        length    width   strike   dip  rake",
            "#-------------------------------------------------------------------------------"
        ]
        for src in p.sources:
            lines.append(
                f" {src.source_id:2d}  {format(src.slip, '.4f')} {format(src.xs, '.6e').replace('e', 'd')}  "
                f"{format(src.ys, '.6e').replace('e', 'd')}  {format(src.zs, '.6e').replace('e', 'd')}   "
                f"{format(src.length, '.6e').replace('e', 'd')}  {format(src.width, '.6e').replace('e', 'd')}   "
                f"{src.strike:6.1f}  {src.dip:5.1f}  {src.rake:6.1f}"
            )
        return lines

    def _generate_earth_model_config(self) -> List[str]:
        """
        Generate the earth model configuration section.
        """
        p = self.parameters
        lines = [
            "#===============================================================================",
            "# If the earth model used is a layered half-space, then the numerical Green's",
            "# function approach is applied. The Green's functions should have been prepared",
            "# with the program \"edgrn\" before the program \"edcmp\" is started. In this case,",
            "# the following input data give the addresses where the Green's functions have",
            "# been stored and the grid side to be used for the automatic discretization",
            "# of the finite rectangular sources.",
            "#",
            "# If the earth model used is a homogeneous half-space, then the analytical",
            "# method of Okada (1992) is applied. In this case, the Green's functions are",
            "# not needed, and the following input data give the shear modulus and the",
            "# Poisson ratio of the model.",
            "#===============================================================================",
            "# CHOICE OF EARTH MODEL",
            "# =====================",
            "# 1. switch for layered (1) or homogeneous (0) model",
            "#",
            "#    IF (layered model) THEN",
            "#",
            "# 2. directory of the Green's functions and the three files for the",
            "#    fundamental Green's functions: grndir, grnfiles(3);",
            "#",
            "#    Note that all file or directory names should not be longer than 80",
            "#    characters. Directories must be ended by / (unix) or \\ (dos)!",
            "#",
            "#    ELSE (homogeneous model) THEN",
            "#",
            "# 2. the observation depth, the two Lame constants parameters of the homogeneous",
            "#    model: zrec [m], lambda [Pa], mu [Pa]",
            "#===============================================================================",
            "#  1",
            "#  '../edgrnfcts_bam/'  'edgrnhs.ss'  'edgrnhs.ds'  'edgrnhs.cl'",
            "#  0",
            "#  0.00d+00  3.0000E+10  3.000E+10"
        ]
        if p.layered_model:
            lines.append(f"  1")
            lines.append(f"  '{p.grn_dir}'  '{p.grn_files[0]}'  '{p.grn_files[1]}'  '{p.grn_files[2]}'")
        else:
            lines.append(f"  0")
            lines.append(f"  {p.zrec:.2f}d+00  {p.lambda_:.4E}  {p.mu:.3E}")
        return lines

# Example usage
if __name__ == "__main__":
    params = EDCMPParameters()
    params.set_rectangular_observation_array(51, -35000, 15000, 51, -25000, 25000)
    params.output_dir = './'
    params.output_flags = (1, 0, 0, 0)
    params.output_files = ('bamhs_2d_wang.disp', 'bamhs.strn', 'bamhs.strss', 'bamhs.tilt')
    params.layered_model = False
    params.zrec = 0.0
    params.lambda_ = 3.0e10
    params.mu = 3.0e10
    params.sources = [
        RectangularSource(
            source_id=1, slip=2.5, xs=0.0, ys=0.0, zs=0.2,
            length=11e3, width=10e3, strike=174.0, dip=88.0, rake=178.0
        )
    ]
    config = EDCMPConfig(params)
    config.write_config_file("edcmp.inp")