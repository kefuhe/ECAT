from ruamel.yaml import YAML
import os

def generate_nonlinear_config(output_path):
    """
    Generate a default configuration file for nonlinear Bayesian inversion with comments.

    Parameters:
    output_path (str): The output path for the configuration file.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Define the configuration with comments
    config = yaml.load("""
# ----------- General Parameters ----------- #
# General settings for the Bayesian inversion process
nchains: 100  # Number of chains for BayesianMultiFaultsInversion
chain_length: 50  # Length of each chain for BayesianMultiFaultsInversion
smc_tempering:
  target_cov: 1.0  # Incremental-weight COV target; standard value
  max_delta_beta: 0.5  # Maximum beta increment; normally keep the default
nfaults: 1  # Number of faults to be modeled
fault_aliasnames: null  # null or exactly one alias per fault (e.g., ['RC', 'KL'])
lon_lat_0: null  # [lon, lat] degrees; set here or pass lon0/lat0

# ----------- Slip Sampling Mode ----------- #
# Set slip sampling mode to 'mag_rake' or 'ss_ds'
slip_sampling_mode: 'mag_rake'  # 'mag_rake' for magnitude and rake, 'ss_ds' for strike-slip and dip-slip

# ----------- Data Clipping Options ----------- #
# Options for clipping the data
clipping_options:
  enabled: false  # Whether to perform clipping
  method: 'lon_lat_range'  # Clipping method (e.g., 'lon_lat_range')
  lon_lat_range: [-119.0, -117.0, 34.0, 36.0]  # [lon_min, lon_max, lat_min, lat_max]

# ----------- Bounds Settings ----------- #
# Define parameter bounds for faults
# Legacy nonlinear configs use [Uniform, lower, range].
prior_bounds_format: lower_range
bounds:
  defaults:  # Default bounds for all faults
    # lon/lat/depth describe the midpoint of the fault top edge.
    lon: [Uniform, 87.3, 0.3]  # Longitude in degrees
    lat: [Uniform, 28.6, 0.2]  # Latitude in degrees
    depth: [Uniform, 0.0, 10.0]  # Depth in km, positive downward
    # Compact geometry accepts -90..180; 0..180 is preferred and 0..90 is
    # simplest for a known side. This legacy file uses lower/range.
    dip: [Uniform, 10, 70]  # Dip in degrees; actual interval is [10, 80]
    width: [Uniform, 1.0, 39.0]  # Fault width bounds (in kilometers)
    length: [Uniform, 1.0, 199.0]  # Fault length bounds (in kilometers)
    strike: [Uniform, 270.0, 90.0]  # Clockwise from North; actual [270, 360]
    slip: [Uniform, 0.0, 10.0]  # Total slip bounds (in meters)
    rake: [Uniform, -150, 120.0]  # Rake is not auto-changed with dip side.
  # fault_1:  # Optional override for fault_1 or its alias name
  #   rake: [Uniform, -30, 60.0]
  #   strike: [Uniform, 0.0, 270.0]

# ----------- Fixed Parameters ----------- #
# Fixed parameters for specific faults
fixed_params: {}
# fixed_params:
#   fault_0:  # Uncomment and set fixed parameters for fault_0 if needed
#     depth: 3.1578  # km; fixed values are not sampled

# ----------- Geodata Parameters ----------- #
# Parameters related to geodata
geodata:
  verticals: true  # Whether to include vertical data (boolean or list of booleans)
  # Legacy entry: enabled estimates one reference offset for selected
  # InSAR/leveling datasets. Use the new nonlinear-geometry template for
  # simplified SAR transforms 1/3/4 or GPS translation.
  polys:
    enabled: true  # Whether to estimate polynomial corrections
    boundaries:
      defaults: [Uniform, -200.0, 400.0]  # Default polynomial correction bounds
  faults: null  # Fault names for each geodata (e.g., [null, null, null, null])
  sigmas:  # Standard deviations for geodata
    mode: 'individual'  # single | individual | grouped
    # Required only for grouped mode. Use actual data.name values and assign
    # every dataset to exactly one group.
    # groups:
    #   sar_group: [dataset_a, dataset_b]
    #   gps_group: [dataset_c]
    update: true  # bool, or one bool per dataset/group
    bounds:
      defaults: [Uniform, -3.0, 6.0]  # Default bounds for sigmas
      sigma_0: [Uniform, -3.0, 6.0]  # Bounds for sigma_0
    # Scalar; individual/grouped also accept an aligned list or name mapping.
    values: 0.0
    log_scaled: true  # true: values and bounds represent log10(sigma)

# ----------- Data Sources ----------- #
# Data sources for GPS and InSAR data
# Relative paths are resolved from the process working directory.
data_sources:
  gps:
    directory: '../gps'  # Directory containing GPS data files
    file_pattern: 'cogps*'  # File pattern to match GPS data files
  insar:
    directory: '../insar'  # Directory containing InSAR data files
    file_pattern: '*.rsp'  # File pattern to match InSAR data files
""")

    # Write the configuration to the output file
    with open(output_path, "w") as file:
        yaml.dump(config, file)

    print(f"Nonlinear Bayesian configuration file generated at: {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a default nonlinear Bayesian configuration file.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="default_config.yml",
        help="Output path for the configuration file (default: default_config.yml)"
    )
    parser.add_argument(
        "--template",
        choices=("legacy", "geometry"),
        default="legacy",
        help=(
            "Template family to generate. 'legacy' preserves the old nonlinear "
            "explore config semantics; 'geometry' generates the new nonlinear "
            "geometry SMC template."
        ),
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    if args.template == "geometry":
        try:
            from .generate_nonlinear_geometry_config import generate_nonlinear_geometry_config
        except ImportError:
            from generate_nonlinear_geometry_config import generate_nonlinear_geometry_config
        generate_nonlinear_geometry_config(output_path)
    else:
        generate_nonlinear_config(output_path)

if __name__ == "__main__":
    main()
