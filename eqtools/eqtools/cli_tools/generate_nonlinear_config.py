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
nfaults: 1  # Number of faults to be modeled
fault_aliasnames: null  # Alias names for faults (e.g., ['RC', 'KL'])
lon_lat_0: null  # UTM coordinates of the origin (e.g., [lon, lat])

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
# All bounds use the format [lower bound, range increment]
bounds:
  defaults:  # Default bounds for all faults
    lon: [Uniform, 87.3, 0.3]  # Longitude bounds
    lat: [Uniform, 28.6, 0.2]  # Latitude bounds
    depth: [Uniform, 0.0, 10.0]  # Depth bounds (in kilometers)
    dip: [Uniform, 10, 70]  # Dip angle bounds (in degrees)
    width: [Uniform, 1.0, 39.0]  # Fault width bounds (in kilometers)
    length: [Uniform, 1.0, 199.0]  # Fault length bounds (in kilometers)
    strike: [Uniform, 270.0, 90.0]  # Strike angle bounds (in degrees)
    magnitude: [Uniform, 0.0, 10.0]  # Magnitude bounds
    rake: [Uniform, -150, 120.0]  # Rake angle bounds (in degrees)
  fault_1:  # Specific bounds for fault_1 or its alias name
    rake: [Uniform, -30, 60.0]  # Rake angle bounds for fault_1 (in degrees)
    strike: [Uniform, 0.0, 270.0]  # Strike angle bounds for fault_1 (in degrees)

# ----------- Fixed Parameters ----------- #
# Fixed parameters for specific faults
fixed_params:
  # fault_0:  # Uncomment and set fixed parameters for fault_0 if needed
    # lon: 102.205
    # depth: 3.1578
  fault_1:  # Fixed parameters for fault_1 or its alias name
    lon: -117.541  # Fixed longitude
    lat: 35.6431  # Fixed latitude
    depth: 0.0  # Fixed depth (in kilometers)
    strike: 227.0  # Fixed strike angle (in degrees)

# ----------- Geodata Parameters ----------- #
# Parameters related to geodata
geodata:
  verticals: true  # Whether to include vertical data (boolean or list of booleans)
  polys:  # Options for estimating polynomial corrections
    enabled: true  # Whether to estimate polynomial corrections
    boundaries:
      defaults: [Uniform, -200.0, 400.0]  # Default polynomial correction bounds
  faults: null  # Fault names for each geodata (e.g., [null, null, null, null])
  sigmas:  # Standard deviations for geodata
    # Update configuration - multiple formats supported:
    # 1. Boolean: true (update all) or false (update none)
    # 2. List of booleans: [true, false, true] (explicit per-dataset)
    # 3. List of indices: [0, 2] (update datasets at these indices)
    # 4. List of names: ["sar_a", "sar_c"] (update datasets by name)
    # 5. Dictionary: {"true_indices": [0, 2]} (legacy format)
    update: true  # Whether to update sigmas during inversion
    bounds:
      defaults: [Uniform, -3.0, 6.0]  # Default bounds for sigmas
      sigma_0: [Uniform, -3.0, 6.0]  # Bounds for sigma_0
    values: [0.0, 0.0, 0.0, 0.0]  # Initial values for sigmas
    log_scaled: true  # Whether sigmas are log-scaled

# ----------- Data Sources ----------- #
# Data sources for GPS and InSAR data
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
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    generate_nonlinear_config(output_path)

if __name__ == "__main__":
    main()