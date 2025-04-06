from ruamel.yaml import YAML
import os

def generate_default_config(output_path):
    """
    Generate a default configuration file for Bayesian inversion with comments.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Define the configuration with comments
    config = yaml.load("""
# ----------- General Parameters ----------- #
# General settings for the Bayesian inversion process
GLs: null  # Custom Green's functions
moment_magnitude_threshold: 7.0  # Threshold for the moment magnitude
magnitude_tolerance: 0.2  # Range of the moment magnitude can be updated
patch_areas: null  # Subfault size
shear_modulus: 3.0e10  # Shear modulus in Pa

# ----------- Bayesian Inversion Parameters ----------- #
# Parameters related to Bayesian inversion
nonlinear_inversion: false  # Whether to use nonlinear inversion
slip_sampling_mode: magnitude_rake  # Slip sampling mode: 'ss_ds', 'magnitude_rake', 'rake_fixed'
rake_angle: 0  # Rake angle to be used when slip_sampling_mode is 'rake_fixed'
bayesian_sampling_mode: 'SMC_F_J'  # Sampling mode: 'FULLSMC' or 'SMC_F_J'
nchains: 100  # Number of chains for BayesianMultiFaultsInversion
chain_length: 50  # Length of each chain
use_bounds_constraints: true  # Whether to use bounds constraints
use_rake_angle_constraints: true  # Whether to use rake angle constraints

# ----------- Data Clipping Parameters ----------- #
# Parameters for data clipping
clipping_options:
  enabled: false  # Whether to enable data clipping
  methods:
    distance_to_fault:
      distance_to_fault: 1.5  # Maximum distance to fault in kilometers
    lon_lat_range:
      lon_lat_range: [86.0, 87.7, 32.5, 33.75]  # Longitude and latitude range

# ----------- Geodata Parameters ----------- #
# Parameters related to geodata
geodata:
  data: null  # Automatically generated from the script
  verticals: true  # Whether to include vertical data, default true for insar data, optional: (boolean, list of boolean)
  polys: 3  # Number of polygons, default 3 for insar data, optional: (value, list of values)
  faults: null  # List of fault names, optional: (null, list of fault name list)
  sigmas:  # Standard deviation of the geodata
    update: true
    initial_value: 0  # Initial value for sigmas, optional: (float, list of floats)
    log_scaled: true  # Whether sigmas are log-scaled

# ----------- Smoothing Parameters ----------- #
# Parameters for smoothing
alpha:
  enabled: true  # Whether to enable smoothing
  update: true  # Whether to update alpha during inversion
  initial_value: [-2.0]  # Initial value for alpha
  log_scaled: true  # Whether alpha is log-scaled
  faults: null  # List of fault names for smoothing

# ----------- Fault Parameters ----------- #
# Parameters for fault geometry and mesh generation
faults:
  defaults:
    geometry:
      update: false  # Whether to update fault geometry
      sample_positions: [0, 0]  # Sample positions for geometry
    method_parameters:
      update_mesh:
        method: 'generate_mesh'  # Method for mesh generation
        segments_dict:
          top_segments: 20
          bottom_segments: 20
          left_segments: 6
          right_segments: 6
          left_right_progression: 1.17
        verbose: 0  # Gmsh verbosity level
        show: false  # Whether to show Gmsh GUI
      update_GFs:
        geodata: null
        verticals: null
      update_Laplacian:
        method: 'Mudpy'  # Method for Laplacian calculation
        bounds: ['free', 'locked', 'free', 'free']  # Top Bottom Left Right
        topscale: 0.25
        bottomscale: 0.03
  ExampleFault:
    geometry:
      update: false
      sample_positions: [0, 4]
    method_parameters:
      update_fault_geometry:
        method: perturb_BottomFixedDir_RotateTransGeom  # Method for fault geometry update
        pivot: midpoint
        angle_unit: degrees
        force_pivot_in_coords: true
      update_mesh:
        method: 'generate_and_deform_mesh'  # Method for mesh deformation
        top_size: 1.5
        bottom_size: 3.0
        num_segments: 12
        disct_z: 8
""")

    # Write the configuration to the output file
    with open(output_path, "w") as file:
        yaml.dump(config, file)

    print(f"Default configuration file generated at: {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a default Bayesian inversion configuration file.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="default_config.yml",
        help="Output path for the configuration file (default: default_config.yml)"
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    generate_default_config(output_path)

if __name__ == "__main__":
    main()