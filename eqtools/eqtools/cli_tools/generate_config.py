from ruamel.yaml import YAML
import os

def generate_default_config(output_path, gf_method=None, interseismic_config_file=None,
                            include_des_config=False,
                            pressure_sources=None, sbarbot_sources=None):
    """
    Generate a default configuration file for Bayesian inversion with comments.
    If gf_method is 'pscmp' or 'edcmp', specific options will be included.
    If interseismic_config_file is provided, a pointer to that file is written.
    If include_des_config is True, Depth-Equalized Smoothing (DES) configuration will be added.
    If pressure_sources is provided, a pressure_sources section will be generated.
    If sbarbot_sources is provided, a sbarbot_sources section will be generated.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Define the basic configuration with comments
    config_text = """
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
slip_sampling_mode: mag_rake  # Slip sampling mode: 'ss_ds', 'rake_fixed', 'magnitude_rake' or its alias 'mag_rake'
rake_angle: 0  # Rake angle to be used when slip_sampling_mode is 'rake_fixed'
bayesian_sampling_mode: 'SMC_FJ'  # Sampling mode: 'FULLSMC' or 'SMC_F_J' or its alias 'SMC_FJ'
nchains: 100  # Number of chains for BayesianMultiFaultsInversion
chain_length: 50  # Length of each chain
use_bounds_constraints: true  # Whether to use bounds constraints
use_rake_angle_constraints: true  # Whether to use rake angle constraints
interseismic_config_file: null  # Optional separate interseismic block-motion config

# ECAT assumes observations, Green's functions, slip variables and constraint
# right-hand sides already use this same numerical unit after reader/factor
# conversion. Use m for cumulative displacement; use m/yr or mm/yr for rates.
units:
  observation: m

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
  
  # Vertical displacement data configuration - multiple formats supported:
  # 1. Boolean: true (all datasets) or false (none)
  # 2. List of booleans: [true, true, false] (per-dataset)
  verticals: true  # Whether to include vertical data, default true for InSAR data
  
  # Polynomial correction configuration - multiple formats supported:
  # 1. null: No polynomial correction for any dataset
  # 2. Integer: Same polynomial order for all datasets
  # 3. List of values: [3, null, 1] (per-dataset polynomial orders)
  # 4. String: polynomial order as string (for compatibility)
  #
  # Recommended values by data type:
  # - InSAR (SAR): 3 (removes orbital ramps and atmospheric effects)
  # - GPS: null (high precision, no polynomial needed)
  # - Optical/Offset (POT/FT): 1 (removes linear trends)
  # - Other displacement data: 1-2 or 'eulerrotation' (depending on quality)
  # 
  # Common usage examples:
  # polys: null                    # No correction for all datasets
  # polys: 3                       # Order-3 polynomial for all datasets
  # polys: [3, null, 1]           # Mixed: SAR=3, GPS=null, Optical=1
  polys: null  # Default: no polynomial correction
  
  faults: null  # List of fault names, optional: (null, list of fault name list)
  sigmas:  # Standard deviation of the geodata
    # Update configuration - multiple formats supported:
    # 1. Boolean: true (update all) or false (update none)
    # 2. List of booleans: [true, false, true] (explicit per-dataset)
    # 3. List of indices: [0, 2] (update datasets at these indices)
    # 4. List of names: ["sar_a", "sar_c"] (update datasets by name)
    # 5. Dictionary: {"true_indices": [0, 2]} (legacy format)
    update: true
    initial_value: 0  # Initial value for sigmas, optional: (float, list of floats)
    log_scaled: true  # Whether sigmas are log-scaled

# ----------- Smoothing Parameters ----------- #
# Parameters for smoothing
alpha:
  enabled: true  # Whether to enable smoothing
  update: true  # Whether to update alpha during inversion
  initial_value: -2.0  # Initial value for alpha. Optional: (float, list of floats with same length as alpha['faults'], or a single float list)
  log_scaled: true  # Whether alpha is log-scaled
  faults: null  # List of fault names for smoothing"""

    # Add DES configuration if requested
    if include_des_config:
        config_text += """

# -------------------------------------------------------------------------
# Depth-Equalized Smoothing (DES) Configuration [Zhang et al., 2025]
# Note: Currently only supported in BLSE/VCE inversion mode.
# -------------------------------------------------------------------------
des:
  enabled: false               # Whether to enable DES (true/false)
  mode: 'per_patch'           # Mode: 'per_patch' (default/recommended), 'per_depth', 'per_column'
  norm: 'l2'                  # Norm type: 'l2' (default), 'l1'
  
  # Configuration below is required only when mode is 'per_depth'
  depth_grouping:
    strategy: 'uniform'       # Grouping strategy: 'uniform' (equidistant), 'custom', 'values'
    interval: 1.0             # Interval for 'uniform' strategy (unit: km)
    # custom_groups: [0, 5, 10, 20, 50] # Depth nodes for 'custom' strategy
    # tolerance: 0.1          # Tolerance for 'values' strategy (unit: km)"""

    # Continue with fault parameters
    config_text += """

# ----------- Fault Parameters ----------- #
# Parameters for fault geometry and mesh generation
# Note: This section is for Fault-type sources only.
# Pressure and Sbarbot sources should be configured in their own sections below.
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
        method: null
        # slipdir: sd  # Slip direction chars: s=strikeslip, d=dipslip, t=tensile, c=coupling (default: sd)
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
"""

    # Add Pressure source section if requested
    if pressure_sources:
        pressure_entries = ""
        for src in pressure_sources:
            pressure_entries += f"""
  {src}:
    method_parameters:
      update_GFs:
        method: homogeneous  # Green's function method for {src}
        options: {{}}"""
        config_text += f"""
# ----------- Pressure Source Parameters ----------- #
# Parameters for Pressure sources (Mogi, CDM, pCDM, Yang, etc.)
# Note: Pressure sources do NOT support smoothing (Laplacian), mesh generation,
# or geometry updates. Only update_GFs is meaningful.
pressure_sources:
  defaults:
    method_parameters:
      update_GFs:
        method: homogeneous  # Default GF method for Pressure sources
        options: {{}}{pressure_entries}
"""
    else:
        config_text += """
# ----------- Pressure Source Parameters ----------- #
# Uncomment and configure if you have Pressure sources (Mogi, CDM, pCDM, Yang, etc.)
# Note: Pressure sources do NOT support smoothing (Laplacian), mesh generation,
# or geometry updates. Only update_GFs is meaningful.
# pressure_sources:
#   defaults:
#     method_parameters:
#       update_GFs:
#         method: homogeneous
#         options: {}
#   MyPressureSource:
#     method_parameters:
#       update_GFs:
#         method: homogeneous
#         options: {}
"""

    # Add Sbarbot source section if requested
    if sbarbot_sources:
        sbarbot_entries = ""
        for src in sbarbot_sources:
            sbarbot_entries += f"""
  {src}:
    method_parameters:
      update_GFs:
        method: null  # Must specify GF method for {src}
        # strain_components: [eps11, eps12, eps13, eps22, eps23, eps33]  # Default: all 6 symmetric tensor components
        options: {{}}"""
        config_text += f"""
# ----------- Sbarbot Source Parameters ----------- #
# Parameters for Sbarbot sources (volumetric strain sources)
# Note: Sbarbot sources do NOT support smoothing (Laplacian), mesh generation,
# or geometry updates. The GF method must be explicitly specified.
sbarbot_sources:
  defaults:
    method_parameters:
      update_GFs:
        method: null  # Must be explicitly set per source (no default inference)
        # strain_components: [eps11, eps12, eps13, eps22, eps23, eps33]  # Default: all 6 symmetric tensor components
        options: {{}}{sbarbot_entries}
"""
    else:
        config_text += """
# ----------- Sbarbot Source Parameters ----------- #
# Uncomment and configure if you have Sbarbot sources (volumetric strain)
# Note: Sbarbot sources do NOT support smoothing (Laplacian), mesh generation,
# or geometry updates. The GF method must be explicitly specified.
# sbarbot_sources:
#   defaults:
#     method_parameters:
#       update_GFs:
#         method: null  # Must be explicitly set per source
#         # strain_components: [eps11, eps12, eps13, eps22, eps23, eps33]  # Default: all 6
#         options: {}
#   MySbarbotSource:
#     method_parameters:
#       update_GFs:
#         method: null  # Specify here
#         # strain_components: [eps12, eps13]  # Override per source if needed
#         options: {}
"""

    # Load the configuration
    config = yaml.load(config_text)
    if interseismic_config_file is not None:
        config["interseismic_config_file"] = interseismic_config_file
        config["units"]["observation"] = "m/yr"

    # Set Green's function method and options if provided
    if gf_method is not None:
        config['faults']['defaults']['method_parameters']['update_GFs']['method'] = gf_method
        if gf_method.lower() == "pscmp":
            from csi.psgrn_pscmp.pscmp_options import PscmpOptions
            config['faults']['defaults']['method_parameters']['update_GFs']['options'] = \
                PscmpOptions.to_commented_map()
        elif gf_method.lower() == "edcmp":
            from csi.edgrn_edcmp.edcmp_backends import EdcmpOptions
            defaults = EdcmpOptions(
                fallback_engines=["exe"],
                n_jobs=8, cleanup_inp=False, force_recompute=False,
            )
            config['faults']['defaults']['method_parameters']['update_GFs']['options'] = \
                EdcmpOptions.to_commented_map(defaults)
        else:
            config['faults']['defaults']['method_parameters']['update_GFs']['options'] = None

    # Write the configuration to the output file
    with open(output_path, "w") as file:
        yaml.dump(config, file)

    print(f"Default configuration file generated at: {output_path}")
    if interseismic_config_file:
        print(f"Interseismic configuration pointer set to: {interseismic_config_file}")
    if pressure_sources:
        print(f"Pressure source(s) configured: {pressure_sources}")
    if sbarbot_sources:
        print(f"Sbarbot source(s) configured: {sbarbot_sources}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a default Bayesian inversion configuration file.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="default_config.yml",
        help="Output path for the configuration file (default: default_config.yml)"
    )
    parser.add_argument(
        "--gf-method",
        type=str,
        default=None,
        help="Green's function calculation method (e.g. pscmp, edcmp, okada, cutde, homogeneous, etc.)"
    )
    parser.add_argument(
        "--interseismic-config",
        type=str,
        default=None,
        help="Optional interseismic_config.yml path to record in default_config.yml"
    )
    parser.add_argument(
        "--include-des-config",
        action="store_true",
        help="Include Depth-Equalized Smoothing (DES) configuration in the generated file"
    )
    parser.add_argument(
        "-p", "--pressure",
        type=str,
        nargs="+",
        help="Pressure source name(s) to include (e.g., 'Mogi1 CDM1 pCDM1')"
    )
    parser.add_argument(
        "-s", "--sbarbot",
        type=str,
        nargs="+",
        help="Sbarbot source name(s) to include (e.g., 'Sbarbot1 Sbarbot2')"
    )
    parser.add_argument(
        "--show-gf-options",
        type=str,
        nargs="?",
        const="all",
        default=None,
        metavar="METHOD",
        help="Show available options for a GF method (edcmp, pscmp) or all methods if no argument given, then exit"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "yaml"],
        default="yaml",
        help="Output format for --show-gf-options (default: yaml)"
    )
    args = parser.parse_args()

    if args.show_gf_options:
        from csi import describe_gf_options
        method = None if args.show_gf_options == "all" else args.show_gf_options
        describe_gf_options(method, format=args.format)
        return

    output_path = os.path.abspath(args.output)
    generate_default_config(output_path, gf_method=args.gf_method, 
                          interseismic_config_file=args.interseismic_config,
                          include_des_config=args.include_des_config,
                          pressure_sources=args.pressure,
                          sbarbot_sources=args.sbarbot)

if __name__ == "__main__":
    main()
