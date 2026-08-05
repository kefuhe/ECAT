from ruamel.yaml import YAML
import os


def generate_nonlinear_geometry_config(output_path):
    """
    Generate a default configuration for the new nonlinear geometry SMC entry.

    The template uses user-facing lower/upper bounds.  The parser normalizes
    them to scipy loc/scale internally before sampling.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    config = yaml.load("""
# ----------- Nonlinear Geometry SMC ----------- #
nchains: 100       # Number of SMC chains/particles
chain_length: 50   # Mutation steps per SMC stage
nfaults: 1         # Number of compact fault sources
fault_aliasnames: null  # null or exactly one alias per fault, e.g. [MainFault]
lon_lat_0: null         # [lon, lat] degrees; set here or pass lon0/lat0

# New nonlinear geometry configs use [Uniform, lower, upper].
prior_bounds_format: lower_upper

slip_sampling_mode: 'mag_rake'  # mag_rake: slip+rake; ss_ds: strikeslip+dipslip

clipping_options:
  enabled: false  # Set true to clip all input datasets before inversion
  method: 'lon_lat_range'  # Currently supported template method
  lon_lat_range: [-119.0, -117.0, 34.0, 36.0]  # [lon_min, lon_max, lat_min, lat_max], degrees

bounds:
  defaults:
    # lon/lat/depth describe the midpoint of the fault top edge.
    lon: [Uniform, 87.3, 87.6]       # degrees
    lat: [Uniform, 28.6, 28.8]       # degrees
    depth: [Uniform, 0.0, 10.0]      # km, positive downward
    # Preferred dip coordinate is 0..180; use a 0..90 range for one known side.
    dip: [Uniform, 10.0, 80.0]       # degrees; strike changes with side at 90
    width: [Uniform, 1.0, 40.0]      # km
    length: [Uniform, 1.0, 200.0]    # km
    strike: [Uniform, 270.0, 360.0]  # degrees clockwise from North
    slip: [Uniform, 0.0, 10.0]       # m; mag_rake only
    rake: [Uniform, -150.0, -30.0]   # degrees; not auto-changed with dip side
    # For ss_ds, replace slip/rake above with:
    # strikeslip: [Uniform, -10.0, 10.0]  # m
    # dipslip: [Uniform, -10.0, 10.0]     # m

fixed_params: {}  # Keys are fault_0/fault alias; fixed values are not sampled
# fixed_params:
#   fault_0:
#     depth: 2.0  # km

geodata:
  verticals: true  # bool or one bool per dataset, in geodata order

  # Simple correction setup. A list must follow the Python geodata order.
  # SAR/InSAR: null=none, 1=offset, 3=offset+x/y ramps,
  # 4=offset+x/y ramps+xy cross term.
  # GPS: null=none or 'translation' (east/north[/up] offsets).
  # polys: 3                       # one or more SAR-only datasets
  # polys: [3, null, translation]  # per-dataset mixed example
  polys: null  # Default: no data-correction parameters
  # Default bounds for all enabled data-correction coefficients.  With
  # prior_bounds_format=lower_upper this means lower=-1000, upper=1000.
  poly_bounds: [Uniform, -1000.0, 1000.0]

  # Advanced overrides: use only for per-dataset/per-parameter customization.
  # data_corrections:
  #   enabled: true
  #   datasets:
  #     asc:
  #       transform: 3
  #       bounds: [Uniform, -1.0, 1.0]
  #       parameter_bounds:
  #         offset: [Uniform, -0.05, 0.05]
  #         x_ramp: [Uniform, -0.5, 0.5]
  #         y_ramp: [Uniform, -0.5, 0.5]
  #       display_names: ["$b_A$", "$r^x_A$", "$r^y_A$"]

  faults: null  # null=all faults; otherwise one fault-name list per dataset
  sigmas:
    mode: 'individual'  # single | individual | grouped
    # Required only for grouped mode. Use actual data.name values and assign
    # every dataset to exactly one group.
    # groups:
    #   sar_group: [dataset_a, dataset_b]
    #   gps_group: [dataset_c]
    update: true  # bool, or one bool per dataset/group
    bounds:
      defaults: [Uniform, -3.0, 3.0]  # Uses prior_bounds_format above
    # Scalar; individual/grouped also accept an aligned list or name mapping.
    values: 0.0
    log_scaled: true  # true: values and bounds represent log10(sigma)

# Relative paths below are resolved from the process working directory.
data_sources:
  gps:
    directory: '../gps'
    file_pattern: 'cogps*'
  insar:
    directory: '../insar'
    file_pattern: '*.rsp'
""")

    with open(output_path, "w", encoding="utf-8") as file:
        yaml.dump(config, file)

    print(f"Nonlinear geometry SMC configuration file generated at: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate a default nonlinear geometry SMC configuration file."
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="nonlinear_geometry.yml",
        help=(
            "Output path for the configuration file "
            "(default: nonlinear_geometry.yml)"
        ),
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    generate_nonlinear_geometry_config(output_path)


if __name__ == "__main__":
    main()
