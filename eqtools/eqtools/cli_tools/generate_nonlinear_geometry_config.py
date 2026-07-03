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
nchains: 100
chain_length: 50
nfaults: 1
fault_aliasnames: null
lon_lat_0: null

# New nonlinear geometry configs use [Uniform, lower, upper].
prior_bounds_format: lower_upper

slip_sampling_mode: 'mag_rake'

clipping_options:
  enabled: false
  method: 'lon_lat_range'
  lon_lat_range: [-119.0, -117.0, 34.0, 36.0]

bounds:
  defaults:
    lon: [Uniform, 87.3, 87.6]
    lat: [Uniform, 28.6, 28.8]
    depth: [Uniform, 0.0, 10.0]
    dip: [Uniform, 10.0, 80.0]
    width: [Uniform, 1.0, 40.0]
    length: [Uniform, 1.0, 200.0]
    strike: [Uniform, 270.0, 360.0]
    slip: [Uniform, 0.0, 10.0]
    rake: [Uniform, -150.0, -30.0]

fixed_params: {}

geodata:
  verticals: true

  # Same mental model as BLSE/Bayesian linear inversion:
  # one correction transform per dataset in geodata.data order.
  # SAR/InSAR currently supports null, 1, 3, and 4.
  # GPS currently supports explicit 'translation' only in this nonlinear entry.
  polys: null
  # Default bounds for all enabled data-correction coefficients.  With
  # prior_bounds_format=lower_upper this means [-1000, 1000].
  poly_bounds: [Uniform, -1000.0, 1000.0]

  # Advanced optional overrides. Keep this block commented until needed.
  # Use these only when one dataset or one correction parameter needs a
  # different transform, bounds, or display label.
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

  faults: null
  sigmas:
    mode: 'individual'
    update: true
    bounds:
      defaults: [Uniform, -3.0, 3.0]
    values: 0.0
    log_scaled: true

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
