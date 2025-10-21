from ruamel.yaml import YAML
import os

def generate_bounds_config(output_path, faultnames=None):
    """
    Generate a bounds configuration file with comments.

    Parameters:
    output_path (str): The output path for the configuration file.
    faultnames (list or None): A single fault name or a list of fault names. If None, a default example fault is used.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Use default fault name if none is provided
    if not faultnames:
        faultnames = ["ExampleFault"]

    # Define the configuration with comments
    header_comment = """
# ----------- Boundary Configuration Guide ----------- #
# This configuration file defines the parameter bounds for Bayesian inversion.
# Different sampling modes and their corresponding boundary constraints are as follows:
#
# 1. SMC-FJ Mode:
#    - Only supports 'ss_ds' slip sampling mode.
#    - Slip is constrained by the following boundaries:
#      - rake_angle
#      - strikeslip
#      - dipslip
#      - Euler constraints (if applicable)
#
# 2. FULLSMC Mode:
#    - Supports 'ss_ds', 'magnitude_rake', and 'rake_fixed' slip sampling modes.
#    - Slip constraints:
#      - 'ss_ds': strikeslip, dipslip
#      - 'magnitude_rake': slip_magnitude, rake_angle
#      - 'rake_fixed': slip_magnitude
#
# 3. BLSE Mode:
#    - Default to 'ss_ds' slip inversion mode.
#    - Slip is constrained by the following boundaries:
#      - rake_angle
#      - strikeslip
#      - dipslip
#
# Note:
# - Left-lateral slip is positive, reverse slip is positive, and opening is positive.
# - Pay attention to the sign conventions when setting slip boundaries.
# ---------------------------------------------------- #
"""

    geometry_bounds = "\n".join([f"    {fault}: [-10, 10]  # Geometry parameter bounds for {fault}" for fault in faultnames])
    slip_magnitude_bounds = "\n".join([f"    {fault}: [0, 15]  # Slip magnitude bounds for {fault} (unit: meters)" for fault in faultnames])
    rake_angle_bounds = "\n".join([f"    {fault}: [-120, -60]  # Rake angle bounds for {fault} (unit: degrees; counterclockwise rotation)" for fault in faultnames])
    strikeslip_bounds = "\n".join([f"    {fault}: [-10, 10]  # Strike-slip bounds for {fault} (unit: meters)" for fault in faultnames])
    dipslip_bounds = "\n".join([f"    {fault}: [-10, 0]  # Dip-slip bounds for {fault} (unit: meters)" for fault in faultnames])
    poly_bounds = "\n".join([f"    {fault}: [-1000, 1000]  # Polynomial parameter bounds for {fault}" for fault in faultnames])

    config = yaml.load(f"""
{header_comment}

# ----------- Global Bounds Settings ----------- #
# Global setting for lower and upper bounds
lb: -3  # Global lower bound for all parameters
ub: 3   # Global upper bound for all parameters

# ----------- Parameter-Specific Bounds ----------- #
# Set parameter boundaries based on parameter types
geometry:
{geometry_bounds}
slip_magnitude:
{slip_magnitude_bounds}
rake_angle:
{rake_angle_bounds}
strikeslip:
{strikeslip_bounds}
dipslip:
{dipslip_bounds}
poly:
{poly_bounds}
sigmas: [-3, 3]  # log(sigma^2), where sigma is the standard deviation of the data
alpha: [-3, 3]   # log(alpha^2), where alpha is the smoothing parameter
""")

    # Write the configuration to the output file
    with open(output_path, "w") as file:
        yaml.dump(config, file)

    print(f"Bounds configuration file generated at: {output_path}")

def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a bounds configuration file.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="bounds_config.yml",
        help="Output path for the configuration file (default: bounds_config.yml)"
    )
    parser.add_argument(
        "-f", "--faultnames",
        type=str,
        nargs="+",
        help="Fault name(s) for which to generate bounds (e.g., 'Fault1 Fault2')"
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    faultnames = args.faultnames
    generate_bounds_config(output_path, faultnames)

if __name__ == "__main__":
    main()