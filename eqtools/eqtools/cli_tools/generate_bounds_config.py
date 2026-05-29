from ruamel.yaml import YAML
import os

def generate_bounds_config(output_path, faultnames=None, pressure_sources=None, sbarbot_sources=None):
    """
    Generate a bounds configuration file with comments.

    Parameters:
    output_path (str): The output path for the configuration file.
    faultnames (list or None): A single fault name or a list of fault names. If None, a default example fault is used.
    pressure_sources (list or None): A list of Pressure source names (e.g., Mogi, CDM, pCDM).
    sbarbot_sources (list or None): A list of Sbarbot source names.
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
# 4. Multi-Source Support (source_bounds):
#    - For non-Fault sources (Pressure, Sbarbot), use the 'source_bounds' section.
#    - Pressure sources: components are 'pressure' (Mogi/CDM) or
#      'pressureDVx', 'pressureDVy', 'pressureDVz' (pCDM).
#    - Sbarbot sources: components match strain tensor names
#      (e.g., 'eps12', 'eps13', etc.).
#    - Each component accepts a global [min, max] or per-element list.
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

    # Build source_bounds section for non-Fault sources
    source_bounds_lines = []
    if pressure_sources:
        for src in pressure_sources:
            source_bounds_lines.append(f"    # Pressure source: {src}")
            source_bounds_lines.append(f"    # For Mogi/CDM use 'pressure'; for pCDM use 'pressureDVx', 'pressureDVy', 'pressureDVz'")
            source_bounds_lines.append(f"    {src}:")
            source_bounds_lines.append(f"        pressure: [-1e6, 1e6]  # Pressure bounds (Pa)")
    if sbarbot_sources:
        for src in sbarbot_sources:
            source_bounds_lines.append(f"    # Sbarbot source: {src}")
            source_bounds_lines.append(f"    {src}:")
            source_bounds_lines.append(f"        eps12: [-1e-4, 1e-4]  # Strain component bounds")
            source_bounds_lines.append(f"        eps13: [-1e-4, 1e-4]")
    if not source_bounds_lines:
        # Add commented-out example
        source_bounds_lines = [
            "    # --- Uncomment and edit for Pressure sources ---",
            "    # MyPressureSource:",
            "    #     pressure: [-1e6, 1e6]  # For Mogi/CDM (Pa)",
            "    # MyPCDMSource:",
            "    #     pressureDVx: [-1e6, 1e6]",
            "    #     pressureDVy: [-1e6, 1e6]",
            "    #     pressureDVz: [-1e6, 1e6]",
            "    # --- Uncomment and edit for Sbarbot sources ---",
            "    # MySbarbotSource:",
            "    #     eps12: [-1e-4, 1e-4]",
            "    #     eps13: [-1e-4, 1e-4]",
        ]
    source_bounds_yaml = "\n".join(source_bounds_lines)

    # Build source_constraints section for inequality/equality constraints
    source_constraints_lines = []
    if pressure_sources:
        for src in pressure_sources:
            source_constraints_lines.append(f"    # Pressure source: {src}")
            source_constraints_lines.append(f"    # {src}:")
            source_constraints_lines.append(f"    #   - {{name: positive_pressure, type: inequality, rule: 'pressure >= 0'}}")
    if sbarbot_sources:
        for src in sbarbot_sources:
            source_constraints_lines.append(f"    # Sbarbot source: {src}")
            source_constraints_lines.append(f"    # {src}:")
            source_constraints_lines.append(f"    #   - {{name: incompressible, type: equality, rule: 'incompressible'}}")
            source_constraints_lines.append(f"    #   - {{name: pos_eps12, type: inequality, rule: 'eps12 >= 0'}}")
    for fault in faultnames:
        source_constraints_lines.append(f"    # Fault source: {fault}")
        source_constraints_lines.append(f"    # {fault}:")
        source_constraints_lines.append(f"    #   - {{name: ss_positive, type: inequality, rule: 'strikeslip >= 0'}}")
        source_constraints_lines.append(f"    #   - {{name: ds_negative, type: inequality, rule: 'dipslip <= 0'}}")
        source_constraints_lines.append(f"    #   - {{name: zero_top_ss, type: equality, rule: 'zero_edge_slip(top, strikeslip)'}}")
        source_constraints_lines.append(f"    #   - {{name: zero_top_ds, type: equality, rule: 'zero_edge_slip(top, dipslip)'}}")
    if not source_constraints_lines:
        source_constraints_lines = [
            "    # --- Uncomment and edit as needed ---",
            "    # MyFault:",
            "    #   - {name: ss_positive, type: inequality, rule: 'strikeslip >= 0'}",
            "    #   - {name: ds_negative, type: inequality, rule: 'dipslip <= 0'}",
            "    #   - {name: zero_top_ss, type: equality, rule: 'zero_edge_slip(top, strikeslip)'}",
            "    #   - {name: zero_top_ds, type: equality, rule: 'zero_edge_slip(top, dipslip)'}",
            "    # MyPressureSource:",
            "    #   - {name: positive_pressure, type: inequality, rule: 'pressure >= 0'}",
            "    # MySbarbotSource:",
            "    #   - {name: incompressible, type: equality, rule: 'incompressible'}",
        ]
    source_constraints_yaml = "\n".join(source_constraints_lines)

    config = yaml.load(f"""
{header_comment}

# ----------- Global Bounds Settings ----------- #
# Global setting for lower and upper bounds
lb: -3  # Global lower bound for all parameters
ub: 3   # Global upper bound for all parameters

# ----------- Parameter-Specific Bounds (Fault sources) ----------- #
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
sigmas: [-3, 3]  # Bounds for sigma parameters; if log_scaled=true, values are log10(sigma)
alpha: [-3, 3]   # Bounds for alpha parameters; if log_scaled=true, values are log10(alpha)

# ----------- Source-Specific Bounds (non-Fault sources) ----------- #
# For Pressure and Sbarbot sources, define component bounds here.
# Each source name maps to its component bounds.
# Values can be [min, max] (applied to all elements) or a list of [min, max] per element.
source_bounds:
{source_bounds_yaml}

# ----------- Source-Specific Constraints (inequality / equality) ----------- #
# Define per-source inequality or equality constraints.
# Each source name maps to a list of constraint definitions:
#   - name: unique identifier
#     type: 'inequality' or 'equality'
#     rule: constraint expression (e.g., 'strikeslip >= 0', 'pressure <= 0')
#
# Supported rules by source type:
#   Fault:    strikeslip>=0, strikeslip<=0, dipslip>=0, dipslip<=0, strikeslip==0, dipslip==0
#             zero_edge_slip(edge, mode) — e.g. zero_edge_slip(top, ss+ds)
#   Pressure: pressure>=0, pressure<=0, pressureDVx>=0, volume>=0, etc.
#   Sbarbot:  incompressible (eps11+eps22+eps33=0), eps12>=0, eps12<=0, eps12==0, etc.
source_constraints:
{source_constraints_yaml}
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
    parser.add_argument(
        "-p", "--pressure",
        type=str,
        nargs="+",
        help="Pressure source name(s) (e.g., 'Mogi1 CDM1 pCDM1')"
    )
    parser.add_argument(
        "-s", "--sbarbot",
        type=str,
        nargs="+",
        help="Sbarbot source name(s) (e.g., 'Sbarbot1')"
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    faultnames = args.faultnames
    generate_bounds_config(output_path, faultnames,
                           pressure_sources=args.pressure,
                           sbarbot_sources=args.sbarbot)

if __name__ == "__main__":
    main()
