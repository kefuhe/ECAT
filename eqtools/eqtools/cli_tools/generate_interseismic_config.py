from ruamel.yaml import YAML


def generate_interseismic_config(output_path, faultnames=None):
    """Generate a minimal interseismic loading and backslip configuration template."""
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    faultnames = faultnames or ["ExampleFault"]
    fault_blocks = "\n".join(
        f"""    {fault_name}:
      blocks: [Block_A, Block_B]
      reference_strike: 0.0
      motion_sense: dextral
      # Optional manual regional overrides for simple long-fault cases.
      # Keep this commented unless different patch groups require different
      # block pairs or reference strikes. Selectors use the same syntax as
      # cap_constraints/backslip_constraints.
      # loading_regions:
      #   - name: north
      #     selector: {{patches: [0, 1, 2]}}
      #     blocks: [Block_A, Block_B]
      #     reference_strike: 0.0"""
        for fault_name in faultnames
    )
    cap_blocks = "\n".join(
        f"""    {fault_name}:
      selector: null
      # mode: motion_sense is the default and works with estimated Euler blocks.
      # Use mode: loading_sign only when both loading blocks are fixed.
      mode: motion_sense
      max_coupling: 1.0"""
        for fault_name in faultnames
    )
    first_fault = faultnames[0]
    config_text = f"""# Interseismic block loading and optional backslip constraints.
version: 1

blocks:
  # Blocks define physical block membership and Euler parameter sources.
  # Each dataset listed here should have eulerrotation in geodata.polys when
  # its rigid block motion is estimated or fixed inside this inversion.
  Block_A:
    datasets: [GPS_Block_A]
    euler:
      mode: estimate
  Block_B:
    datasets: [GPS_Block_B]
    euler:
      mode: estimate
  # Fixed blocks are also supported:
  # Stable_Block:
  #   datasets: [GPS_Stable]
  #   euler:
  #     mode: fixed_pole
  #     value: [110.0, 35.0, 0.25]
  #     units: [degrees, degrees, degrees_per_myr]

fault_loading:
  enabled: false
  defaults:
    reference_strike: 0.0
    motion_sense: dextral
  faults:
{fault_blocks}

cap_constraints:
  enabled: false
  defaults:
    selector: null
    mode: motion_sense
    hard_overlap: skip
    max_coupling: 1.0
  faults:
{cap_blocks}

backslip_constraints:
  - fault: {first_fault}
    state: full_coupling
    selector: {{edge: top}}
    component: strikeslip
    name: top_full_coupling
    overwrite: true

  - fault: {first_fault}
    state: prescribed_coupling
    coupling: 0.0
    selector: {{edge: bottom}}
    component: strikeslip
    name: bottom_free_coupling
    overwrite: true

outputs:
  fields:
    - tectonic_loading_rate
    - backslip_rate
    - slip_deficit_signed
    - slip_deficit_magnitude
    - coupling_ratio
    - coupling_magnitude
    - creep_rate_signed
"""
    config = yaml.load(config_text)
    with open(output_path, "w", encoding="utf-8") as f:
        yaml.dump(config, f)
    print(f"Interseismic configuration file generated at: {output_path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate an interseismic block and fault-loading configuration file.")
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="interseismic_config.yml",
        help="Output path for the configuration file (default: interseismic_config.yml)",
    )
    parser.add_argument(
        "-f", "--fault",
        type=str,
        nargs="+",
        dest="faultnames",
        help="Fault name(s) to include in the template",
    )
    args = parser.parse_args()
    generate_interseismic_config(args.output, faultnames=args.faultnames)


if __name__ == "__main__":
    main()
