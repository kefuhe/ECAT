import argparse
from csi.psgrn_pscmp.pscmp_config import (
    PSCMPConfig,
    create_observation_template_csv,
    create_fault_template_csv,
)

def main():
    parser = argparse.ArgumentParser(
        description="Generate PSCMP observation/fault/config template files."
    )
    parser.add_argument(
        "--obs-csv",
        type=str,
        help="Output CSV filename for observation points template (e.g., obs_rectangular.csv)",
    )
    parser.add_argument(
        "--obs-type",
        type=str,
        default="rectangular",
        choices=["rectangular", "profile", "irregular"],
        help="Observation array type for template: rectangular, profile, or irregular (default: rectangular)",
    )
    parser.add_argument(
        "--fault-csv",
        type=str,
        help="Output CSV filename for fault sources template (e.g., faults_with_patches.csv)",
    )
    parser.add_argument(
        "--fault-type",
        type=str,
        default="mixed",
        choices=["single_patches", "uniform_segments", "mixed"],
        help="Fault template type: single_patches, uniform_segments, or mixed (default: mixed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Output PSCMP config template filename (e.g., pscmp_template.dat)",
    )
    parser.add_argument(
        "--template",
        type=str,
        default="simple",
        choices=["simple", "insar", "coulomb", "detailed"],
        help="PSCMP config template type: simple, insar, coulomb, or detailed (default: simple)",
    )

    args = parser.parse_args()

    if args.obs_csv:
        create_observation_template_csv(args.obs_csv, args.obs_type)
    if args.fault_csv:
        create_fault_template_csv(args.fault_csv, args.fault_type)
    if args.config:
        config = PSCMPConfig.from_template(args.template)
        config.write_config_file(args.config)
    if not (args.obs_csv or args.fault_csv or args.config):
        parser.print_help()

if __name__ == "__main__":
    main()