import argparse
import pandas as pd
from csi.edgrn_edcmp.edgrn_config import EDGRNConfig, EDGRNParameters, EdgrnLayer

def create_layer_template_csv(filename="edgrn_layer_template.csv"):
    """
    Create a CSV template file for EDGRN layer definitions.
    """
    df = pd.DataFrame([
        {'depth': 0.0, 'vp': 5570.0, 'vs': 3216.0, 'rho': 2900.0}
    ])
    df.to_csv(filename, index=False)
    print(f"EDGRN layer template saved to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate EDGRN layer/config template files.")
    parser.add_argument('--csv', type=str, help="Output CSV filename for layer template (e.g., edgrn_layer_template.csv)")
    parser.add_argument('--config', type=str, help="Output EDGRN config template filename (e.g., edgrn_template.inp)")
    args = parser.parse_args()

    if args.csv:
        create_layer_template_csv(args.csv)
    if args.config:
        params = EDGRNParameters()
        config = EDGRNConfig(params)
        config.write_config_file(args.config, verbose=True)
    if not (args.csv or args.config):
        parser.print_help()

if __name__ == "__main__":
    main()