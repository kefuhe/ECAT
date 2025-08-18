import argparse
from csi.psgrn_pscmp.psgrn_config import PSGRNConfig, create_layer_template_csv, create_layer_template_excel

def main():
    parser = argparse.ArgumentParser(description="Generate PSGRN layer template files or config templates.")
    parser.add_argument('--csv', type=str, help="Output CSV filename for layer template (e.g., layer_template.csv)")
    parser.add_argument('--excel', type=str, help="Output Excel filename for layer template (e.g., layer_template.xlsx)")
    parser.add_argument('--config', type=str, help="Output PSGRN config template filename (e.g., psgrn_template.dat)")
    parser.add_argument('--template', type=str, default='continental', help="Model template type: continental, oceanic, simple_viscous, layered_viscous")
    args = parser.parse_args()

    if args.csv:
        create_layer_template_csv(args.csv)
    if args.excel:
        create_layer_template_excel(args.excel)
    if args.config:
        config = PSGRNConfig.from_template(args.template)
        config.write_config_file(args.config)
    if not (args.csv or args.excel or args.config):
        parser.print_help()

if __name__ == "__main__":
    main()