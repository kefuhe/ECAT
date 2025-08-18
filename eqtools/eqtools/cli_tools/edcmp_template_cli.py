import argparse
from csi.edgrn_edcmp.edcmp_config import EDCMPConfig, EDCMPParameters, RectangularSource

def create_layered_template(filename="edcmp_layered.inp"):
    """
    Generate a layered model EDCMP template input file.
    """
    params = EDCMPParameters()
    params.set_rectangular_observation_array(51, -35000, 15000, 51, -25000, 25000)
    params.output_dir = './'
    params.output_flags = (1, 0, 0, 0)
    params.output_files = ('edcmp.disp', 'edcmp.strn', 'edcmp.strss', 'edcmp.tilt')
    params.layered_model = True
    params.grn_dir = './edgrnfcts/'
    params.grn_files = ('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    params.sources = [
        RectangularSource(
            source_id=1, slip=2.5, xs=0.0, ys=0.0, zs=200.0,
            length=11000.0, width=10000.0, strike=174.0, dip=88.0, rake=178.0
        )
    ]
    config = EDCMPConfig(params)
    config.write_config_file(filename)
    print(f"Layered model EDCMP template written to: {filename}")

def create_homogeneous_template(filename="edcmp_homogeneous.inp"):
    """
    Generate a homogeneous half-space EDCMP template input file.
    """
    params = EDCMPParameters()
    params.set_rectangular_observation_array(51, -35000, 15000, 51, -25000, 25000)
    params.output_dir = './'
    params.output_flags = (1, 0, 0, 0)
    params.output_files = ('edcmp.disp', 'edcmp.strn', 'edcmp.strss', 'edcmp.tilt')
    params.layered_model = False
    params.zrec = 0.0
    params.lambda_ = 3.0e10
    params.mu = 3.0e10
    params.sources = [
        RectangularSource(
            source_id=1, slip=2.5, xs=0.0, ys=0.0, zs=200.0,
            length=11000.0, width=10000.0, strike=174.0, dip=88.0, rake=178.0
        )
    ]
    config = EDCMPConfig(params)
    config.write_config_file(filename)
    print(f"Homogeneous model EDCMP template written to: {filename}")

def main():
    parser = argparse.ArgumentParser(description="Generate EDCMP template input files.")
    parser.add_argument('--layered', type=str, help="Output filename for layered model template (e.g., edcmp_layered.inp)")
    parser.add_argument('--homogeneous', type=str, help="Output filename for homogeneous model template (e.g., edcmp_homogeneous.inp)")
    args = parser.parse_args()

    if args.layered:
        create_layered_template(args.layered)
    if args.homogeneous:
        create_homogeneous_template(args.homogeneous)
    if not (args.layered or args.homogeneous):
        parser.print_help()

if __name__ == "__main__":
    main()