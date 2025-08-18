from ruamel.yaml import YAML
import shutil
import os

def generate_downsample_config(output_path, mode=None, copy_script=False):
    """
    Generate a default configuration file for downsampling with comments.

    Parameters:
    output_path (str): The output path for the configuration file.
    mode (str): The mode for the configuration file ('sar', 'optical', or None for full configuration).
    copy_script (bool): Whether to copy the downsampling script to the current working directory.
    """
    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)

    # Define the full configuration template
    full_config = yaml.load("""
#-------------------------------------------------------------------------------#
# General configuration                                                        #
# This section contains general settings such as the reference longitude and   #
# latitude, mask boundaries, and plotting box limits.                          #
#-------------------------------------------------------------------------------#
general:
  lon0: 96
  lat0: 21.5
  maskOut:
    minLon: 95.5
    maxLon: 96.48
    minLat: 18.0
    maxLat: 22.6
  downsample_box:
    minlat: 18.5
    maxlat: 25
    minlon: 93.5
    maxlon: 98.5
  plot_box:
    plotMinLat: null
    plotMaxLat: null
    plotMinLon: null
    plotMaxLon: null

#-------------------------------------------------------------------------------#
# Data type configuration                                                      #
# Specify the type of data being processed: 'sar' or 'optical'.                #
#-------------------------------------------------------------------------------#
data_type: "sar"  # Default is 'sar'. Change to 'optical' for optical data.

#-------------------------------------------------------------------------------#
# SAR-specific configuration                                                   #
# This section contains settings specific to SAR data processing.              #
#-------------------------------------------------------------------------------#
sar_config:
  outName: "S1_T143A"
  prefix: "off_20250327_20250402"
  offset_direction: "az"  # Options: 'r' for range offset, 'az' for azimuth offset
  sar_dict:
    phsname: "roff_20250319_20250331.phs"
    rscname: "roff_20250319_20250331.phs.rsc"
    azifile: "off_20250319_20250331.azi"
    incfile: "off_20250319_20250331.inc"
  use_offset_sar: true # Set to true to use offset SAR data
  output_check: true # Set to true to enable output check
  outlier_threshold: 12.0 # Threshold for outlier detection
  downsample4covar: 1 # Downsampling factor for covariance estimation
  plot_raw_sar:
    save_fig: true
    rawdownsample4plot: 1
    colorbar_x: 0.4
    colorbar_y: 0.25
    colorbar_length: 0.2
    vmin: -300
    vmax: 300
    factor4plot: 100.0

#-------------------------------------------------------------------------------#
# Optical-specific configuration                                               #
# This section contains settings specific to optical data processing.          #
#-------------------------------------------------------------------------------#
optical_config:
  outName: "Optical_S2_part1"
  filename: "Sagaing_S2_Part1.tif"
  vel_type: 'north' # Options: 'north', 'east'
  output_check: true # Set to true to enable output check
  outlier_threshold: 10.0 # Threshold for outlier detection
  plot_raw_optical:
    save_fig: true
    rawdownsample4plot: 1
    colorbar_x: 0.4
    colorbar_y: 0.25
    colorbar_length: 0.2
    vmin: [-1.0, -3.0]
    vmax: [1.0, 3.0]
    factor4plot: 1.0

#-------------------------------------------------------------------------------#
# Covariance configuration                                                     #
# This section contains settings for covariance estimation.                    #
#-------------------------------------------------------------------------------#
covar:
  do_covar: false
  function: "exp"
  frac: 0.002
  every: 2.0
  distmax: 100.0
  rampEst: true

#-------------------------------------------------------------------------------#
# Downsampling configuration                                                   #
# This section contains settings for downsampling methods and parameters.      #
#-------------------------------------------------------------------------------#
downsample:
  enabled: false # Set to true to enable downsampling
  method: "trirb"  # Options: 'std', 'trirb'
  std_config:
    startingsize: 5.0
    minimumsize: 0.25
    tolerance: 0.05
    plot: false
    decimorig: 10
    std_threshold: 0.005
    smooth: null
    slipdirection: "sd"
  trirb_config:
    minimumsize: 1.25
    tolerance: 0.25
    plot: false
    decimorig: 10
    max_samples: 2500
    change_threshold: 10
    smooth_factor: 0.25
    slipdirection: "sd"
  plot_decim:
    figsize: [3.0, 5.0]
    cbaxis: [0.7, 0.13, 0.03, 0.25]

#-------------------------------------------------------------------------------#
# Fault configuration                                                          #
# This section contains settings for fault modeling, including trace file and  #
# geometry. Each fault can be enabled or disabled individually.                #
#-------------------------------------------------------------------------------#
faults:
  - enabled: false  # Set to true to enable this fault
    trace_file: "../../../../Faults/example_rupture_trace.dat"
    dip_angle: 86
    dip_direction: 258
    top_size: 5.0
    bottom_size: 8.0
    top_depth: 0.0
    bottom_depth: 18.0
""")

    # Adjust the configuration and comments based on the selected mode
    if mode == "sar":
        # First, adjust comments for optical_config before deleting it
        full_config.yaml_set_comment_before_after_key("optical_config", before=None, after=None)  # Remove optical_config's preceding comments
        # full_config.yaml_set_comment_before_after_key("sar_config", before="SAR-specific configuration")  # Keep sar_config's preceding comments
        # Set data type
        full_config["data_type"] = "sar"
        # Delete optical_config
        del full_config["optical_config"]

        # Apply Optical section comment in a clean way
        comment_token = "------------------------------------------------------------------------------#\n"
        comment_text = f"{comment_token}Covariance configuration                                               #\n"
        comment_text += f"This section contains settings for covariance estimation.          #\n{comment_token}"
        full_config.yaml_set_comment_before_after_key("covar", before=comment_text)
    elif mode == "optical":
        # First, adjust comments for sar_config before deleting it
        full_config.yaml_set_comment_before_after_key("sar_config", before=None, after=None)  # Remove sar_config's preceding comments
        # full_config.yaml_set_comment_before_after_key("optical_config", before="Optical-specific configuration")  # Keep optical_config's preceding comments
        # Set data type
        full_config["data_type"] = "optical"
        # Delete sar_config
        del full_config["sar_config"]

        # Apply Optical section comment in a clean way
        comment_token = "------------------------------------------------------------------------------#\n"
        comment_text = f"{comment_token}Optical-specific configuration                                               #\n"
        comment_text += f"This section contains settings specific to optical data processing.          #\n{comment_token}"
        full_config.yaml_set_comment_before_after_key("optical_config", before=comment_text)

    # Write the configuration to the output file
    with open(output_path, "w") as file:
        yaml.dump(full_config, file)

    print(f"Downsampling configuration file generated at: {output_path}")

    # if copy_script:
    #   # Automatically copy the downsampling script to the current working directory
    #   script_src = os.path.join(os.path.dirname(__file__), "../scripts/downsample_script.py")
    #   script_dst = os.path.join(os.getcwd(), "downsample_script.py")
    #   shutil.copy(script_src, script_dst)
    #   print(f"Downsampling script copied to: {script_dst}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a default downsampling configuration file.")
    parser.add_argument(
        "-m", "--mode",
        type=str,
        choices=["sar", "optical"],
        help="Mode for the configuration file ('sar', 'optical', or None for full configuration)."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default="downsample_config.yml",
        help="Output path for the configuration file (default: downsample_config.yml)."
    )
    parser.add_argument(
        "-c", "--copy_script", 
        action="store_true",
        help="Copy the downsampling script to the current working directory."
    )
    args = parser.parse_args()

    output_path = os.path.abspath(args.output)
    generate_downsample_config(output_path, args.mode, args.copy_script)

if __name__ == "__main__":
    main()