from pathlib import Path
import shutil

from ruamel.yaml import YAML

from eqtools.csiExtend.downsample.config import (
    DOWNSAMPLE_METHOD_CHOICES,
    SAR_MODE_CHOICES,
    SAR_READER_CHOICES,
)


def find_adapter_downsampling_template_dir():
    """Return the source-maintained adapter_downsampling template directory."""

    here = Path(__file__).resolve()
    candidates = [
        here.parent / "templates" / "adapter_downsampling",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    candidate_text = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise FileNotFoundError(
        "Could not find adapter_downsampling template directory. Checked:\n"
        f"{candidate_text}"
    )


def copy_adapter_downsampling_template(output_dir):
    """Copy adapter_downsampling template files next to the generated config."""

    template_dir = find_adapter_downsampling_template_dir()
    output_dir = Path(output_dir)
    copied = []
    for source in template_dir.iterdir():
        if source.is_file():
            target = output_dir / source.name
            shutil.copy2(source, target)
            copied.append(target)
    return copied


def gamma_file_block(sar_mode):
    if sar_mode == "range_offset":
        return """
    prefix: null
    value: "roff_20250319_20250331.phs"
    metadata: "roff_20250319_20250331.phs.rsc"
    geometry:
      azimuth: "off_20250319_20250331.azi"
      incidence: "off_20250319_20250331.inc"
"""
    if sar_mode == "azimuth_offset":
        return """
    prefix: null
    value: "azoff_20250319_20250331.phs"
    metadata: "azoff_20250319_20250331.phs.rsc"
    geometry:
      azimuth: "off_20250319_20250331.azi"
      incidence: "off_20250319_20250331.inc"
"""
    return """
    prefix: "geo_20250319_20250331"
"""


def tiff_file_block(reader):
    prefix = "gamma_tiff_20250319_20250331" if reader == "gamma_tiff" else "hyp3_20250319_20250331"
    return f"""
    prefix: "{prefix}"
"""


def gmtsar_file_block(sar_mode):
    if sar_mode == "azimuth_offset":
        return """
    value: "azimuth_range/T33D_az.grd"
    projection:
      east: "enu_az/e.grd"
      north: "enu_az/n.grd"
"""
    if sar_mode == "range_offset":
        return """
    value: "azimuth_range/T33D_range.grd"
    projection:
      east: "enu_range/e_sample.grd"
      north: "enu_range/n_sample.grd"
      up: "enu_range/u_sample.grd"
"""
    if sar_mode == "unwrapped_phase":
        return """
    value: "phase/T33D_phasefilt_ll.grd"
    projection:
      east: "enu/e.grd"
      north: "enu/n.grd"
      up: "enu/u.grd"
"""
    return """
    value: "los/T33D_los_m.grd"
    projection:
      east: "enu/e.grd"
      north: "enu/n.grd"
      up: "enu/u.grd"
"""


def sar_read_block(sar_reader, factor_to_m, byte_order, template):
    if sar_reader == "gmtsar":
        grid = ""
        if template == "full":
            grid = """  grid:
    engine: null
    value_variable: null
    projection_variable: null
    lon_name: null
    lat_name: null
    coord_is_lonlat: null
"""
        return f"""  read:
    downsample: 1              # used for quick-look and downsample runs
    downsample_for_covar: 1    # used only when estimating covariance
    zero2nan: true
    wavelength: null           # required for unwrapped_phase
    factor_to_m: {factor_to_m} # direct displacement unit scale; keep 1.0 for phase radians
{grid}"""
    byte_order_line = (
        f'    byte_order: "{byte_order}"   # native preserves the historical GAMMA default\n'
        if sar_reader == "gamma"
        else ""
    )
    grid = ""
    if template == "full" and sar_reader != "gamma":
        grid = """  grid:
    phase_band: 1
    azi_band: 1
    inc_band: 1
    coord_is_lonlat: null
"""
    return f"""  read:
    downsample: 1              # used for quick-look and downsample runs
    downsample_for_covar: 1    # used only when estimating covariance
    zero2nan: true
    wavelength: null           # GAMMA .rsc metadata is used when available
    factor_to_m: {factor_to_m} # displacement/offset unit scale; keep 1.0 for phase radians
{byte_order_line}{grid}"""


def build_sar_config_text(
    sar_reader, sar_mode, template, acquisition_look_side="right",
    byte_order="native"
):
    if sar_reader == "hyp3" and sar_mode not in ("los_displacement", "unwrapped_phase"):
        raise ValueError(
            "HyP3 template supports sar_mode='los_displacement' or "
            "'unwrapped_phase'."
        )

    if sar_reader == "gamma":
        files = gamma_file_block(sar_mode)
    elif sar_reader == "gmtsar":
        files = gmtsar_file_block(sar_mode)
    else:
        files = tiff_file_block(sar_reader)
    factor_to_m = 1.0
    if sar_mode == "unwrapped_phase":
        plot_vmin, plot_vmax = -30, 30
    elif sar_mode == "azimuth_offset":
        plot_vmin, plot_vmax = -300, 300
    else:
        plot_vmin, plot_vmax = -30, 30

    azimuth_role = {
        "gamma": "sensor_to_ground",
        "gamma_tiff": "heading",
        "hyp3": "ground_to_sensor",
    }.get(sar_reader)

    advanced = ""
    if template == "full":
        if sar_reader == "gmtsar":
            projection_axis = "azimuth" if sar_mode == "azimuth_offset" else "los"
            projection_direction = (
                "along_heading"
                if projection_axis == "azimuth"
                else "ground_to_sensor"
            )
            advanced = f"""
  # Advanced: describe the supplied ENU projection only when it differs
  # from this mode's built-in GMTSAR convention.
  # projection_convention:
  #   input_projection_axis: "{projection_axis}"
  #   input_projection_direction: "{projection_direction}"
"""
        else:
            advanced = f"""
  # Advanced: describe the angle rasters only when they differ from this
  # reader's built-in convention.
  # geometry_convention:
  #   azimuth_angle_role: "{azimuth_role}"
"""

    side_line = (
        f'  acquisition_look_side: "{acquisition_look_side}"  # right | left\n'
        if sar_reader != "gmtsar"
        else ""
    )
    return f"""
sar_config:
  outName: "S1_example"
  reader: "{sar_reader}"       # gamma | gamma_tiff | gmtsar | hyp3
  mode: "{sar_mode}"           # unwrapped_phase | los_displacement | range_offset | azimuth_offset
{side_line}  directory: ".."
  output_check: true
  output_suffix: "auto"        # auto appends _RngOff/_AziOff for range/azimuth offsets if not already present
{advanced}
  files:{files}
{sar_read_block(sar_reader, factor_to_m, byte_order, template)}
  data_filters:
    enabled: false
    report: true
    report_file: "auto"
    rules:
      - name: "valid_observation_range"
        enabled: false
        kind: "value_range"
        value_space: "observation"
        min: null
        max: null
  qc:
    summary_percentile: 99.0
"""


def processing_region_config_text():
    return """
processing_region:
  enabled: false
  report: true
  report_file: "auto"
  coord_type: "lonlat"     # lonlat | xy
  geometry: "box"          # box | polygon | polygon_file
  box: null                # [minlon, maxlon, minlat, maxlat]; longitude matches modulo 360
  polygon: null            # inline [[lon, lat], ...] or [[x, y], ...]
  polygon_file: null       # text file with two columns, relative to the config file
"""


def observation_correction_config_text(template="minimal"):
    advanced = """
  # Replace fixed_coefficients: null with the following fixed-plane schema:
  # fixed_coefficients:
  #   origin: [lon, lat]
  #   <component>:  # observation for SAR; east and north for optical
  #     offset: 0.0
  #     east_gradient: 0.0
  #     north_gradient: 0.0
  # Polygon regions and complete fixed-coefficient examples are in the reference.
""" if template == "full" else ""
    return f"""
# Optional deterministic preprocessing before covariance/downsampling.
# Keep enabled=false when only quick-look, covariance, or downsampling is required.
# Common: offset + one stable far-field circle sets the zero reference.
# Long-wave ramp: plane + all valid data, optionally excluding deformation boxes.
# Formula: corrected = observation - correction_surface.
# estimate: offset/plane coefficients are solved automatically.
# fixed: leave fit regions/exclusions empty and provide fixed_coefficients explicitly.
observation_correction:
  enabled: false
  model: "offset"               # offset | plane
  coefficient_mode: "estimate" # estimate | fixed
  report: true
  report_file: "auto"           # auto = <outName>_observation_correction.yml
  fit:
    coord_type: "lonlat"
    regions: []                 # offset: required; plane: [] = all valid observations
    exclude_regions: []         # plane: optional deformation/noise exclusions
  fixed_coefficients: null      # fixed: component coefficients; plane also requires origin

# Offset example - replace regions: [] above with:
#     regions:
#       - kind: "circle"
#         center: [lon, lat]    # stable far-field zero-reference area
#         radius_km: 20.0
# Plane example - keep regions: []; replace exclude_regions: [] above with:
#     exclude_regions:
#       - kind: "box"
#         bounds: [min_lon, max_lon, min_lat, max_lat]
#         # deformation/noisy area excluded from ramp estimation
{advanced}
"""


def observation_export_config_text():
    return """
# Optional reusable data/display exports; scientific values are not resampled.
export:
  observation_grid:
    enabled: false
    format: "netcdf"            # canonical CF-NetCDF format
    file: "auto"                # auto = <outName>_observation.nc; explicit .h5/.hdf5 is supported
    geotiff_sidecar: "auto"     # auto writes exact-affine TIFF sidecars when possible
    # GAMMA binary normally has no affine/CRS, so auto writes NetCDF only.
    verify: true                # reopen output and verify values/coordinates
    report: true
    report_file: "auto"         # auto = <outName>_observation_grid.yml
  google_earth:
    enabled: false              # run: ecat-downsample -f this_file.yml
    file: "auto"                # auto = <outName>_google_earth.kmz
    mask: "source_valid"        # full valid source pixels; analysis_valid shows only the analysis subset
    visible: true               # initial checkbox state in Google Earth; this is not a style field
    style:
      cmap: "RdBu_r"
      display_factor: 100.0     # stored meters remain unchanged; display as cm
      display_unit: "cm"
      vmin: null                # display color minimum in display_unit; not a geographic range
      vmax: null                # set both vmin/vmax or leave both null for auto
      symmetry: true            # auto limits are centered on zero; explicit vmin/vmax win
"""


def optical_config_text():
    return """
optical_config:
  outName: "Optical_S2_part1"
  directory: ".."
  filename: "Sagaing_S2_Part1.tif"
  vel_type: "north"       # north | east; component used by trirb downsampling
  read:
    downsample: 1              # used for quick-look and downsample runs
    downsample_for_covar: 1    # used only when estimating covariance
    zero2nan: true             # convert zero-valued pixels to NaN before CSI import
    remove_nan: true           # drop pixels where east or north is NaN
    factor_to_m: 10.0          # product unit scale to meters
  grid:
    ew_band: 1
    sn_band: 2
  output_check: true
  data_filters:
    enabled: false
    report: true
    report_file: "auto"
    rules:
      - name: "valid_horizontal_component_range"
        enabled: false
        kind: "component_range"
        components: ["east", "north"]
        min: null
        max: null
  qc:
    summary_percentile: 99.0
"""


def std_downsample_config_text():
    return """  # Used only when method: std.
  std_config:
    startingsize: 5.0       # initial square block size in CSI projected units, usually km
    minimumsize: 0.25       # smallest allowed block size
    min_valid_fraction: 0.1 # minimum valid-pixel fraction to keep a block
    split_std_threshold: 0.005 # split blocks until within-block std is below this value
    split_metric_correction: "median" # robust default; set std to reproduce CSI-compatible tuning
    split_metric_smoothing: null # optional smoothing length for split metric; null disables it
    use_variance: false     # false = compare std; true = compare variance with split_std_threshold
    amplitude_stat: "mean_abs" # mean_abs | abs_mean | median_abs; used by high/low amplitude controls
    focus_region:
      enabled: false        # true limits refinement outside the region below
      coord_type: "lonlat"  # focus box coordinates are lon/lat
      box: null             # [minlon, maxlon, minlat, maxlat]
      polygon: null         # inline [[lon, lat], ...]
      polygon_file: null    # text file with two columns; relative to config file
      max_splits_outside: 5 # maximum std split level outside the focus region
    high_value_refinement:
      enabled: false        # true forces extra splits where block amplitude is large
      high_value_ratio: 0.25 # amplitude threshold as fraction of reference max value
      min_splits: 1         # minimum split level for high-value blocks
      reference_max_value: null # null = use max(abs(input value))
    low_amplitude_cap:
      enabled: false        # true caps splitting in low-amplitude, likely noisy areas
      amplitude_ratio: 0.05 # low amplitude threshold as fraction of reference max value
      max_splits: 3         # maximum split level for low-amplitude blocks
      reference_max_value: null # null = share CSI reference max value
      apply_inside_focus_region: false # false keeps the focus region free to refine
    plot: false             # pass plotting flag to CSI during downsampling iterations
    decimorig: 10           # plot decimation used only when plot: true
    itmax: 100              # maximum std-based splitting iterations
    verboseLevel: "minimum"
"""


def data_downsample_config_text():
    return """  # Used only when method: data.
  data_config:
    startingsize: 5.0       # initial square block size in CSI projected units, usually km
    minimumsize: 0.25       # smallest allowed block size
    min_valid_fraction: 0.1 # minimum valid-pixel fraction to keep a block
    split_metric_threshold: 0.01 # split blocks when the chosen data metric exceeds this value
    split_metric: "curvature" # curvature | gradient
    split_metric_smoothing: null # optional smoothing length for the data metric; null disables it
    plot: false             # pass plotting flag to CSI during downsampling iterations
    decimorig: 10           # plot decimation used only when plot: true
    itmax: 100              # maximum data-based splitting iterations
    verboseLevel: "minimum"
"""


def trirb_downsample_config_text():
    return """  # Used only when method: trirb; also enable triangular fault_models below.
  trirb_config:
    minimumsize: 1.25      # smallest triangle/block size in CSI projected units, usually km
    min_valid_fraction: 0.1 # minimum valid-pixel fraction to keep an initial triangle
    max_samples: 2500      # target maximum number of downsampled observations
    change_threshold: 10   # stop when sample-count change is below this percent
    smooth_factor: 0.25    # smoothing weight used by the resolution criterion
    slipdirection: "sd"    # fault slip components for Green functions: s, d, t, or combinations
    plot: false            # pass plotting flag to CSI during downsampling iterations
    decimorig: 10          # plot decimation used only when plot: true
    verboseLevel: "maximum"
"""


def extraction_config_text():
    return """  extraction:
    value_statistic: "median"     # mean | median | center_nearest | trimmed_mean
    error_statistic: "std"        # std | mad | sem | none
    coordinate_statistic: "mean"  # mean | block_center | center_nearest
    projection_statistic: "mean_normalized" # mean_normalized | center_nearest
    trim_fraction: 0.1            # used only by value_statistic: trimmed_mean
"""


def guide_grid_config_text():
    return """  guide_grid:
    enabled: false                # true enables Guided Quadtree Downsampling for std/data
    source: "filtered_observation" # currently filtered_observation
    component: "auto"             # auto: SAR observation; optical both components
    filter:
      kind: "gaussian"            # gaussian | median
      sigma: null                 # gaussian only; required when enabled
      unit: "km"                  # gaussian only: km | pixel
      radius_sigma: 3.0           # gaussian only: search radius / sigma
      # window_size: 3            # median only; odd pixel window >= 3 (default: 3)
"""


def downsample_report_config_text():
    return """  report:
    enabled: true                 # write <outName>_downsample_report.yml for tuning diagnostics
    report_file: "auto"           # auto | path | null/false
    quality: true                 # include reconstruction residual diagnostics when available
"""


def check_plots_config_text(data_type="sar", template="minimal"):
    raw_figsize = "double" if data_type == "optical" else "single"
    raw_figsize_note = (
        "typical optical two-column; try [7, 5] or [8, 5] for tall maps"
        if data_type == "optical"
        else "typical SAR single plot; try [4, 5] for tall maps"
    )
    full = template == "full"
    raw_contour_advanced = """
      labels: false               # true writes contour values in displayed units
      color: "0.20"               # Matplotlib color for contour lines
      linewidth: 0.5
      alpha: 0.8
""" if full else ""
    raw_advanced = """
    components: "auto"            # SAR: observation; optical: east+north
    layout: "auto"                # auto | single | columns
    auto_percentile: null         # null = qc.summary_percentile
    style_context: "science"
    axis_max_major_ticks: 5       # null keeps Matplotlib/viztools automatic ticks
    axis_minor_ticks: false       # true enables minor ticks explicitly
    axis_minor_subdivisions: 2
    colorbar_label: "auto"
    colorbar_mode: "outside"      # outside | inside
    colorbar_loc: null            # null = right for vertical, centered bottom for horizontal
    colorbar_minor_ticks: false
    colorbar_minor_subdivisions: 2
    cb_label_loc: null
    tickfontsize: null            # null = auto from figsize; explicit number fixes colorbar tick labels
    labelfontsize: null           # null = auto from figsize; explicit number fixes colorbar labels
    trace_color: "black"
    trace_linewidth: 0.5
""" if full else ""
    decim_advanced = """
    components: "auto"            # optical auto writes east+north in one two-column figure
    layout: "auto"
    auto_percentile: null         # null = qc.summary_percentile
    style_context: "science"
    axis_max_major_ticks: 5
    axis_minor_ticks: false
    axis_minor_subdivisions: 2
    colorbar_label: "auto"
    colorbar_mode: "outside"      # outside | inside
    colorbar_loc: null            # null = right for vertical, centered bottom for horizontal
    colorbar_minor_ticks: false
    colorbar_minor_subdivisions: 2
    cb_label_loc: null
    tickfontsize: null            # null = auto from figsize; explicit number fixes colorbar tick labels
    labelfontsize: null           # null = auto from figsize; explicit number fixes colorbar labels
    trace_color: "black"
    trace_linewidth: 0.5
""" if full else ""
    return f"""
check_plots:
  raw:
    show: true
    save_fig: true
    file_path: "auto"             # auto = sar_values.png or <outName>_deformation_map.jpg
    coordrange: null              # display extent only; longitude matches modulo 360
    plot_stride: 1                # quick-look stride on the raw grid
    figsize: "{raw_figsize}"             # {raw_figsize_note}
    dpi: 300                      # saved-output dpi; screen display is capped internally
    fontsize: null                 # null = auto from figsize, about 6-10 pt
    factor4plot: "auto"           # SAR auto = 100 (m to cm); optical auto = 1
    vmin: null                    # null = robust auto color limit
    vmax: null
    symmetry: true
    cmap: "cmc.roma_r"
    contours:
      enabled: false              # optional diagnostic layer for structured raw 2-D grids
      levels: "auto"              # auto | integer count | increasing values in displayed units
{raw_contour_advanced}
    axis_tick_direction: "out"    # map-axis ticks: out | in | inout
    colorbar_orientation: "auto"  # auto = vertical for single, horizontal for optical columns
    colorbar_pad: null            # null = layout default; smaller/larger moves outside colorbar
    colorbar_size: null           # null = layout default; colorbar length relative to map axis
    colorbar_thickness: null      # null = layout default; colorbar thickness relative to map axis
    panel_pad: null               # null = compact; panel gap as a fraction of figure width
    colorbar_tick_direction: "out" # out | in | inout
    colorbar_max_major_ticks: 3
{raw_advanced}

  decim:
    show: false                   # true displays the figure after downsampling
    save_fig: true
    file_path: "auto"             # auto = <outName>_decim.png
    coordrange: null              # display extent only; longitude matches modulo 360
    cell_style: "cells"           # cells = draw CSI cells; points = sample centers
    figsize: "double"             # typical decim check; use single for compact SAR, [7, 5] for optical columns
    dpi: 300                      # saved-output dpi; screen display is capped internally
    fontsize: null                 # null = auto from figsize, about 6-10 pt
    factor4plot: "inherit_raw"
    vmin: null                    # optical also accepts [east, north]
    vmax: null
    symmetry: true
    cmap: "cmc.roma_r"
    axis_tick_direction: "out"    # map-axis ticks: out | in | inout
    colorbar_orientation: "auto"  # auto = vertical for single, horizontal for optical columns
    colorbar_pad: null
    colorbar_size: null
    colorbar_thickness: null
    panel_pad: null               # minimum gap between neighboring map+colorbar panels
    colorbar_tick_direction: "out"
    colorbar_max_major_ticks: 3
    edgewidth: 0.1
    edgecolor: "black"
    alpha: 1.0
    markersize: 10
{decim_advanced}
"""


def from_rsp_downsample_config_text():
    return """  # Used only when method: from_rsp.
  from_rsp_config:
    rsp_file: "reference_ifg.rsp" # existing CSI .rsp grid to reuse; suffix optional; 10/18-col rectangle or 8-col triangle
    geometry: "auto"      # auto | rectangle | triangle
    min_valid_fraction: 0.0 # minimum valid-pixel fraction inside each reused cell
    plot: false           # pass plotting flag to CSI during the resampling pass
    decimorig: 10         # plot decimation used only when plot: true
"""


def general_config_text():
    return """
general:
  origin: "auto"           # auto = use input data center; manual = use lon0/lat0 below
  lon0: null               # required only when origin: manual
  lat0: null               # required only when origin: manual
"""


def processing_config_text(template, downsample_method, data_type="sar"):
    include_std = template == "full" or downsample_method == "std"
    include_data = template == "full" or downsample_method == "data"
    include_trirb = template == "full" or downsample_method == "trirb"
    include_from_rsp = template == "full" or downsample_method == "from_rsp"
    std_config = std_downsample_config_text() if include_std else ""
    data_config = data_downsample_config_text() if include_data else ""
    trirb_config = trirb_downsample_config_text() if include_trirb else ""
    from_rsp_config = from_rsp_downsample_config_text() if include_from_rsp else ""
    method_note = {
        "std": "std is fault-free and easiest for first tests; trirb needs fault geometry",
        "data": "data is fault-free and splits by curvature or gradient",
        "trirb": "trirb is fault-aware; use std for a fault-free first test",
        "from_rsp": "from_rsp reuses an existing CSI .rsp grid; no fault geometry required",
    }[downsample_method]
    fault_model_use_for = '["trirb"]' if downsample_method == "trirb" else "[]"
    fault_model_use_note = (
        "preselected for this TriRB template; enabled must also be true"
        if downsample_method == "trirb"
        else 'set to ["trirb"] for resolution-based downsampling'
    )
    return f"""
covar:
  do_covar: false          # can also be enabled with ecat-downsample -c
  mask_out: null           # required for covariance; longitude matches modulo 360
  missing_policy: "existing_or_identity" # when downsampling without covariance: existing_or_identity | identity | error
  function: "exp"          # exp | gauss covariance model
  frac: 0.002              # fraction of remaining background pixels sampled by CSI imagecovariance
  every: 2.0               # distance-bin spacing for empirical covariance, usually in km
  distmax: 100.0           # maximum fitting distance, same unit as every
  rampEst: true            # ask CSI imagecovariance to estimate/remove a ramp before fitting

downsample:
  enabled: false           # can also be enabled with ecat-downsample -d
  compute:
    cutde_backend: "cpp"   # cpp | cuda | opencl | auto; mainly affects trirb/cutde GF calls
  method: "{downsample_method}"          # {method_note}
{extraction_config_text()}
{guide_grid_config_text()}
{downsample_report_config_text()}
{std_config}
{data_config}
{trirb_config}
{from_rsp_config}
{check_plots_config_text(data_type, template)}
fault_traces:
  - enabled: false
    id: "example_trace"
    file: "../../../../Faults/example_rupture_trace.dat"
    # segments: "all"          # optional: all | zero-based integer | integer list
    stages: ["raw", "decim"]   # raw | decim | all; display only
    marker: null                 # null = line only; set "x" to show input trace vertices
    markersize: 3.0              # used only when marker is not null

# Repeat or mix entries as needed. Every enabled triangular model with
# use_for: ["trirb"] participates jointly; plot.stages controls display only.
fault_models:
  - enabled: false
    id: "example_generated_triangular_fault"
    type: "generated_from_trace"   # model input: build from trace | read CSI GMT
    geometry: "triangular"         # patch shape: triangular | rectangular; trirb requires triangular
    trace_file: "../../../../Faults/example_rupture_trace.dat"
    # segment: 0                    # required only when trace_file contains multiple segments
    dip_angle: 86                  # degrees from horizontal
    dip_direction: 258             # degrees clockwise from north
    top_size: 5.0                  # top mesh target size, usually km
    bottom_size: 8.0               # bottom mesh target size, usually km
    top_depth: 0.0                 # km
    bottom_depth: 18.0             # km
    use_for: {fault_model_use_for} # {fault_model_use_note}
    plot:
      stages: []                   # e.g. ["decim"] to draw patch edges
      mode: "edges"                # edges | trace | both

  - enabled: false
    id: "example_csi_gmt_fault"
    type: "csi_gmt"
    geometry: "triangular"
    file: "fault_mesh.gmt"
    readpatchindex: true           # true when each GMT segment header stores 3 topology indices
    donotreadslip: true            # geometry-only use: ignore any slip columns
    gmtslip: true                  # triangular only: true when the segment header contains -Z...
    use_for: {fault_model_use_for} # {fault_model_use_note}
    plot:
      stages: []
      mode: "edges"
"""


def build_config(
    mode, sar_reader, sar_mode, template, downsample_method="std",
    acquisition_look_side="right", byte_order="native"
):
    data_type = mode if mode in ("sar", "optical") else "sar"
    blocks = [f'config_version: 1\ndata_type: "{data_type}"\n']
    blocks.append(general_config_text())

    if mode in (None, "full", "sar"):
        blocks.append(
            build_sar_config_text(
                sar_reader,
                sar_mode,
                template,
                acquisition_look_side=acquisition_look_side,
                byte_order=byte_order,
            )
        )
    if mode in (None, "full", "optical"):
        blocks.append(optical_config_text())
    blocks.append(observation_correction_config_text(template))
    blocks.append(observation_export_config_text())
    blocks.append(processing_region_config_text())
    blocks.append(processing_config_text(template, downsample_method, data_type))
    return "\n".join(blocks)


def generate_downsample_config(
    output_path,
    mode=None,
    copy_adapter_template=False,
    sar_reader="gamma",
    sar_mode="unwrapped_phase",
    template="minimal",
    downsample_method="std",
    acquisition_look_side="right",
    byte_order="native",
):
    """
    Generate a downsampling configuration file.

    Parameters
    ----------
    output_path : str
        Output YAML file path.
    mode : {"sar", "optical", "full", None}
        Which template sections to include. None keeps the historical full
        template behavior.
    copy_adapter_template : bool
        Copy the adapter_downsampling template files into the output directory.
    sar_reader : {"gamma", "gamma_tiff", "gmtsar", "hyp3"}
        SAR reader family for SAR templates.
    sar_mode : {"unwrapped_phase", "los_displacement", "range_offset", "azimuth_offset"}
        SAR observation mode for SAR templates.
    template : {"minimal", "full"}
        "full" includes all downsampling method configs, advanced check-plot
        layout fields, and advanced SAR convention hints.
    downsample_method : {"std", "data", "trirb", "from_rsp"}
        Method written as downsample.method. Minimal templates include only
        this method's config. TriRB templates also preselect the disabled
        triangular examples for the ``trirb`` compute role.
    acquisition_look_side : {"right", "left"}
        Side of platform heading containing the imaged swath.
    byte_order : {"native", "little", "big"}
        GAMMA raw-float32 byte order. Native preserves historical behavior.
    """
    if mode == "full":
        mode = None
    if mode not in (None, "sar", "optical"):
        raise ValueError("mode must be 'sar', 'optical', 'full', or None.")
    if sar_reader not in SAR_READER_CHOICES:
        raise ValueError(f"sar_reader must be one of: {', '.join(SAR_READER_CHOICES)}.")
    if sar_mode not in SAR_MODE_CHOICES:
        raise ValueError(f"sar_mode must be one of: {', '.join(SAR_MODE_CHOICES)}.")
    if template not in ("minimal", "full"):
        raise ValueError("template must be 'minimal' or 'full'.")
    if downsample_method not in DOWNSAMPLE_METHOD_CHOICES:
        raise ValueError(f"downsample_method must be one of: {', '.join(DOWNSAMPLE_METHOD_CHOICES)}.")
    if acquisition_look_side not in ("right", "left"):
        raise ValueError("acquisition_look_side must be 'right' or 'left'.")
    if byte_order not in ("native", "little", "big"):
        raise ValueError("byte_order must be native, little, or big.")

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    config = yaml.load(
        build_config(
            mode,
            sar_reader,
            sar_mode,
            template,
            downsample_method,
            acquisition_look_side=acquisition_look_side,
            byte_order=byte_order,
        )
    )
    if copy_adapter_template:
        config["input_adapter"] = {"enabled": True}

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        yaml.dump(config, file)

    print(f"Downsampling configuration file generated at: {output_path}")

    if copy_adapter_template:
        copied = copy_adapter_downsampling_template(output_path.parent)
        for path in copied:
            print(f"Adapter downsampling template copied to: {path}")


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Generate a downsampling configuration file.")
    parser.add_argument(
        "-m",
        "--mode",
        type=str,
        choices=["sar", "optical", "full"],
        default="sar",
        help="Template mode: sar, optical, or full. Default: sar.",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="downsample_config.yml",
        help="Output path for the configuration file.",
    )
    parser.add_argument(
        "--sar-reader",
        choices=SAR_READER_CHOICES,
        default="gamma",
        help="SAR reader family for SAR templates.",
    )
    parser.add_argument(
        "--sar-mode",
        choices=SAR_MODE_CHOICES,
        default="unwrapped_phase",
        help="SAR observation mode for SAR templates.",
    )
    parser.add_argument(
        "--sar-look-side",
        choices=["right", "left"],
        default="right",
        help=(
            "Acquisition look side written for angle-raster readers. "
            "Default: right."
        ),
    )
    parser.add_argument(
        "--sar-byte-order",
        choices=["native", "little", "big"],
        default="native",
        help="GAMMA float32 byte order. Default: native.",
    )
    parser.add_argument(
        "--template",
        choices=["minimal", "full"],
        default="minimal",
        help="Use 'full' to include all method configs plus advanced check-plot and SAR convention fields.",
    )
    parser.add_argument(
        "--downsample-method",
        choices=DOWNSAMPLE_METHOD_CHOICES,
        default="std",
        help="Downsampling method written to the template. Default: std.",
    )
    parser.add_argument(
        "--copy-adapter-template",
        "--copy_adapter_template",
        action="store_true",
        help="Copy adapter_downsampling template files to the output config directory.",
    )
    args = parser.parse_args()

    generate_downsample_config(
        Path(args.output).resolve(),
        mode=args.mode,
        copy_adapter_template=args.copy_adapter_template,
        sar_reader=args.sar_reader,
        sar_mode=args.sar_mode,
        template=args.template,
        downsample_method=args.downsample_method,
        acquisition_look_side=args.sar_look_side,
        byte_order=args.sar_byte_order,
    )


if __name__ == "__main__":
    main()
