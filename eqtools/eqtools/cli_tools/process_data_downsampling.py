import argparse
import inspect
import os
import re
import warnings
from copy import copy, deepcopy
from datetime import datetime, timezone
from glob import glob
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from eqtools.csiExtend.downsample.config import (
    compact_kwargs,
    get_sar_output_name,
    normalize_downsample_config,
    normalize_optical_config,
    normalize_sar_config,
)
from eqtools.csiExtend.downsample.data_filters import (
    apply_data_filters,
    filter_report_file,
    format_filter_report,
)
from eqtools.csiExtend.downsample.fault_inputs import (
    load_fault_models_for_compute,
    load_plot_fault_overlays,
)
from eqtools.csiExtend.downsample.grid_template import (
    apply_rsp_grid_template,
    read_rsp_grid_template,
)
from eqtools.csiExtend.downsample.plotting import plot_decimated_geodata
from eqtools.csiExtend.downsample.processing_region import (
    apply_processing_region,
    format_processing_region_report,
    processing_region_report_file,
)
from eqtools.csiExtend.downsample.report import (
    build_downsample_report,
    downsample_report_file,
    format_downsample_report,
    write_downsample_report,
)
from eqtools.csiExtend.compute.backend import (
    configure_cutde_backend,
    cutde_backend_summary,
)

try:
    import cmcrameri as cmc  # noqa: F401  # register cmc.* colormaps for CSI plotting
except ModuleNotFoundError:
    cmc = None

PROCESSING_IMPORT_ERROR = None
PROCESSING_DEPENDENCIES_LOADED = False
gdal = None
osr = None
utm_zone_epsg = None
imcov = None
imagedownsampling = None
insar = None
opticorr = None
TriangularPatches = None
RectangularPatches = None
TiffoptiReader = None
GammasarReader = None
GmtsarReader = None
GammaTiffReader = None
Hyp3TiffReader = None
GammasarConfig = None
GmtsarConfig = None
GammaTiffConfig = None
Hyp3TiffConfig = None
SarReaderConfig = None


DEFAULT_CMAP = "cmc.roma_r" if cmc is not None else "RdBu_r"


SAR_READER_CLASSES = {}
SAR_CONFIG_CLASSES = {}

SAR_PLOT_KWARGS = {
    "save_fig",
    "file_path",
    "dpi",
    "show",
    "rawdownsample4plot",
    "value_space",
    "factor4plot",
    "vmin",
    "vmax",
    "symmetry",
    "cmap",
    "colorbar_orientation",
    "colorbar_mode",
    "colorbar_loc",
    "colorbar_size",
    "colorbar_thickness",
    "colorbar_pad",
    "colorbar_x",
    "colorbar_y",
    "colorbar_length",
    "colorbar_height",
    "cb_label",
    "cb_label_loc",
    "tickfontsize",
    "labelfontsize",
    "figsize",
    "fontsize",
    "style",
    "coordrange",
    "text",
    "text_position",
    "text_fontsize",
    "text_color",
}

OPTICAL_PLOT_KWARGS = {
    "save_fig",
    "file_path",
    "dpi",
    "show",
    "rawdownsample4plot",
    "coordrange",
    "data",
    "factor4plot",
    "vmin",
    "vmax",
    "symmetry",
    "cmap",
    "figsize",
    "title",
    "unified_colorbar",
    "add_colorbar",
    "cb_label",
    "trace_color",
    "trace_linewidth",
    "style",
    "fontsize",
    "colorbar_length",
    "colorbar_height",
    "colorbar_x",
    "colorbar_y",
    "colorbar_orientation",
    "cb_label_loc",
    "tickfontsize",
    "labelfontsize",
    "sharey",
    "equal_aspect",
}


def resolve_cmap(cmap):
    if cmc is None and isinstance(cmap, str) and cmap.startswith("cmc."):
        return "RdBu_r"
    return cmap


def load_processing_dependencies():
    global PROCESSING_IMPORT_ERROR, PROCESSING_DEPENDENCIES_LOADED
    global gdal, osr, utm_zone_epsg, imcov, imagedownsampling, insar, opticorr
    global TriangularPatches, RectangularPatches, TiffoptiReader
    global GammasarReader, GmtsarReader, GammaTiffReader, Hyp3TiffReader
    global GammasarConfig, GmtsarConfig, GammaTiffConfig, Hyp3TiffConfig, SarReaderConfig

    if PROCESSING_DEPENDENCIES_LOADED:
        return

    try:
        from osgeo import gdal as imported_gdal, osr as imported_osr

        from csi.csiutils import utm_zone_epsg as imported_utm_zone_epsg
        from csi.imagecovariance import imagecovariance as imported_imcov
        from csi.imagedownsampling import imagedownsampling as imported_imagedownsampling
        from csi.insar import insar as imported_insar
        from csi.opticorr import opticorr as imported_opticorr

        from eqtools.csiExtend.AdaptiveTriangularPatches import (
            AdaptiveTriangularPatches as ImportedTriangularPatches,
        )
        from csi.RectangularPatches import RectangularPatches as ImportedRectangularPatches
        from eqtools.csiExtend.optiUtils.readTiff2csiopti import (
            TiffoptiReader as ImportedTiffoptiReader,
        )
        from eqtools.csiExtend.sarUtils.readGamma2csisar import (
            GammasarReader as ImportedGammasarReader,
        )
        from eqtools.csiExtend.sarUtils.readGmtsar2csisar import (
            GmtsarReader as ImportedGmtsarReader,
        )
        from eqtools.csiExtend.sarUtils.readTiff2csisar import (
            GammaTiffReader as ImportedGammaTiffReader,
            Hyp3TiffReader as ImportedHyp3TiffReader,
        )
        from eqtools.csiExtend.sarUtils.sar_conventions import (
            GammaTiffConfig as ImportedGammaTiffConfig,
            GammasarConfig as ImportedGammasarConfig,
            GmtsarConfig as ImportedGmtsarConfig,
            Hyp3TiffConfig as ImportedHyp3TiffConfig,
            SarReaderConfig as ImportedSarReaderConfig,
        )
    except ModuleNotFoundError as exc:
        PROCESSING_IMPORT_ERROR = exc
        return

    gdal = imported_gdal
    osr = imported_osr
    utm_zone_epsg = imported_utm_zone_epsg
    imcov = imported_imcov
    imagedownsampling = imported_imagedownsampling
    insar = imported_insar
    opticorr = imported_opticorr
    TriangularPatches = ImportedTriangularPatches
    RectangularPatches = ImportedRectangularPatches
    TiffoptiReader = ImportedTiffoptiReader
    GammasarReader = ImportedGammasarReader
    GmtsarReader = ImportedGmtsarReader
    GammaTiffReader = ImportedGammaTiffReader
    Hyp3TiffReader = ImportedHyp3TiffReader
    GammasarConfig = ImportedGammasarConfig
    GmtsarConfig = ImportedGmtsarConfig
    GammaTiffConfig = ImportedGammaTiffConfig
    Hyp3TiffConfig = ImportedHyp3TiffConfig
    SarReaderConfig = ImportedSarReaderConfig

    SAR_READER_CLASSES.update(
        {
            "gamma": GammasarReader,
            "gammasar": GammasarReader,
            "gamma_binary": GammasarReader,
            "gamma_tiff": GammaTiffReader,
            "gammatiff": GammaTiffReader,
            "gmtsar": GmtsarReader,
            "hyp3": Hyp3TiffReader,
            "hyp3_tiff": Hyp3TiffReader,
        }
    )
    SAR_CONFIG_CLASSES.update(
        {
            "gamma": GammasarConfig,
            "gammasar": GammasarConfig,
            "gamma_binary": GammasarConfig,
            "gamma_tiff": GammaTiffConfig,
            "gammatiff": GammaTiffConfig,
            "gmtsar": GmtsarConfig,
            "hyp3": Hyp3TiffConfig,
            "hyp3_tiff": Hyp3TiffConfig,
        }
    )
    PROCESSING_IMPORT_ERROR = None
    PROCESSING_DEPENDENCIES_LOADED = True


def require_processing_dependencies():
    load_processing_dependencies()
    if PROCESSING_IMPORT_ERROR is not None:
        raise ModuleNotFoundError(
            "process_data_downsampling.py requires CSI and eqtools SAR/optical "
            "processing dependencies. Install those dependencies before reading "
            "or downsampling data."
        ) from PROCESSING_IMPORT_ERROR


def filter_supported_kwargs(callable_obj, kwargs):
    params = inspect.signature(callable_obj).parameters
    return {key: value for key, value in kwargs.items() if key in params and value is not None}


def single_file_match(pattern, label, exclude_suffixes=()):
    matches = sorted(glob(pattern))
    if exclude_suffixes:
        suffixes = tuple(suffix.lower() for suffix in exclude_suffixes)
        matches = [
            path
            for path in matches
            if not path.replace("\\", "/").lower().endswith(suffixes)
        ]

    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise FileNotFoundError(f"Could not find {label} matching {pattern!r}.")
    raise ValueError(f"Found multiple {label} files matching {pattern!r}: {', '.join(matches)}.")


def center_from_lonlat(lon, lat, label="input data"):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    lon_values = lon[np.isfinite(lon)]
    lat_values = lat[np.isfinite(lat)]
    if lon_values.size == 0 or lat_values.size == 0:
        raise ValueError(f"Cannot infer projection origin from {label}: no finite lon/lat values.")
    return (
        0.5 * (float(np.nanmin(lon_values)) + float(np.nanmax(lon_values))),
        0.5 * (float(np.nanmin(lat_values)) + float(np.nanmax(lat_values))),
    )


def load_config(config_file):
    with open(config_file, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_config_relative_path(path, config_dir=None):
    path = Path(path)
    if path.is_absolute() or config_dir is None:
        return path
    return Path(config_dir) / path


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Process SAR or optical data for covariance estimation and downsampling."
    )
    parser.add_argument(
        "--config",
        "-f",
        type=str,
        default="downsample_config.yml",
        help="Path to the configuration file.",
    )
    parser.add_argument("--do_covar", "-c", action="store_true", help="Perform covariance estimation.")
    parser.add_argument("--do_downsample", "-d", action="store_true", help="Perform downsampling.")
    parser.add_argument("--show_raw_data", "-s", action="store_true", help="Show a data quick-look plot.")
    parser.add_argument("--vmin", type=float, help="Minimum value for quick-look color scale.")
    parser.add_argument("--vmax", type=float, help="Maximum value for quick-look color scale.")
    parser.add_argument(
        "--workers",
        type=int,
        help=(
            "Number of CSI downsampling workers. Default is chosen by CSI "
            "(1 on Windows, auto on POSIX unless overridden by environment)."
        ),
    )
    return parser.parse_args()


def sar_config_object(reader_key, convention):
    require_processing_dependencies()
    config_cls = SAR_CONFIG_CLASSES.get(reader_key, SarReaderConfig)
    try:
        return config_cls(**(convention or {}))
    except TypeError as exc:
        raise ValueError(
            f"Unsupported SAR convention keys for reader={reader_key!r}: {exc}"
        ) from exc


def build_sar_reader(sar_config, lon0, lat0):
    require_processing_dependencies()
    reader_key = sar_config["reader"]
    try:
        reader_cls = SAR_READER_CLASSES[reader_key]
    except KeyError as exc:
        choices = ", ".join(sorted(SAR_READER_CLASSES))
        raise ValueError(f"Unsupported SAR reader {reader_key!r}. Available readers: {choices}.") from exc

    selected_semantics = [
        sar_config.get("mode") is not None,
        sar_config.get("preset") is not None,
        sar_config.get("convention") is not None,
    ]
    if sum(selected_semantics) > 1:
        raise ValueError("Use only one of sar_config.mode, sar_config.preset, or sar_config.convention.")

    reader_kwargs = {
        "name": "mysar",
        "lon0": lon0,
        "lat0": lat0,
        "directory_name": sar_config["directory"],
    }
    if sar_config.get("convention") is not None:
        reader_kwargs["config"] = sar_config_object(reader_key, sar_config["convention"])
    elif sar_config.get("preset") is not None:
        reader_kwargs["preset"] = sar_config["preset"]
    elif sar_config.get("mode") is not None:
        reader_kwargs["mode"] = sar_config["mode"]

    return reader_cls(**reader_kwargs)


def sar_extract_kwargs(sar_config):
    files = sar_config["files"]
    geometry = files.get("geometry", {}) or {}
    projection = files.get("projection", {}) or {}
    if sar_config["reader"] == "gmtsar":
        file_kwargs = compact_kwargs(
            {
                "valuefile": files.get("value"),
                "eastfile": projection.get("east"),
                "northfile": projection.get("north"),
                "upfile": projection.get("up"),
            }
        )
    else:
        explicit_files = compact_kwargs(
            {
                "phsname": files.get("value"),
                "rscname": files.get("metadata"),
                "azifile": geometry.get("azimuth"),
                "incfile": geometry.get("incidence"),
            }
        )
        file_kwargs = explicit_files if explicit_files else compact_kwargs({"prefix": files.get("prefix")})

    read = sar_config["read"]
    grid = sar_config.get("grid", {})
    extract_kwargs = {
        "directory_name": sar_config["directory"],
        **file_kwargs,
        "zero2nan": read.get("zero2nan"),
        "wavelength": read.get("wavelength"),
        "factor_to_m": read.get("factor_to_m"),
        "phase_band": grid.get("phase_band"),
        "azi_band": grid.get("azi_band"),
        "inc_band": grid.get("inc_band"),
        "is_lonlat": grid.get("coord_is_lonlat"),
        "value_variable": grid.get("value_variable"),
        "projection_variable": grid.get("projection_variable"),
        "east_variable": grid.get("east_variable"),
        "north_variable": grid.get("north_variable"),
        "up_variable": grid.get("up_variable"),
        "lon_name": grid.get("lon_name"),
        "lat_name": grid.get("lat_name"),
        "grid_engine": grid.get("engine"),
        "coord_is_lonlat": grid.get("coord_is_lonlat"),
    }
    return extract_kwargs


def sar_value_file_path(sar_config):
    files = sar_config["files"]
    directory = Path(sar_config["directory"])
    if files.get("value"):
        return directory / files["value"]

    prefix = files.get("prefix")
    if not prefix:
        raise ValueError("Set sar_config.files.prefix or files.value before using origin='auto'.")

    reader = sar_config["reader"]
    if reader == "gmtsar":
        raise ValueError("reader='gmtsar' requires explicit files.value before using origin='auto'.")
    if reader == "gamma":
        return Path(single_file_match(str(directory / f"{prefix}*.phs"), "GAMMA value file"))
    return Path(
        single_file_match(
            str(directory / f"{prefix}*.tif"),
            "SAR value TIFF",
            exclude_suffixes=(".azi.tif", ".inc.tif"),
        )
    )


def gamma_rsc_file_path(sar_config):
    files = sar_config["files"]
    directory = Path(sar_config["directory"])
    if files.get("metadata"):
        return directory / files["metadata"]
    prefix = files.get("prefix")
    if not prefix:
        raise ValueError("Set sar_config.files.prefix or files.metadata before using origin='auto'.")
    return Path(single_file_match(str(directory / f"{prefix}*.phs.rsc"), "GAMMA resource file"))


def infer_gamma_projection_origin(sar_config):
    rsc_file = gamma_rsc_file_path(sar_config)
    rsc = pd.read_csv(rsc_file, sep=r"\s+", names=["name", "value"])
    rsc.set_index("name", inplace=True)

    width = int(rsc.loc["WIDTH", "value"])
    length = int(rsc.loc["FILE_LENGTH", "value"])
    x_first = float(rsc.loc["X_FIRST", "value"])
    y_first = float(rsc.loc["Y_FIRST", "value"])
    x_step = float(rsc.loc["X_STEP", "value"])
    y_step = float(rsc.loc["Y_STEP", "value"])

    lon = [x_first, x_first + x_step * (width - 1)]
    lat = [y_first, y_first + y_step * (length - 1)]
    return center_from_lonlat(lon, lat, label=str(rsc_file))


def tiff_corners_from_geotransform(geotransform, width, height):
    def pixel_to_map(col, row):
        x = geotransform[0] + col * geotransform[1] + row * geotransform[2]
        y = geotransform[3] + col * geotransform[4] + row * geotransform[5]
        return x, y

    return [
        pixel_to_map(0, 0),
        pixel_to_map(width, 0),
        pixel_to_map(width, height),
        pixel_to_map(0, height),
    ]


def transform_tiff_corners_to_lonlat(corners, projection_wkt, is_lonlat=None):
    if is_lonlat is True:
        return corners
    if not projection_wkt:
        if is_lonlat is False:
            raise ValueError("GeoTIFF has no projection; set sar_config.grid.coord_is_lonlat=true or specify manual lon0/lat0.")
        return corners

    source = osr.SpatialReference()
    source.ImportFromWkt(projection_wkt)
    if is_lonlat is None and source.IsGeographic():
        return corners

    target = osr.SpatialReference()
    target.ImportFromEPSG(4326)
    if hasattr(source, "SetAxisMappingStrategy"):
        source.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
        target.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)
    transform = osr.CoordinateTransformation(source, target)

    lonlat = []
    for x, y in corners:
        lon, lat, _ = transform.TransformPoint(float(x), float(y))
        lonlat.append((lon, lat))
    return lonlat


def infer_tiff_projection_origin(tiff_file, is_lonlat=None):
    if gdal is None or osr is None:
        require_processing_dependencies()
    dataset = gdal.Open(str(tiff_file), gdal.GA_ReadOnly)
    if dataset is None:
        raise FileNotFoundError(f"Unable to open file: {tiff_file}")

    try:
        corners = tiff_corners_from_geotransform(
            dataset.GetGeoTransform(),
            dataset.RasterXSize,
            dataset.RasterYSize,
        )
        lonlat = transform_tiff_corners_to_lonlat(corners, dataset.GetProjection(), is_lonlat=is_lonlat)
    finally:
        del dataset

    lon, lat = zip(*lonlat)
    return center_from_lonlat(lon, lat, label=str(tiff_file))


def infer_gmtsar_projection_origin(sar_config):
    from eqtools.csiExtend.sarUtils.grid_io import read_lonlat_grid

    value_file = sar_value_file_path(sar_config)
    if not value_file.exists():
        raise FileNotFoundError(
            "GMTSAR value grid file not found while resolving origin='auto': "
            f"{value_file}. Check sar_config.directory and files.value."
        )
    grid = sar_config.get("grid", {})
    _, lon, lat, _, _ = read_lonlat_grid(
        value_file,
        variable=grid.get("value_variable"),
        lon_name=grid.get("lon_name"),
        lat_name=grid.get("lat_name"),
        engine=grid.get("engine"),
        coord_is_lonlat=grid.get("coord_is_lonlat"),
    )
    return center_from_lonlat(lon, lat, label=str(value_file))


def infer_sar_projection_origin(sar_config):
    sar_config = normalize_sar_config(sar_config)
    reader = sar_config["reader"]
    if reader == "gamma":
        return infer_gamma_projection_origin(sar_config)
    if reader == "gmtsar":
        return infer_gmtsar_projection_origin(sar_config)
    if reader in ("gamma_tiff", "hyp3"):
        return infer_tiff_projection_origin(
            sar_value_file_path(sar_config),
            is_lonlat=sar_config.get("grid", {}).get("coord_is_lonlat"),
        )
    raise ValueError(f"origin='auto' is not implemented for SAR reader {reader!r}.")


def infer_optical_projection_origin(optical_config):
    tiff_file = Path(optical_config.get("directory", "..")) / optical_config["filename"]
    return infer_tiff_projection_origin(tiff_file, is_lonlat=None)


def print_projection_origin(config):
    general = config["general"]
    lon0 = float(general["lon0"])
    lat0 = float(general["lat0"])
    utmzone, epsg = utm_zone_epsg(lon0, lat0)
    general["utmzone"] = int(utmzone)
    general["epsg"] = int(epsg)
    print("Projection origin:")
    print(f"  mode : {general.get('origin', 'manual')}")
    print(f"  lon0 : {lon0:.8g}")
    print(f"  lat0 : {lat0:.8g}")
    print(f"  UTM zone : {utmzone}")
    print(f"  EPSG : {epsg}")
    print("  note : used for local x/y coordinates during downsampling")
    return lon0, lat0


def resolve_projection_origin(config):
    require_processing_dependencies()
    general = config["general"]
    origin = general.get("origin", "manual")

    if origin == "auto":
        if config["data_type"] == "sar":
            lon0, lat0 = infer_sar_projection_origin(config["sar_config"])
        elif config["data_type"] == "optical":
            lon0, lat0 = infer_optical_projection_origin(config["optical_config"])
        else:
            raise ValueError(f"Unsupported data type: {config['data_type']}")
        general["lon0"] = float(lon0)
        general["lat0"] = float(lat0)
        general["origin_resolved_from"] = "input_data_center"
    else:
        general["lon0"] = float(general["lon0"])
        general["lat0"] = float(general["lat0"])
        general["origin_resolved_from"] = "user"

    return print_projection_origin(config)


def sar_read_observation_kwargs(sar_config, do_covar):
    read = sar_config["read"]
    downsample = read.get("downsample_for_covar") if do_covar else read.get("downsample")
    downsample = 1 if downsample is None else int(downsample)
    kwargs = {
        "downsample": downsample,
        "zero2nan": read.get("zero2nan"),
        "wavelength": read.get("wavelength"),
    }
    kwargs.update(read.get("overrides", {}) or {})
    return kwargs


def process_sar_data(sar_config, lon0, lat0, do_covar=False, config_dir=None):
    sar_config = normalize_sar_config(sar_config)
    mysar = build_sar_reader(sar_config, lon0, lat0)

    extract_kwargs = filter_supported_kwargs(mysar.extract_raw_grd, sar_extract_kwargs(sar_config))
    if not any(key in extract_kwargs for key in ("prefix", "phsname", "valuefile")):
        raise ValueError(
            "SAR files are not configured. Set sar_config.files.prefix or "
            "explicit files.value/metadata/geometry/projection entries."
        )
    mysar.extract_raw_grd(**extract_kwargs)

    read_kwargs = filter_supported_kwargs(
        mysar.read_observation,
        sar_read_observation_kwargs(sar_config, do_covar=do_covar),
    )
    mysar.read_observation(**read_kwargs)

    mysar.checkZeros()
    mysar.checkNaNs()
    mysar.checkLosEqualsOne()

    filter_report = apply_data_filters(
        mysar,
        sar_config.get("data_filters", {}),
        out_name=get_sar_output_name(sar_config),
        base_dir=config_dir,
    )
    mysar.data_filter_report = filter_report
    filter_report_text = format_filter_report(filter_report)
    if filter_report_text:
        print(filter_report_text)

    mysar.print_input_summary(central_percentile=sar_config["qc"].get("summary_percentile", 99.0))
    return mysar, sar_config


def process_optical_data(optical_config, lon0, lat0, config_dir=None):
    require_processing_dependencies()
    optical_config = normalize_optical_config(optical_config)
    filename = optical_config["filename"]
    opti_data = TiffoptiReader(
        "myopti",
        lon0=lon0,
        lat0=lat0,
        directory_name=optical_config.get("directory", ".."),
    )
    opti_data.extract_raw_grd(
        filename=filename,
        factor_to_m=optical_config.get("factor_to_m", 10.0),
        ew_band=optical_config.get("ew_band", 1),
        sn_band=optical_config.get("sn_band", 2),
    )
    opti_data.read_from_tiff(remove_nan=optical_config.get("remove_nan", True))

    filter_report = apply_data_filters(
        opti_data,
        optical_config.get("data_filters", {}),
        out_name=optical_config["outName"],
        base_dir=config_dir,
    )
    opti_data.data_filter_report = filter_report
    filter_report_text = format_filter_report(filter_report)
    if filter_report_text:
        print(filter_report_text)

    opti_data.print_input_summary(
        central_percentile=optical_config["qc"].get("summary_percentile", 99.0)
    )
    return opti_data, optical_config


def process_compute_fault_models(config, lon0, lat0):
    require_processing_dependencies()
    method = config.get("downsample", {}).get("method")
    if method is None:
        return []
    return load_fault_models_for_compute(
        config,
        method,
        lon0,
        lat0,
        TriangularPatches,
        rectangular_cls=RectangularPatches,
        base_dir=config.get("_config_dir"),
    )


def process_plot_fault_overlays(config, lon0, lat0, stage):
    require_processing_dependencies()
    return load_plot_fault_overlays(
        config,
        stage,
        lon0,
        lat0,
        triangular_cls=TriangularPatches,
        rectangular_cls=RectangularPatches,
        base_dir=config.get("_config_dir"),
    )


def sar_plot_values_for_stats(mysar, plot_config):
    values = np.array(mysar.raw_vel, dtype=float, copy=True)
    if str(plot_config.get("value_space", "observation")).replace("-", "_").lower() != "raw":
        spec = mysar.observation_spec if mysar.observation_spec is not None else mysar.build_observation_spec()
        values = mysar.convert_observation_values(values, spec)
    filter_index = getattr(mysar, "data_filter_raw_valid_index", None)
    if filter_index is not None:
        flat = values.reshape(-1)
        keep = np.zeros(flat.size, dtype=bool)
        filter_index = np.asarray(filter_index, dtype=int)
        filter_index = filter_index[(filter_index >= 0) & (filter_index < flat.size)]
        keep[filter_index] = True
        flat[~keep] = np.nan
    return values * float(plot_config.get("factor4plot", 1.0))


def finite_value_diagnostics(values, central_percentile=99.0):
    if not 0.0 < float(central_percentile) <= 100.0:
        raise ValueError("central_percentile must be in the interval (0, 100].")
    array = np.asarray(values, dtype=float)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "central_percentile": float(central_percentile),
            "total_count": int(array.size),
            "valid_count": 0,
            "invalid_count": int(array.size),
            "full_min": np.nan,
            "full_max": np.nan,
            "robust_min": np.nan,
            "robust_max": np.nan,
        }

    tail = (100.0 - float(central_percentile)) / 2.0
    robust_min, robust_max = np.nanpercentile(finite, [tail, 100.0 - tail])
    return {
        "central_percentile": float(central_percentile),
        "total_count": int(array.size),
        "valid_count": int(finite.size),
        "invalid_count": int(array.size - finite.size),
        "full_min": float(np.nanmin(finite)),
        "full_max": float(np.nanmax(finite)),
        "robust_min": float(robust_min),
        "robust_max": float(robust_max),
    }


def format_numeric_range(vmin, vmax):
    return f"[{vmin:.6g}, {vmax:.6g}]"


def clipping_diagnostics(values, vmin, vmax):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0 or vmin is None or vmax is None:
        return None
    below = int(np.count_nonzero(finite < float(vmin)))
    above = int(np.count_nonzero(finite > float(vmax)))
    return {
        "below_count": below,
        "above_count": above,
        "valid_count": int(finite.size),
        "below_percent": float(below / finite.size * 100.0),
        "above_percent": float(above / finite.size * 100.0),
    }


def resolve_plot_limits(plot_config, args, values):
    vmin = args.vmin if args.vmin is not None else plot_config.get("vmin")
    vmax = args.vmax if args.vmax is not None else plot_config.get("vmax")
    if vmin is not None and vmax is not None:
        return vmin, vmax

    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return vmin, vmax
    percentile = float(plot_config.get("auto_percentile", plot_config.get("robust_percentile", 99.0)))
    stats = finite_value_diagnostics(finite, central_percentile=percentile)
    auto_min = stats["robust_min"]
    auto_max = stats["robust_max"]
    if plot_config.get("symmetry", True):
        absmax = max(abs(auto_min), abs(auto_max))
        auto_min, auto_max = -absmax, absmax
    return (auto_min if vmin is None else vmin), (auto_max if vmax is None else vmax)


def write_sar_metadata_file(mysar, sar_config, args, output_file="sar_output.txt"):
    plot_config = deepcopy(sar_config["qc"]["plot"])
    values = sar_plot_values_for_stats(mysar, plot_config)
    vmin, vmax = resolve_plot_limits(plot_config, args, values)
    percentile = sar_config["qc"].get("summary_percentile", 99.0)
    stats = finite_value_diagnostics(values, central_percentile=percentile)
    clipping = clipping_diagnostics(values, vmin, vmax)

    output_lines = [
        f"# mode: {sar_config.get('mode')}",
        f"# reader: {sar_config.get('reader')}",
        f"# plot_value_space: {plot_config.get('value_space')}",
        f"# plot_factor: {plot_config.get('factor4plot')}",
        f"# plot_finite: {stats['valid_count']}/{stats['total_count']}",
        f"# plot_full_range: {format_numeric_range(stats['full_min'], stats['full_max'])}",
        (
            f"# plot_robust_{stats['central_percentile']:g}_range: "
            f"{format_numeric_range(stats['robust_min'], stats['robust_max'])}"
        ),
        f"# plot_vmin: {vmin}",
        f"# plot_vmax: {vmax}",
    ]
    if clipping is not None:
        output_lines.append(
            "# plot_clipped: "
            f"below={clipping['below_count']}/{clipping['valid_count']} "
            f"({clipping['below_percent']:.3g}%), "
            f"above={clipping['above_count']}/{clipping['valid_count']} "
            f"({clipping['above_percent']:.3g}%)"
        )

    prefix = sar_config["files"].get("prefix")
    corner_file = None
    if prefix:
        corner_file = Path(sar_config["directory"]) / f"{prefix}.corner"

    if corner_file is not None and corner_file.exists():
        number = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
        lat_lon_pattern = re.compile(
            rf"latitude \(deg\.\):\s+({number})\s+longitude \(deg\.\):\s+({number})"
        )
        coordinates = []
        with corner_file.open("r", encoding="utf-8") as f:
            for line in f:
                match = lat_lon_pattern.search(line)
                if match:
                    lat, lon = float(match.group(1)), float(match.group(2))
                    coordinates.append((lat, lon))

        if len(coordinates) == 4:
            corner_coords = {
                "Top-left": coordinates[2],
                "Top-right": coordinates[3],
                "Bottom-right": coordinates[1],
                "Bottom-left": coordinates[0],
            }
            # Plane-angle quick-look convention: 0 degree is east and positive
            # is counterclockwise. This is not a north-clockwise heading.
            flight_direction = np.degrees(
                np.arctan2(
                    corner_coords["Top-left"][0] - corner_coords["Bottom-left"][0],
                    corner_coords["Top-left"][1] - corner_coords["Bottom-left"][1],
                )
            )
            output_lines.extend(
                [
                    "# Corner Coordinates:",
                    "# Top-left, Top-right, Bottom-right, Bottom-left",
                    f"  {corner_coords['Top-left'][1]:.3f}, {corner_coords['Top-left'][0]:.3f}",
                    f"  {corner_coords['Top-right'][1]:.3f}, {corner_coords['Top-right'][0]:.3f}",
                    f"  {corner_coords['Bottom-right'][1]:.3f}, {corner_coords['Bottom-right'][0]:.3f}",
                    f"  {corner_coords['Bottom-left'][1]:.3f}, {corner_coords['Bottom-left'][0]:.3f}",
                    "# Flight Direction: "
                    f"{flight_direction:.2f} degree "
                    "(reference=east, direction=counterclockwise)",
                ]
            )

    output_text = "\n".join(output_lines) + "\n"
    print(output_text)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    return vmin, vmax


def optical_plot_config_with_overrides(plot_config, args):
    plot_config = deepcopy(plot_config)
    if args.vmin is not None:
        plot_config["vmin"] = args.vmin
    if args.vmax is not None:
        plot_config["vmax"] = args.vmax
    return plot_config


def write_optical_metadata_file(opti_data, optical_config, args, output_file="optical_output.txt"):
    plot_config = optical_plot_config_with_overrides(optical_config["qc"]["plot"], args)
    percentile = optical_config["qc"].get("summary_percentile", 99.0)
    summary = opti_data.input_summary(central_percentile=percentile)
    factor = plot_config.get("factor4plot", 1.0)
    output_lines = [
        f"# outName: {optical_config.get('outName')}",
        f"# filename: {optical_config.get('filename')}",
        f"# factor_to_m: {optical_config.get('factor_to_m')}",
        f"# plot_factor: {factor}",
        f"# valid_pair: {summary['valid_pair_count']}/{summary['total_count']}",
    ]
    for index, component in enumerate(("east", "north")):
        values = np.asarray(getattr(opti_data, component), dtype=float) * float(factor)
        stats = finite_value_diagnostics(values, central_percentile=percentile)
        vmin = plot_config_value(plot_config.get("vmin"), index)
        vmax = plot_config_value(plot_config.get("vmax"), index)
        clipping = clipping_diagnostics(values, vmin, vmax)
        output_lines.extend(
            [
                f"# {component}_finite: {stats['valid_count']}/{stats['total_count']}",
                f"# {component}_full_range: {format_numeric_range(stats['full_min'], stats['full_max'])}",
                (
                    f"# {component}_robust_{stats['central_percentile']:g}_range: "
                    f"{format_numeric_range(stats['robust_min'], stats['robust_max'])}"
                ),
                f"# {component}_plot_vmin: {vmin}",
                f"# {component}_plot_vmax: {vmax}",
            ]
        )
        if clipping is not None:
            output_lines.append(
                f"# {component}_plot_clipped: "
                f"below={clipping['below_count']}/{clipping['valid_count']} "
                f"({clipping['below_percent']:.3g}%), "
                f"above={clipping['above_count']}/{clipping['valid_count']} "
                f"({clipping['above_percent']:.3g}%)"
            )

    norm_values = (
        np.sqrt(
            np.asarray(opti_data.east, dtype=float) ** 2
            + np.asarray(opti_data.north, dtype=float) ** 2
        )
        * float(factor)
    )
    norm_stats = finite_value_diagnostics(norm_values, central_percentile=percentile)
    output_lines.append(
        f"# horizontal_norm_robust_{norm_stats['central_percentile']:g}_range: "
        f"{format_numeric_range(norm_stats['robust_min'], norm_stats['robust_max'])}"
    )

    output_text = "\n".join(output_lines) + "\n"
    print(output_text)
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(output_text)
    return plot_config.get("vmin"), plot_config.get("vmax")


def plot_sar_quicklook(data, sar_config, selected_faults, args):
    plot_config = deepcopy(sar_config["qc"]["plot"])
    values = sar_plot_values_for_stats(data, plot_config)
    vmin, vmax = resolve_plot_limits(plot_config, args, values)
    plot_config["vmin"] = vmin
    plot_config["vmax"] = vmax

    kwargs = {key: value for key, value in plot_config.items() if key in SAR_PLOT_KWARGS and value is not None}
    kwargs["cmap"] = resolve_cmap(kwargs.get("cmap"))
    kwargs["faults"] = selected_faults
    data.plot_sar_values(**kwargs)


def estimate_covariance(data, config):
    require_processing_dependencies()
    mask_out_entries = covariance_mask_out_entries(config["covar"].get("mask_out"))
    if not mask_out_entries:
        raise ValueError(
            "covar.mask_out is required when estimating covariance. "
            "Mask the deformation-source area before running covariance estimation."
        )
    covar = imcov("Covariance estimator", data, verbose=True)
    for mask_out in mask_out_entries:
        covar.maskOut(mask_out)
    covar.computeCovariance(
        function=config["covar"]["function"],
        frac=config["covar"]["frac"],
        every=config["covar"]["every"],
        distmax=config["covar"]["distmax"],
        rampEst=config["covar"]["rampEst"],
    )
    covar.plot(data="all")
    covar.write2file(savedir="./")
    return covar


def expected_covariance_estimator_files(data_type):
    if data_type == "sar":
        return ["Covariance_estimator.cov"]
    if data_type == "optical":
        return ["Covariance_estimator_East.cov", "Covariance_estimator_North.cov"]
    raise ValueError(f"Unsupported data type: {data_type}")


def read_existing_covariance(data, data_type, missing_policy="existing_or_identity"):
    missing_policy = str(missing_policy or "existing_or_identity").replace("-", "_").lower()
    if missing_policy not in ("existing_or_identity", "identity", "error"):
        raise ValueError(
            "covar.missing_policy must be 'existing_or_identity', 'identity', or 'error'."
        )
    expected_files = expected_covariance_estimator_files(data_type)
    missing_files = [path for path in expected_files if not Path(path).exists()]
    if missing_policy == "identity" or missing_files:
        if missing_policy == "error":
            missing_text = ", ".join(missing_files)
            raise FileNotFoundError(
                "Cannot downsample with covariance file(s) missing: "
                f"{missing_text}. Run covariance estimation first or set "
                "covar.missing_policy: existing_or_identity."
            )
        missing_text = ", ".join(missing_files) if missing_files else "not used by policy"
        warnings.warn(
            "No covariance estimator will be used for this downsampling run "
            f"({missing_text}). The decimated covariance file will be written "
            "as an identity matrix. ",
            UserWarning,
            stacklevel=2,
        )
        return None

    require_processing_dependencies()
    covar = imcov("Covariance estimator", data, verbose=True)
    if data_type == "sar":
        covar.read_from_covfile("Covariance estimator", "Covariance_estimator.cov")
    elif data_type == "optical":
        for suffix in ["East", "North"]:
            covar.read_from_covfile(
                "Covariance estimator" + " " + suffix,
                "Covariance_estimator" + "_" + suffix + ".cov",
            )
    print("read previously calculated covariance estimates")
    return covar


def copy_processing_attributes(target, source, attributes):
    for attribute in attributes:
        if hasattr(source, attribute):
            setattr(target, attribute, getattr(source, attribute))


def build_sar_processing_image(data):
    require_processing_dependencies()
    processing = insar(
        f"{getattr(data, 'name', 'SAR data')} processing",
        utmzone=getattr(data, "utmzone", None),
        lon0=getattr(data, "lon0", None),
        lat0=getattr(data, "lat0", None),
        verbose=False,
    )
    processing.dtype = "insar"
    copy_processing_attributes(
        processing,
        data,
        (
            "x",
            "y",
            "lon",
            "lat",
            "vel",
            "los",
            "factor",
            "heading",
            "incidence",
            "err",
            "utmzone",
            "lon0",
            "lat0",
            "ll2xy",
            "xy2ll",
        ),
    )
    if not hasattr(processing, "factor"):
        processing.factor = 1.0
    return processing


def build_optical_processing_image(data):
    require_processing_dependencies()
    processing = opticorr(
        f"{getattr(data, 'name', 'Optical data')} processing",
        utmzone=getattr(data, "utmzone", None),
        lon0=getattr(data, "lon0", None),
        lat0=getattr(data, "lat0", None),
        verbose=False,
    )
    processing.dtype = "opticorr"
    copy_processing_attributes(
        processing,
        data,
        (
            "x",
            "y",
            "lon",
            "lat",
            "east",
            "north",
            "err_east",
            "err_north",
            "factor",
            "utmzone",
            "lon0",
            "lat0",
            "ll2xy",
            "xy2ll",
        ),
    )
    if not hasattr(processing, "factor"):
        processing.factor = 1.0
    return processing


def build_processing_image(data, data_type):
    if data_type == "sar":
        return build_sar_processing_image(data)
    if data_type == "optical":
        return build_optical_processing_image(data)
    raise ValueError(f"Unsupported data type: {data_type}")


def instantiate_downsampler(sampler_cls, name, data, **kwargs):
    return sampler_cls(name, data, **filter_supported_kwargs(sampler_cls, kwargs))


def configure_downsampler_extraction(downsampler, downsample_config):
    extraction = downsample_config.get("extraction")
    if not extraction:
        return
    if hasattr(downsampler, "setExtractionConfig"):
        downsampler.setExtractionConfig(extraction)
    elif hasattr(downsampler, "set_extraction_config"):
        downsampler.set_extraction_config(extraction)
    else:
        downsampler.extraction_config = extraction


def estimated_point_spacing(x, y):
    from scipy.spatial import KDTree

    points = np.column_stack((np.asarray(x, dtype=float), np.asarray(y, dtype=float)))
    finite = np.all(np.isfinite(points), axis=1)
    points = points[finite]
    if points.shape[0] < 2:
        raise ValueError("Cannot estimate guide-grid pixel spacing from fewer than two finite points.")
    tree = KDTree(points)
    distances, _ = tree.query(points, k=2)
    nearest = distances[:, 1]
    nearest = nearest[np.isfinite(nearest) & (nearest > 0.0)]
    if nearest.size == 0:
        raise ValueError("Cannot estimate guide-grid pixel spacing from duplicated or invalid coordinates.")
    return float(np.nanmedian(nearest))


def regular_grid_index(lon, lat, values_size, max_grid_factor=2.0):
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    if lon.shape[0] != values_size or lat.shape[0] != values_size:
        return None
    finite = np.isfinite(lon) & np.isfinite(lat)
    if not np.any(finite):
        return None

    unique_lon, lon_inverse = np.unique(lon[finite], return_inverse=True)
    unique_lat, lat_inverse = np.unique(lat[finite], return_inverse=True)
    grid_size = unique_lon.size * unique_lat.size
    if grid_size < values_size or grid_size > max(values_size * max_grid_factor, values_size + 1):
        return None

    row = np.full(values_size, -1, dtype=int)
    col = np.full(values_size, -1, dtype=int)
    finite_indices = np.flatnonzero(finite)
    row[finite_indices] = lat_inverse
    col[finite_indices] = lon_inverse

    flat_index = row[finite_indices] * unique_lon.size + col[finite_indices]
    if np.unique(flat_index).size != flat_index.size:
        return None
    return unique_lat.size, unique_lon.size, row, col


def grid_spacing_from_coordinates(x_grid, y_grid, axis):
    dx = np.diff(x_grid, axis=axis)
    dy = np.diff(y_grid, axis=axis)
    distances = np.sqrt(dx**2 + dy**2)
    distances = distances[np.isfinite(distances) & (distances > 0.0)]
    if distances.size == 0:
        return None
    return float(np.nanmedian(distances))


def gaussian_smooth_regular_grid(x, y, values, sigma, radius_sigma, lon=None, lat=None, return_info=False):
    from scipy.ndimage import gaussian_filter

    if lon is None or lat is None:
        return None

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    grid_index = regular_grid_index(lon, lat, values.size)
    if grid_index is None:
        return None

    nlat, nlon, row, col = grid_index
    valid_position = (row >= 0) & (col >= 0)
    value_grid = np.full((nlat, nlon), np.nan, dtype=float)
    x_grid = np.full((nlat, nlon), np.nan, dtype=float)
    y_grid = np.full((nlat, nlon), np.nan, dtype=float)
    value_grid[row[valid_position], col[valid_position]] = values[valid_position]
    x_grid[row[valid_position], col[valid_position]] = x[valid_position]
    y_grid[row[valid_position], col[valid_position]] = y[valid_position]

    x_spacing = grid_spacing_from_coordinates(x_grid, y_grid, axis=1)
    y_spacing = grid_spacing_from_coordinates(x_grid, y_grid, axis=0)
    if x_spacing is None or y_spacing is None:
        return None

    sigma_pixels = (float(sigma) / y_spacing, float(sigma) / x_spacing)
    finite_values = np.isfinite(value_grid)
    if not np.any(finite_values):
        raise ValueError("Cannot build guide grid from data with no finite observation values.")
    weighted_values = gaussian_filter(
        np.where(finite_values, value_grid, 0.0),
        sigma=sigma_pixels,
        mode="nearest",
        truncate=float(radius_sigma),
    )
    weights = gaussian_filter(
        finite_values.astype(float),
        sigma=sigma_pixels,
        mode="nearest",
        truncate=float(radius_sigma),
    )
    smoothed_grid = np.full_like(value_grid, np.nan)
    valid_weight = weights > 0.0
    smoothed_grid[valid_weight] = weighted_values[valid_weight] / weights[valid_weight]

    output = np.full(values.shape, np.nan, dtype=float)
    valid_output = valid_position & np.isfinite(values)
    output[valid_output] = smoothed_grid[row[valid_output], col[valid_output]]
    if not return_info:
        return output
    return output, {
        "backend": "regular_grid",
        "grid_shape": [int(nlat), int(nlon)],
        "x_spacing": float(x_spacing),
        "y_spacing": float(y_spacing),
        "sigma_pixels_x": float(sigma / x_spacing),
        "sigma_pixels_y": float(sigma / y_spacing),
    }


def gaussian_smooth_scattered_values(x, y, values, sigma, radius_sigma=3.0, return_info=False):
    from scipy.spatial import KDTree

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    values = np.asarray(values, dtype=float)
    sigma = float(sigma)
    radius = float(radius_sigma) * sigma
    if sigma <= 0.0 or radius <= 0.0:
        raise ValueError("guide_grid.filter.sigma and radius_sigma must be positive.")

    points = np.column_stack((x, y))
    finite_xy = np.all(np.isfinite(points), axis=1)
    finite_values = np.isfinite(values)
    valid = finite_xy & finite_values
    tree_points = points[valid]
    if tree_points.size == 0:
        raise ValueError("Cannot build guide grid from data with no finite coordinates and values.")
    tree = KDTree(tree_points)
    original_indices = np.flatnonzero(valid)
    output = np.full(values.shape, np.nan, dtype=float)

    for original_index in original_indices:
        neighbor_positions = tree.query_ball_point(points[original_index], r=radius)
        if not neighbor_positions:
            continue
        neighbor_indices = original_indices[np.asarray(neighbor_positions, dtype=int)]
        neighbor_indices = neighbor_indices[finite_values[neighbor_indices]]
        if neighbor_indices.size == 0:
            continue
        distances2 = (
            (x[neighbor_indices] - x[original_index]) ** 2
            + (y[neighbor_indices] - y[original_index]) ** 2
        )
        weights = np.exp(-0.5 * distances2 / (sigma ** 2))
        output[original_index] = float(np.sum(weights * values[neighbor_indices]) / np.sum(weights))
    if not return_info:
        return output
    return output, {
        "backend": "scattered",
        "valid_points": int(original_indices.size),
        "radius": float(radius),
    }


def gaussian_smooth_point_values(x, y, values, sigma, radius_sigma=3.0, lon=None, lat=None, return_info=False):
    sigma = float(sigma)
    radius_sigma = float(radius_sigma)
    if sigma <= 0.0 or radius_sigma <= 0.0:
        raise ValueError("guide_grid.filter.sigma and radius_sigma must be positive.")

    grid_result = gaussian_smooth_regular_grid(
        x,
        y,
        values,
        sigma,
        radius_sigma,
        lon=lon,
        lat=lat,
        return_info=return_info,
    )
    if grid_result is not None:
        return grid_result
    return gaussian_smooth_scattered_values(x, y, values, sigma, radius_sigma, return_info=return_info)


def guide_grid_sigma_in_xy_units(data, guide_grid):
    guide_filter = guide_grid.get("filter", {}) or {}
    sigma = guide_filter.get("sigma")
    unit = str(guide_filter.get("unit", "km")).replace("-", "_").lower()
    if sigma is None:
        raise ValueError("downsample.guide_grid.filter.sigma is required when guide_grid.enabled is true.")
    sigma = float(sigma)
    if unit == "pixel":
        sigma *= estimated_point_spacing(data.x, data.y)
    elif unit != "km":
        raise ValueError("downsample.guide_grid.filter.unit must be 'km' or 'pixel'.")
    return sigma


def build_guide_grid_image(data, data_type, guide_grid):
    guide = copy(data)
    guide.name = f"{getattr(data, 'name', 'data')} guide-grid"
    sigma = guide_grid_sigma_in_xy_units(data, guide_grid)
    radius_sigma = float((guide_grid.get("filter", {}) or {}).get("radius_sigma", 3.0))
    component = str(guide_grid.get("component", "auto")).replace("-", "_").lower()
    component_reports = {}

    def smooth_component(name, values):
        smoothed, info = gaussian_smooth_point_values(
            data.x,
            data.y,
            values,
            sigma,
            radius_sigma,
            lon=getattr(data, "lon", None),
            lat=getattr(data, "lat", None),
            return_info=True,
        )
        component_reports[name] = info
        return smoothed

    if data_type == "sar":
        if component not in ("auto", "observation"):
            raise ValueError("SAR guide_grid.component must be 'auto' or 'observation'.")
        guide.vel = smooth_component("observation", data.vel)
    elif data_type == "optical":
        east = np.asarray(data.east, dtype=float)
        north = np.asarray(data.north, dtype=float)
        if component == "auto":
            component = "magnitude"
        if component == "both":
            guide.east = smooth_component("east", east)
            guide.north = smooth_component("north", north)
        elif component == "east":
            guide.east = smooth_component("east", east)
            guide.north = np.zeros_like(guide.east)
        elif component == "north":
            guide.north = smooth_component("north", north)
            guide.east = np.zeros_like(guide.north)
        elif component == "magnitude":
            magnitude = np.sqrt(east ** 2 + north ** 2)
            guide.east = smooth_component("magnitude", magnitude)
            guide.north = np.zeros_like(guide.east)
        else:
            raise ValueError("Optical guide_grid.component must be auto, magnitude, east, north, or both.")
    else:
        raise ValueError(f"Unsupported data type for guide grid: {data_type!r}.")
    guide.guide_grid_source = guide_grid
    first_report = next(iter(component_reports.values()), {})
    guide.guide_grid_report = {
        "enabled": True,
        "source": guide_grid.get("source", "filtered_observation"),
        "component": component,
        "filter": guide_grid.get("filter", {}),
        "backend": first_report.get("backend"),
        "components": component_reports,
    }
    return guide


def replace_downsampler_image(downsampler, data):
    from scipy.spatial import KDTree

    downsampler.image = data
    downsampler.datatype = data.dtype
    downsampler.PIXXY = np.vstack((data.x, data.y)).T
    downsampler.PIXXY_Tree = KDTree(downsampler.PIXXY)


def guide_grid_enabled(downsample_config, method):
    guide_grid = downsample_config.get("guide_grid", {}) or {}
    return bool(guide_grid.get("enabled", False)) and method in ("std", "data")


def initialize_downsampler_report_context(downsampler):
    downsampler.guide_grid_report = {"enabled": False}


def read_polygon_file(path):
    try:
        points = pd.read_csv(
            path,
            sep=r"[\s,]+",
            comment="#",
            header=None,
            engine="python",
        )
    except FileNotFoundError:
        raise
    except Exception as exc:
        raise ValueError(f"Could not read focus-region polygon file {path!s}: {exc}") from exc

    if points.shape[1] < 2:
        raise ValueError(f"Focus-region polygon file {path!s} must contain at least two columns.")
    polygon = points.iloc[:, :2].to_numpy(dtype=float)
    if polygon.shape[0] < 3:
        raise ValueError(f"Focus-region polygon file {path!s} must contain at least three points.")
    return polygon


def resolve_std_focus_polygon(downsampler, std_config, config_dir=None):
    focus_region = std_config.get("focus_region") or {}
    if not focus_region.get("enabled", False):
        return None

    if focus_region.get("polygon_file") is not None:
        polygon_path = resolve_config_relative_path(focus_region["polygon_file"], config_dir=config_dir)
        polygon = read_polygon_file(polygon_path)
        focus_region["resolved_polygon_file"] = str(polygon_path)
    else:
        polygon = np.asarray(focus_region["polygon"], dtype=float)

    coord_type = str(focus_region.get("coord_type", "lonlat")).replace("-", "_").lower()
    if coord_type == "lonlat":
        x, y = downsampler.ll2xy(polygon[:, 0], polygon[:, 1])
        polygon = np.column_stack((x, y))
    return polygon


def std_based_kwargs(downsampler, std_config, config_dir=None):
    kwargs = {
        "smooth": std_config.get("split_metric_smoothing"),
        "itmax": std_config.get("itmax", 100),
        "use_variance": std_config.get("use_variance", False),
        "amplitude_stat": std_config.get("amplitude_stat", "mean_abs"),
        "correction": std_config.get("split_metric_correction", "std"),
    }

    focus_region = std_config.get("focus_region") or {}
    if focus_region.get("enabled", False):
        kwargs["mask_polygon"] = resolve_std_focus_polygon(downsampler, std_config, config_dir=config_dir)
        kwargs["max_splits_mask_out"] = focus_region.get("max_splits_outside", 5)

    high_value_refinement = std_config.get("high_value_refinement") or {}
    if high_value_refinement.get("enabled", False):
        kwargs["high_value_ratio"] = high_value_refinement.get("high_value_ratio", 0.25)
        kwargs["min_splits_high_value"] = high_value_refinement.get("min_splits", 1)
        kwargs["reference_max_value"] = high_value_refinement.get("reference_max_value")

    low_amplitude_cap = std_config.get("low_amplitude_cap") or {}
    if low_amplitude_cap.get("enabled", False):
        kwargs["low_amplitude_ratio"] = low_amplitude_cap.get("amplitude_ratio", 0.05)
        kwargs["max_splits_low_amplitude"] = low_amplitude_cap.get("max_splits", 3)
        kwargs["low_amplitude_apply_inside_mask"] = low_amplitude_cap.get("apply_inside_focus_region", False)
        low_reference = low_amplitude_cap.get("reference_max_value")
        high_reference = kwargs.get("reference_max_value")
        if (
            low_reference is not None
            and high_reference is not None
            and not np.isclose(float(low_reference), float(high_reference))
        ):
            raise ValueError(
                "std_config.high_value_refinement.reference_max_value and "
                "std_config.low_amplitude_cap.reference_max_value must match, "
                "because CSI stdBased uses one shared reference_max_value."
            )
        if low_reference is not None:
            kwargs["reference_max_value"] = low_reference

    return compact_kwargs(kwargs)


def run_downsampling(data, data_type, config, selected_faults, out_name):
    require_processing_dependencies()
    downsample_config = config["downsample"]
    method = downsample_config["method"]
    sampler_cls = imagedownsampling
    rsp_template = None

    if method == "from_rsp":
        from_rsp_config = downsample_config["from_rsp_config"]
        rsp_template = read_rsp_grid_template(
            from_rsp_config["rsp_file"],
            geometry=from_rsp_config.get("geometry", "auto"),
        )
        from_rsp_config["resolved_geometry"] = rsp_template.geometry

    if method == "trirb":
        if not selected_faults:
            raise ValueError(
                "downsample.method='trirb' requires at least one enabled triangular fault_model in "
                "the config because CSI triangular resolution-based downsampling "
                "builds a fault Laplacian. Enable a fault model with use_for: [trirb], or use "
                "downsample.method='std' for fault-free first tests."
            )

    uses_triangular_sampler = method == "trirb" or (
        rsp_template is not None and rsp_template.geometry == "triangle"
    )
    if uses_triangular_sampler:
        from csi.imagedownsamplingTriangular import (
            imagedownsamplingTriangular as sampler_cls,
        )

    workers = config.get("_workers")

    if method == "std":
        downsampler = instantiate_downsampler(
            sampler_cls,
            "Downsampler",
            data,
            faults=selected_faults,
            workers=workers,
        )
        initialize_downsampler_report_context(downsampler)
        configure_downsampler_extraction(downsampler, downsample_config)
        guide_data = None
        if guide_grid_enabled(downsample_config, method):
            guide_data = build_guide_grid_image(data, data_type, downsample_config["guide_grid"])
            downsampler.guide_grid_report = getattr(guide_data, "guide_grid_report", {"enabled": True})
            replace_downsampler_image(downsampler, guide_data)
            print("Guide-grid enabled: using filtered guide image for std grid construction.")
        std_config = downsample_config["std_config"]
        plot = std_config.get("plot", False)
        decimorig = std_config.get("decimorig", 10)
        guide_plot = False if guide_data is not None else plot
        downsampler.initialstate(
            startingsize=std_config["startingsize"],
            minimumsize=std_config["minimumsize"],
            tolerance=std_config["min_valid_fraction"],
            plot=guide_plot,
            decimorig=decimorig,
        )
        std_kwargs = std_based_kwargs(downsampler, std_config, config_dir=config.get("_config_dir"))
        downsampler.stdBased(
            std_config["split_std_threshold"],
            plot=guide_plot,
            verboseLevel=std_config.get("verboseLevel", "minimum"),
            decimorig=decimorig,
            **std_kwargs,
        )
        if guide_data is not None:
            replace_downsampler_image(downsampler, data)
            downsampler.downsample(plot=plot, decimorig=decimorig)
    elif method == "data":
        downsampler = instantiate_downsampler(
            sampler_cls,
            "Downsampler",
            data,
            faults=selected_faults,
            workers=workers,
        )
        initialize_downsampler_report_context(downsampler)
        configure_downsampler_extraction(downsampler, downsample_config)
        guide_data = None
        if guide_grid_enabled(downsample_config, method):
            guide_data = build_guide_grid_image(data, data_type, downsample_config["guide_grid"])
            downsampler.guide_grid_report = getattr(guide_data, "guide_grid_report", {"enabled": True})
            replace_downsampler_image(downsampler, guide_data)
            print("Guide-grid enabled: using filtered guide image for data grid construction.")
        data_config = downsample_config["data_config"]
        plot = data_config.get("plot", False)
        decimorig = data_config.get("decimorig", 10)
        guide_plot = False if guide_data is not None else plot
        downsampler.initialstate(
            startingsize=data_config["startingsize"],
            minimumsize=data_config["minimumsize"],
            tolerance=data_config["min_valid_fraction"],
            plot=guide_plot,
            decimorig=decimorig,
        )
        downsampler.dataBased(
            data_config["split_metric_threshold"],
            plot=guide_plot,
            verboseLevel=data_config.get("verboseLevel", "minimum"),
            decimorig=decimorig,
            quantity=data_config.get("split_metric", "curvature"),
            smooth=data_config.get("split_metric_smoothing"),
            itmax=data_config.get("itmax", 100),
        )
        if guide_data is not None:
            replace_downsampler_image(downsampler, data)
            downsampler.downsample(plot=plot, decimorig=decimorig)
    elif method == "trirb":
        if data_type == "optical":
            vel_type = config["optical_config"].get("vel_type", "north")
            downsampler = instantiate_downsampler(
                sampler_cls,
                "Downsampler",
                data,
                faults=selected_faults,
                vel_type=vel_type,
                workers=workers,
            )
        else:
            downsampler = instantiate_downsampler(
                sampler_cls,
                "Downsampler",
                data,
                faults=selected_faults,
                workers=workers,
            )

        initialize_downsampler_report_context(downsampler)
        configure_downsampler_extraction(downsampler, downsample_config)
        trirb_config = downsample_config["trirb_config"]
        plot = trirb_config.get("plot", False)
        decimorig = trirb_config.get("decimorig", 10)
        downsampler.initialstate(
            minimumsize=trirb_config["minimumsize"],
            tolerance=trirb_config["min_valid_fraction"],
            plot=plot,
            decimorig=decimorig,
        )
        downsampler.resolutionBased(
            max_samples=trirb_config["max_samples"],
            change_threshold=trirb_config["change_threshold"],
            smooth_factor=trirb_config["smooth_factor"],
            slipdirection=trirb_config.get("slipdirection", "sd"),
            plot=plot,
            verboseLevel=trirb_config.get("verboseLevel", "maximum"),
            decimorig=decimorig,
            vertical=trirb_config.get("vertical", False),
        )
    elif method == "from_rsp":
        if data_type == "optical" and uses_triangular_sampler:
            vel_type = config["optical_config"].get("vel_type", "north")
            downsampler = instantiate_downsampler(
                sampler_cls,
                "Downsampler",
                data,
                faults=selected_faults,
                vel_type=vel_type,
                workers=workers,
            )
        else:
            downsampler = instantiate_downsampler(
                sampler_cls,
                "Downsampler",
                data,
                faults=selected_faults,
                workers=workers,
            )

        initialize_downsampler_report_context(downsampler)
        configure_downsampler_extraction(downsampler, downsample_config)
        from_rsp_config = downsample_config["from_rsp_config"]
        plot = from_rsp_config.get("plot", False)
        decimorig = from_rsp_config.get("decimorig", 10)
        apply_rsp_grid_template(
            downsampler,
            rsp_template,
            tolerance=from_rsp_config.get("min_valid_fraction", 0.0),
        )
        print(
            "Reusing downsampling grid from "
            f"{rsp_template.path} "
            f"(geometry={rsp_template.geometry}, cells={rsp_template.cell_count})"
        )
        downsampler.downsample(plot=plot, decimorig=decimorig)
    else:
        raise ValueError(f"Unsupported downsample method: {method!r}.")

    downsampler.writeDownsampled2File(prefix=out_name + "_ifg", rsp=True)
    return downsampler


def box_values(box, *, label):
    if box is None:
        return None
    if isinstance(box, dict):
        key_groups = (
            ("minLon", "maxLon", "minLat", "maxLat"),
            ("minlon", "maxlon", "minlat", "maxlat"),
        )
        for keys in key_groups:
            if all(key in box for key in keys):
                values = [box[key] for key in keys]
                break
        else:
            raise ValueError(
                f"{label} must contain minLon/maxLon/minLat/maxLat "
                "or minlon/maxlon/minlat/maxlat."
            )
    else:
        values = list(box)
    if len(values) != 4:
        raise ValueError(f"{label} must contain four values.")
    if any(value is None for value in values):
        return None
    return [float(value) for value in values]


def covariance_mask_out_entries(mask_out):
    if mask_out is None:
        return []
    if isinstance(mask_out, dict):
        values = box_values(mask_out, label="covar.mask_out")
        return [] if values is None else [values]
    if len(mask_out) == 4 and not isinstance(mask_out[0], (dict, list, tuple)):
        values = box_values(mask_out, label="covar.mask_out")
        return [] if values is None else [values]

    entries = []
    for index, entry in enumerate(mask_out):
        values = box_values(entry, label=f"covar.mask_out[{index}]")
        if values is not None:
            entries.append(values)
    return entries


def require_covariance_mask_for_estimation(config):
    if not covariance_mask_out_entries(config["covar"].get("mask_out")):
        raise ValueError(
            "covar.mask_out is required when covar.do_covar is true or "
            "ecat-downsample -c is used. Mask the deformation-source area "
            "before estimating the empirical covariance."
        )


def downsample_uses_triangular_cells(config):
    method = config["downsample"]["method"]
    if method == "trirb":
        return True
    if method != "from_rsp":
        return False
    from_rsp_config = config["downsample"].get("from_rsp_config", {})
    geometry = from_rsp_config.get("resolved_geometry", from_rsp_config.get("geometry"))
    return str(geometry or "").replace("-", "_").lower() == "triangle"


def read_sar_decimated_dataset(config, out_name):
    require_processing_dependencies()
    sardecim = insar("Decimated DataSet", lon0=config["general"]["lon0"], lat0=config["general"]["lat0"])
    triangular = downsample_uses_triangular_cells(config)
    sardecim.read_from_varres(out_name + "_ifg", factor=1.0, step=0.0, triangular=triangular)
    sardecim.checkLosEqualsOne()
    return sardecim


def read_optical_decimated_dataset(config, out_name):
    require_processing_dependencies()
    optdecim = opticorr("Decimated DataSet", lon0=config["general"]["lon0"], lat0=config["general"]["lat0"])
    triangular = downsample_uses_triangular_cells(config)
    optdecim.read_from_varres(out_name + "_ifg", factor=1.0, step=0.0, triangular=triangular)
    return optdecim


def decimated_covariance_size(decimated_data, data_type):
    if data_type == "sar":
        return int(np.asarray(decimated_data.vel).size)
    if data_type == "optical":
        return int(np.asarray(decimated_data.east).size + np.asarray(decimated_data.north).size)
    raise ValueError(f"Unsupported data type: {data_type}")


def write_decimated_covariance(decimated_data, covar, data_type, out_name):
    cov_file = out_name + "_ifg.cov"
    if covar is None:
        nd = decimated_covariance_size(decimated_data, data_type)
        decimated_data.Cd = np.eye(nd, dtype=np.float32)
        decimated_data.Cd.tofile(cov_file)
        print(f"Decimated covariance written as identity matrix: {cov_file}")
        return "identity"

    decimated_data.Cd = covar.buildCovarianceMatrix(
        decimated_data,
        "Covariance estimator",
        write2file=cov_file,
    )
    return "estimated"


def plot_config_value(config_value, index=None):
    if index is None or config_value is None:
        return config_value
    if isinstance(config_value, (list, tuple)):
        return config_value[index]
    return config_value


def display_unit(factor):
    try:
        factor = float(factor)
    except (TypeError, ValueError):
        return ""
    if np.isclose(factor, 100.0):
        return "cm"
    if np.isclose(factor, 1.0):
        return "m"
    return f"x{factor:g}"


def sar_decimated_label(sar_config, factor4plot=1.0):
    unit = display_unit(factor4plot)
    unit = f" ({unit})" if unit else ""
    mode = str(sar_config.get("mode") or "").replace("-", "_").lower()
    if mode in ("az", "azimuth", "azimuth_offset"):
        return f"Azimuth offset{unit}"
    if mode in ("range", "range_offset"):
        return f"Range/LOS observation{unit}"
    return f"LOS disp.{unit}"


def optical_decimated_label(component, factor4plot=1.0):
    unit = display_unit(factor4plot)
    unit = f" ({unit})" if unit else ""
    return f"{component.capitalize()}ward disp.{unit}"


def resolved_decim_factor(plot_decim, raw_plot):
    factor = plot_decim.get("factor4plot", raw_plot.get("factor4plot", 1.0))
    return raw_plot.get("factor4plot", 1.0) if factor is None else factor


def decimated_plot_kwargs(config, raw_plot=None, args=None, component_index=None, cb_label=None):
    plot_decim = config["downsample"].get("plot_decim", {}) or {}
    raw_plot = raw_plot or {}
    factor4plot = resolved_decim_factor(plot_decim, raw_plot)

    vmin = plot_decim.get("vmin", raw_plot.get("vmin"))
    vmax = plot_decim.get("vmax", raw_plot.get("vmax"))
    if args is not None:
        vmin = args.vmin if args.vmin is not None else vmin
        vmax = args.vmax if args.vmax is not None else vmax
    vmin = plot_config_value(vmin, component_index)
    vmax = plot_config_value(vmax, component_index)

    label = plot_decim.get("cb_label", plot_decim.get("cblabel"))
    if label is None:
        label = cb_label

    return {
        "style": plot_decim.get("style", "cells"),
        "coordrange": plot_decim.get("coordrange"),
        "factor4plot": factor4plot,
        "vmin": vmin,
        "vmax": vmax,
        "symmetry": plot_decim.get("symmetry", raw_plot.get("symmetry", True)),
        "cmap": resolve_cmap(plot_decim.get("cmap", raw_plot.get("cmap", DEFAULT_CMAP))),
        "figsize": plot_decim.get("figsize", [3.0, 5.0]),
        "dpi": plot_decim.get("dpi", 300),
        "edgewidth": plot_decim.get("edgewidth", 0.1),
        "edgecolor": plot_decim.get("edgecolor", "black"),
        "alpha": plot_decim.get("alpha", 1.0),
        "markersize": plot_decim.get("markersize", 10),
        "colorbar_orientation": plot_decim.get(
            "colorbar_orientation",
            plot_decim.get("cborientation", "vertical"),
        ),
        "colorbar_mode": plot_decim.get("colorbar_mode", "auto"),
        "colorbar_loc": plot_decim.get("colorbar_loc"),
        "colorbar_size": plot_decim.get("colorbar_size"),
        "colorbar_thickness": plot_decim.get("colorbar_thickness"),
        "colorbar_pad": plot_decim.get("colorbar_pad"),
        "colorbar_x": plot_decim.get("colorbar_x"),
        "colorbar_y": plot_decim.get("colorbar_y"),
        "colorbar_length": plot_decim.get("colorbar_length"),
        "colorbar_height": plot_decim.get("colorbar_height"),
        "cb_label": label,
        "cb_label_loc": plot_decim.get("cb_label_loc"),
        "tickfontsize": plot_decim.get("tickfontsize", 10),
        "labelfontsize": plot_decim.get("labelfontsize", 10),
        "style_context": plot_decim.get("style_context", ["science"]),
        "fontsize": plot_decim.get("fontsize"),
    }


def plot_sar_downsample_check(sardecim, config, selected_faults, out_name, args):
    require_processing_dependencies()
    output_check = config["sar_config"].get("output_check", True)
    if not output_check:
        return

    raw_plot = config["sar_config"]["qc"]["plot"]
    plot_decim = config["downsample"].get("plot_decim", {}) or {}
    factor4plot = resolved_decim_factor(plot_decim, raw_plot)
    plot_kwargs = decimated_plot_kwargs(
        config,
        raw_plot=raw_plot,
        args=args,
        cb_label=sar_decimated_label(config["sar_config"], factor4plot=factor4plot),
    )
    plot_decimated_geodata(
        sardecim.lon,
        sardecim.lat,
        sardecim.vel,
        corners=getattr(sardecim, "corner", None),
        faults=selected_faults,
        trace_color=plot_decim.get("trace_color", "black"),
        trace_linewidth=plot_decim.get("trace_linewidth", 0.5),
        show=False,
        savefig=out_name + "_decim.png",
        **plot_kwargs,
    )


def plot_optical_downsample_check(optdecim, config, selected_faults, out_name, args):
    require_processing_dependencies()
    output_check = config["optical_config"].get("output_check", True)
    if not output_check:
        return

    raw_plot = config["optical_config"]["qc"]["plot"]
    plot_decim = config["downsample"].get("plot_decim", {}) or {}
    factor4plot = resolved_decim_factor(plot_decim, raw_plot)
    for index, component, values, suffix in (
        (0, "east", optdecim.east, "_East_decim.png"),
        (1, "north", optdecim.north, "_North_decim.png"),
    ):
        plot_kwargs = decimated_plot_kwargs(
            config,
            raw_plot=raw_plot,
            args=args,
            component_index=index,
            cb_label=optical_decimated_label(component, factor4plot=factor4plot),
        )
        plot_decimated_geodata(
            optdecim.lon,
            optdecim.lat,
            values,
            corners=getattr(optdecim, "corner", None),
            faults=selected_faults,
            trace_color=plot_decim.get("trace_color", "black"),
            trace_linewidth=plot_decim.get("trace_linewidth", 0.5),
            show=False,
            savefig=out_name + suffix,
            **plot_kwargs,
        )


def plot_optical_quicklook(data, config, selected_faults, out_name, args):
    raw_plot = optical_plot_config_with_overrides(
        config["optical_config"]["qc"]["plot"],
        args,
    )
    if raw_plot.get("file_path") is None:
        raw_plot["file_path"] = out_name + "_deformation_map.jpg"
    kwargs = {key: value for key, value in raw_plot.items() if key in OPTICAL_PLOT_KWARGS and value is not None}
    kwargs["cmap"] = resolve_cmap(kwargs.get("cmap", DEFAULT_CMAP))
    kwargs["faults"] = selected_faults
    data.plot_optical_values(**kwargs)


def resolve_run_steps(args, config):
    do_covar = args.do_covar or config["covar"].get("do_covar", False)
    do_downsample = args.do_downsample or config["downsample"].get("enabled", False)
    show_raw_data = args.show_raw_data

    if show_raw_data:
        do_covar = False
        do_downsample = False

    return {
        "do_covar": bool(do_covar),
        "do_downsample": bool(do_downsample),
        "show_raw_data": bool(show_raw_data),
    }


def expected_outputs_for_run(config, out_name, steps):
    data_type = config["data_type"]
    outputs = []

    if steps["show_raw_data"]:
        if data_type == "sar":
            outputs.append("sar_output.txt")
            plot_config = config["sar_config"]["qc"]["plot"]
            if plot_config.get("save_fig", False):
                outputs.append(plot_config.get("file_path", "sar_values.png"))
        elif data_type == "optical":
            outputs.append("optical_output.txt")
            raw_plot = config["optical_config"]["qc"]["plot"]
            if raw_plot.get("save_fig", True):
                outputs.append(raw_plot.get("file_path") or out_name + "_deformation_map.jpg")

    if steps["do_covar"]:
        if data_type == "sar":
            outputs.append("Covariance_estimator.cov")
        elif data_type == "optical":
            outputs.extend(["Covariance_estimator_East.cov", "Covariance_estimator_North.cov"])

    if steps["do_downsample"]:
        outputs.extend([out_name + "_ifg.txt", out_name + "_ifg.rsp", out_name + "_ifg.cov"])
        report_config = config["downsample"].get("report", {})
        report_path = downsample_report_file(report_config, out_name)
        if report_config.get("enabled", True) and report_path:
            outputs.append(report_path)
        if data_type == "sar" and config["sar_config"].get("output_check", True):
            outputs.append(out_name + "_decim.png")
        if data_type == "optical" and config["optical_config"].get("output_check", True):
            outputs.extend([out_name + "_East_decim.png", out_name + "_North_decim.png"])

    if data_type in ("sar", "optical"):
        data_config = config["sar_config"] if data_type == "sar" else config["optical_config"]
        data_filters = data_config.get("data_filters", {})
        report_path = filter_report_file(data_filters, out_name)
        if data_filters.get("enabled", False) and data_filters.get("report", True) and report_path:
            outputs.append(report_path)

    processing_region = config.get("processing_region", {})
    region_report_path = processing_region_report_file(processing_region, out_name)
    if (
        (steps["do_covar"] or steps["do_downsample"])
        and processing_region.get("enabled", False)
        and processing_region.get("report", True)
        and region_report_path
    ):
        outputs.append(region_report_path)

    return outputs


def write_run_metadata(config, args, out_name, steps, outputs, metadata_file=None):
    metadata_file = metadata_file or f"{out_name}_run_metadata.yml"
    general = config["general"]
    compute_config = config.get("downsample", {}).get("compute", {}) or {}
    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_file": args.config,
        "data_type": config["data_type"],
        "out_name": out_name,
        "projection": {
            "origin": general.get("origin"),
            "origin_resolved_from": general.get("origin_resolved_from"),
            "lon0": general.get("lon0"),
            "lat0": general.get("lat0"),
            "utmzone": general.get("utmzone"),
            "epsg": general.get("epsg"),
            "note": "Used for local x/y coordinates during downsampling.",
        },
        "steps": steps,
        "compute": {
            "downsample_cutde": cutde_backend_summary(
                compute_config.get("cutde_backend")
            ),
        },
        "command_overrides": {
            "vmin": args.vmin,
            "vmax": args.vmax,
            "workers": getattr(args, "workers", None),
        },
        "expected_outputs": outputs,
        "effective_config": config,
    }
    with open(metadata_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(metadata, f, allow_unicode=True, sort_keys=False)
    print(f"Run metadata written to: {metadata_file}")
    return metadata_file


def configure_downsample_compute_backend(config, steps=None):
    compute_config = config.get("downsample", {}).get("compute", {}) or {}
    requested = compute_config.get("cutde_backend", "cpp")
    state = configure_cutde_backend(requested, default="cpp")
    config.setdefault("_runtime", {})["downsample_cutde_backend"] = state

    method = config.get("downsample", {}).get("method")
    if steps is None or steps.get("do_downsample", False):
        if method == "trirb":
            active = state.get("active_backend") or "not loaded yet"
            print(
                "Downsampling compute backend: "
                f"requested cutde_backend={state['requested_backend']}, "
                f"active={active}"
            )
    return state


def prepare_run(args):
    config = normalize_downsample_config(load_config(args.config))
    config["_config_dir"] = str(Path(args.config).resolve().parent)
    if getattr(args, "workers", None) is not None and args.workers < 1:
        raise ValueError("--workers must be a positive integer.")
    config["_workers"] = getattr(args, "workers", None)
    steps = resolve_run_steps(args, config)
    configure_downsample_compute_backend(config, steps=steps)
    if steps["do_covar"]:
        require_covariance_mask_for_estimation(config)

    print(
        "do_covar: {do_covar}, do_downsample: {do_downsample}, "
        "show_raw_data: {show_raw_data}".format(**steps)
    )

    return config, steps


def load_input_data(config, args, steps):
    lon0, lat0 = resolve_projection_origin(config)
    data_type = config["data_type"]
    if data_type == "sar":
        print("Processing SAR data...")
        data, normalized_sar_config = process_sar_data(
            config["sar_config"],
            lon0,
            lat0,
            do_covar=steps["do_covar"],
            config_dir=config.get("_config_dir"),
        )
        config["sar_config"] = normalized_sar_config
        out_name = get_sar_output_name(normalized_sar_config)
    elif data_type == "optical":
        print("Processing optical data...")
        data, normalized_optical_config = process_optical_data(
            config["optical_config"],
            lon0,
            lat0,
            config_dir=config.get("_config_dir"),
        )
        config["optical_config"] = normalized_optical_config
        out_name = normalized_optical_config["outName"]
    else:
        raise ValueError(f"Unsupported data type: {data_type}")

    return data, out_name


def ensure_downsample_data_ready(data, data_type):
    """Validate a pre-built CSI data object before standard downsampling runtime."""

    if data_type == "sar":
        if getattr(data, "dtype", None) != "insar":
            raise TypeError("Adapter SAR input must be a csi.insar.insar object.")
        required = ("lon", "lat", "x", "y", "vel", "los")
    elif data_type == "optical":
        if getattr(data, "dtype", None) != "opticorr":
            raise TypeError("Adapter optical input must be a csi.opticorr.opticorr object.")
        required = ("lon", "lat", "x", "y", "east", "north", "err_east", "err_north")
    else:
        raise ValueError(f"Unsupported data_type: {data_type!r}")

    missing = [
        name for name in required
        if not hasattr(data, name) or getattr(data, name) is None
    ]
    if missing:
        raise TypeError(
            f"Adapter {data_type} input is missing CSI data attributes: "
            + ", ".join(missing)
        )
    if not hasattr(data, "factor"):
        data.factor = 1.0
    return data


def run_downsample_from_data(
    data,
    config,
    out_name,
    *,
    args=None,
    steps=None,
    compute_faults=None,
    plot_faults=None,
    write_metadata=True,
):
    """
    Run the standard downsampling workflow from a pre-built CSI data object.

    This is the stable adapter boundary: callers may build SAR/optical CSI data
    however they need, then hand it to the same runtime used by ecat-downsample.
    """

    data_type = config["data_type"]
    ensure_downsample_data_ready(data, data_type)

    if steps is None:
        steps = {
            "do_covar": bool(config.get("covar", {}).get("do_covar", False)),
            "do_downsample": bool(config.get("downsample", {}).get("enabled", False)),
            "show_raw_data": False,
        }
    if args is None:
        args = argparse.Namespace(
            config=None,
            vmin=None,
            vmax=None,
            workers=config.get("_workers"),
        )

    configure_downsample_compute_backend(config, steps=steps)
    if steps["do_covar"]:
        require_covariance_mask_for_estimation(config)

    lon0 = config["general"]["lon0"]
    lat0 = config["general"]["lat0"]
    if compute_faults is None:
        compute_faults = (
            process_compute_fault_models(config, lon0, lat0)
            if steps["do_downsample"]
            else []
        )
    if plot_faults is None:
        plot_faults = {}
        if steps["show_raw_data"]:
            plot_faults["raw"] = process_plot_fault_overlays(config, lon0, lat0, "raw")
        if steps["do_downsample"]:
            plot_faults["decim"] = process_plot_fault_overlays(config, lon0, lat0, "decim")

    execute_requested_steps(data, config, compute_faults, plot_faults, out_name, args, steps)

    metadata_file = None
    if write_metadata:
        outputs = expected_outputs_for_run(config, out_name, steps)
        metadata_file = write_run_metadata(config, args, out_name, steps, outputs)
    return {
        "out_name": out_name,
        "steps": steps,
        "metadata_file": metadata_file,
    }


def execute_requested_steps(data, config, compute_faults, plot_faults, out_name, args, steps):
    data_type = config["data_type"]
    processing_data = None
    if steps["do_covar"] or steps["do_downsample"]:
        processing_data = build_processing_image(data, data_type)
        region_report = apply_processing_region(
            processing_data,
            config.get("processing_region", {}),
            out_name=out_name,
            base_dir=config.get("_config_dir"),
        )
        processing_data.processing_region_report = region_report
        region_report_text = format_processing_region_report(region_report)
        if region_report_text:
            print(region_report_text)

    covar = None
    if steps["do_covar"]:
        covar = estimate_covariance(processing_data, config)
    elif steps["do_downsample"]:
        covar = read_existing_covariance(
            processing_data,
            data_type,
            missing_policy=config["covar"].get("missing_policy", "existing_or_identity"),
        )

    if steps["do_downsample"]:
        downsampler = run_downsampling(processing_data, data_type, config, compute_faults, out_name)
        report = build_downsample_report(processing_data, data_type, config, downsampler, out_name)
        report_path = write_downsample_report(report, config, out_name)
        if report_path:
            print(format_downsample_report(report, report_path))
        if data_type == "sar":
            sardecim = read_sar_decimated_dataset(config, out_name)
            write_decimated_covariance(sardecim, covar, data_type, out_name)
            plot_sar_downsample_check(sardecim, config, plot_faults.get("decim", []), out_name, args)
        elif data_type == "optical":
            optdecim = read_optical_decimated_dataset(config, out_name)
            write_decimated_covariance(optdecim, covar, data_type, out_name)
            plot_optical_downsample_check(optdecim, config, plot_faults.get("decim", []), out_name, args)

    if steps["show_raw_data"]:
        if data_type == "sar":
            write_sar_metadata_file(data, config["sar_config"], args)
            plot_sar_quicklook(data, config["sar_config"], plot_faults.get("raw", []), args)
        elif data_type == "optical":
            write_optical_metadata_file(data, config["optical_config"], args)
            plot_optical_quicklook(data, config, plot_faults.get("raw", []), out_name, args)


def main():
    args = parse_arguments()
    config, steps = prepare_run(args)
    data, out_name = load_input_data(config, args, steps)
    lon0, lat0 = config["general"]["lon0"], config["general"]["lat0"]
    compute_faults = (
        process_compute_fault_models(config, lon0, lat0)
        if steps["do_downsample"]
        else []
    )
    plot_faults = {}
    if steps["show_raw_data"]:
        plot_faults["raw"] = process_plot_fault_overlays(config, lon0, lat0, "raw")
    if steps["do_downsample"]:
        plot_faults["decim"] = process_plot_fault_overlays(config, lon0, lat0, "decim")
    execute_requested_steps(data, config, compute_faults, plot_faults, out_name, args, steps)
    outputs = expected_outputs_for_run(config, out_name, steps)
    write_run_metadata(config, args, out_name, steps, outputs)


if __name__ == "__main__":
    main()
