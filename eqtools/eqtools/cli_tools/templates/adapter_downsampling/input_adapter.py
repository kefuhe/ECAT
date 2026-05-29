"""
Data adapter for ECAT adapter_downsampling templates.

Edit this file when your data source is not covered by the standard
GAMMA/GMTSAR/HyP3/optical readers. The only contract is:

    load_data(config, context) -> (csi_data_object, out_name)

For SAR, return a csi.insar.insar object with lon/lat/x/y/vel/los.
For optical, return a csi.opticorr.opticorr object with
lon/lat/x/y/east/north/err_east/err_north.
"""

from eqtools.csiExtend.downsample.config import get_sar_output_name
from eqtools.cli_tools.process_data_downsampling import (
    process_optical_data,
    process_sar_data,
)


def load_standard_sar(config, context):
    data, sar_config = process_sar_data(
        config["sar_config"],
        config["general"]["lon0"],
        config["general"]["lat0"],
        do_covar=context.steps["do_covar"],
        config_dir=config.get("_config_dir"),
    )
    config["sar_config"] = sar_config
    return data, get_sar_output_name(sar_config)


def load_standard_optical(config, context):
    data, optical_config = process_optical_data(
        config["optical_config"],
        config["general"]["lon0"],
        config["general"]["lat0"],
        config_dir=config.get("_config_dir"),
    )
    config["optical_config"] = optical_config
    return data, optical_config["outName"]


def load_data(config, context):
    """
    Main adapter hook.

    The default implementation delegates to the standard ECAT readers. For a
    custom source, replace this function body and return an already prepared
    CSI data object plus output prefix.
    """

    data_type = config["data_type"]
    if data_type == "sar":
        return load_standard_sar(config, context)
    if data_type == "optical":
        return load_standard_optical(config, context)
    raise ValueError(f"Unsupported data_type: {data_type!r}")


def load_epoch_data(config, context, epoch):
    """
    Time-series adapter hook.

    Replace this for time-series InSAR. Each call should return the CSI data
    object for one epoch and an epoch-specific output prefix.
    """

    raise NotImplementedError(
        "Edit input_adapter.load_epoch_data() to read one time-series epoch."
    )
