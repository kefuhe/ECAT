import argparse
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from eqtools.csiExtend.downsample.config import normalize_downsample_config
from eqtools.cli_tools.process_data_downsampling import (
    configure_downsample_compute_backend,
    load_config,
    require_covariance_mask_for_estimation,
    resolve_projection_origin,
    resolve_run_steps,
    run_downsample_from_data,
)

import input_adapter


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run ECAT downsampling for adapter-based time-series data."
    )
    parser.add_argument("-f", "--config", default="downsample.yml")
    parser.add_argument("-s", "--show-raw-data", action="store_true")
    parser.add_argument("-c", "--do-covar", action="store_true")
    parser.add_argument("-d", "--do-downsample", action="store_true")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument("--workers", type=int)
    return parser.parse_args()


def prepare_config(args):
    config_path = Path(args.config).resolve()
    config = normalize_downsample_config(load_config(str(config_path)))
    config["_config_dir"] = str(config_path.parent)
    config["_workers"] = args.workers

    steps = resolve_run_steps(args, config)
    configure_downsample_compute_backend(config, steps=steps)
    if steps["do_covar"]:
        require_covariance_mask_for_estimation(config)

    resolve_projection_origin(config)
    print(
        "do_covar: {do_covar}, do_downsample: {do_downsample}, "
        "show_raw_data: {show_raw_data}".format(**steps)
    )
    return config, steps


def configured_epochs(config):
    ts_config = config.get("timeseries", {}) or {}
    epochs = ts_config.get("epochs")
    if not epochs:
        raise ValueError("timeseries.epochs must list the epochs to process.")
    return list(epochs)


def configure_from_rsp(config, rsp_file):
    updated = deepcopy(config)
    downsample = updated.setdefault("downsample", {})
    previous = downsample.get("from_rsp_config", {}) or {}
    downsample["method"] = "from_rsp"
    downsample["from_rsp_config"] = {
        "rsp_file": str(rsp_file),
        "geometry": previous.get("geometry", "auto"),
        "min_valid_fraction": previous.get("min_valid_fraction", 0.0),
        "plot": previous.get("plot", False),
        "decimorig": previous.get("decimorig", 10),
    }
    return normalize_downsample_config(updated)


def run_one_epoch(config, steps, args, epoch, mode):
    context = SimpleNamespace(args=args, steps=steps, mode=mode, epoch=epoch)
    data, out_name = input_adapter.load_epoch_data(config, context, epoch)
    run_downsample_from_data(data, config, out_name, args=args, steps=steps)
    return out_name


def run_independent(config, steps, args):
    for epoch in configured_epochs(config):
        run_one_epoch(config, steps, args, epoch, mode="independent")


def run_reference_grid(config, steps, args):
    if not steps["do_downsample"]:
        print(
            "timeseries.mode='reference_grid' only affects downsampling grid reuse. "
            "No downsampling step was requested, so epochs will be processed independently."
        )
        run_independent(config, steps, args)
        return

    ts_config = config.get("timeseries", {}) or {}
    reference_epoch = ts_config.get("reference_epoch")
    if reference_epoch is None:
        raise ValueError("timeseries.reference_epoch is required for reference_grid mode.")
    epochs = configured_epochs(config)
    if reference_epoch not in epochs:
        raise ValueError("timeseries.reference_epoch must also appear in timeseries.epochs.")

    reference_out = run_one_epoch(config, steps, args, reference_epoch, mode="reference")
    reference_rsp = Path(f"{reference_out}_ifg.rsp")

    reuse_steps = dict(steps)
    reuse_steps["do_covar"] = False
    reuse_steps["do_downsample"] = True
    reuse_config = configure_from_rsp(config, reference_rsp)
    for epoch in epochs:
        if epoch == reference_epoch:
            continue
        run_one_epoch(reuse_config, reuse_steps, args, epoch, mode="reference_grid_reuse")


def main():
    args = parse_args()
    config, steps = prepare_config(args)
    ts_config = config.get("timeseries", {}) or {}
    mode = str(ts_config.get("mode", "independent")).replace("-", "_").lower()
    if mode == "independent":
        run_independent(config, steps, args)
    elif mode == "reference_grid":
        run_reference_grid(config, steps, args)
    else:
        raise ValueError("timeseries.mode must be 'independent' or 'reference_grid'.")


if __name__ == "__main__":
    main()
