import argparse
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
        description="Run ECAT downsampling from an editable input adapter."
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


def main():
    args = parse_args()
    config, steps = prepare_config(args)
    context = SimpleNamespace(args=args, steps=steps, mode="single")
    data, out_name = input_adapter.load_data(config, context)
    run_downsample_from_data(
        data,
        config,
        out_name,
        args=args,
        steps=steps,
    )


if __name__ == "__main__":
    main()
