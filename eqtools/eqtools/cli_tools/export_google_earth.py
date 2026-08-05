"""Thin command-line entry for ECAT Google Earth exports."""

from __future__ import annotations

import argparse


def _add_output_arguments(parser):
    parser.add_argument("-o", "--output", required=True, help="Output .kmz file")
    parser.add_argument(
        "--document-name",
        help="Top-level name shown in Google Earth",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact output .kmz if it already exists",
    )


def _add_style_arguments(parser):
    parser.add_argument("--cmap", help="Matplotlib colormap")
    parser.add_argument(
        "--vmin",
        type=float,
        help="Fixed color minimum after display-factor is applied",
    )
    parser.add_argument(
        "--vmax",
        type=float,
        help="Fixed color maximum after display-factor is applied",
    )
    symmetry = parser.add_mutually_exclusive_group()
    symmetry.add_argument(
        "--symmetry",
        dest="symmetry",
        action="store_true",
        default=None,
        help="Use an automatic color range symmetric around zero",
    )
    symmetry.add_argument(
        "--no-symmetry",
        dest="symmetry",
        action="store_false",
        help="Use the automatic finite data range",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        help="Layer alpha within [0, 1]: 0 is transparent and 1 is opaque",
    )
    parser.add_argument(
        "--display-factor",
        type=float,
        help="Display-only value multiplier; stored values are unchanged",
    )
    parser.add_argument("--display-unit", help="Unit label after display scaling")
    parser.add_argument(
        "--normalization",
        choices=("linear", "cyclic"),
        help="Color normalization",
    )
    parser.add_argument(
        "--cyclic-period",
        type=float,
        help="Positive display period for cyclic normalization",
    )


def _style_from_args(args):
    names = (
        "cmap",
        "vmin",
        "vmax",
        "symmetry",
        "alpha",
        "display_factor",
        "display_unit",
        "normalization",
        "cyclic_period",
    )
    style = {
        name: getattr(args, name)
        for name in names
        if getattr(args, name, None) is not None
    }
    return style or None


def build_parser():
    """Build the public ``ecat-export-google-earth`` parser."""

    parser = argparse.ArgumentParser(
        prog="ecat-export-google-earth",
        description=(
            "Create display-only Google Earth KMZ files from canonical ECAT "
            "observations, CSI varres results, or earthquake catalogs."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    observation_grid = subparsers.add_parser(
        "observation-grid",
        help="Export one variable from an ECAT standard NetCDF/HDF5 grid",
    )
    observation_grid.add_argument(
        "source",
        help="ECAT standard observation-grid file",
    )
    observation_grid.add_argument(
        "--variable",
        help="Variable such as observation, corrected_observation, east, or north",
    )
    observation_grid.add_argument(
        "--mask",
        choices=("source_valid", "analysis_valid", "finite"),
        default="source_valid",
        help="Pixel mask (default: source_valid)",
    )
    observation_grid.add_argument("--layer-id", default="observation")
    observation_grid.add_argument("--name", help="Layer name")
    _add_style_arguments(observation_grid)
    _add_output_arguments(observation_grid)

    varres = subparsers.add_parser(
        "varres",
        help="Export CSI downsampled cells from a paired .txt/.rsp prefix",
    )
    varres.add_argument("source", help="Common prefix or .txt/.rsp member")
    varres.add_argument(
        "--data-type",
        choices=("sar", "optical"),
        default="sar",
        help="CSI result row contract (default: sar)",
    )
    varres.add_argument(
        "--geometry",
        choices=("auto", "rectangle", "triangle"),
        default="auto",
    )
    varres.add_argument(
        "--component",
        help="SAR observation; optical east, north, or magnitude",
    )
    varres.add_argument("--units", default="m", help="Stored observation unit")
    varres.add_argument("--convention", help="Positive-direction convention")
    varres.add_argument("--layer-id", default="downsampled_cells")
    varres.add_argument("--name", help="Layer name")
    _add_style_arguments(varres)
    _add_output_arguments(varres)

    catalog = subparsers.add_parser(
        "catalog",
        help="Export an earthquake-client CSV catalog as points",
    )
    catalog.add_argument("source", help="Earthquake catalog CSV")
    catalog.add_argument("--layer-id", default="earthquakes")
    catalog.add_argument("--name", default="Earthquakes")
    _add_style_arguments(catalog)
    _add_output_arguments(catalog)

    project = subparsers.add_parser(
        "project",
        help="Export a strict multi-layer project YAML",
    )
    project.add_argument("source", help="Project YAML")
    project.add_argument(
        "--force",
        action="store_true",
        help="Replace the exact configured .kmz if it already exists",
    )
    return parser


def _run_layer_command(args):
    from eqtools.geoexport import (
        cells_from_varres_file,
        earthquakes_from_client_catalog,
        raster_from_observation_file,
        write_kmz,
    )

    style = _style_from_args(args)
    common = {
        "layer_id": args.layer_id,
        "name": args.name,
        "style": style,
    }
    if args.command == "observation-grid":
        layer = raster_from_observation_file(
            args.source,
            variable=args.variable,
            mask=args.mask,
            **common,
        )
    elif args.command == "varres":
        layer = cells_from_varres_file(
            args.source,
            data_type=args.data_type,
            geometry=args.geometry,
            component=args.component,
            units=args.units,
            convention=args.convention,
            **common,
        )
    elif args.command == "catalog":
        layer = earthquakes_from_client_catalog(args.source, **common)
    else:
        raise AssertionError(f"Unhandled command: {args.command}.")
    return write_kmz(
        [layer],
        args.output,
        overwrite=args.force,
        document_name=args.document_name,
    )


def main(argv=None):
    """Run the CLI and return zero after a successful export."""

    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "project":
            from eqtools.geoexport import export_project

            result = export_project(args.source, overwrite=args.force)
        else:
            result = _run_layer_command(args)
    except (
        FileExistsError,
        FileNotFoundError,
        ImportError,
        KeyError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))
    for output_file in result.output_files:
        print(f"Wrote {output_file}")
    for warning in result.warnings:
        print(f"Warning: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
