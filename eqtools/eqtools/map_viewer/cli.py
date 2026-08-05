"""Command-line entry point for the ECAT research map viewer."""

import argparse
import csv
from dataclasses import replace
from pathlib import Path
import sys

from .backgrounds import create_background_project, with_packaged_backgrounds
from .models import LayerSpec, ViewerProject
from .project import load_viewer_project


def _parse_bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "expected one of: true/false, yes/no, on/off, 1/0"
    )


def build_parser():
    """Build the quick-view and project-style map viewer CLI parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Open curated ECAT background data or an explicit research-map "
            "project in a local browser app."
        )
    )
    parser.add_argument(
        "project",
        nargs="?",
        help=(
            "Optional viewer project YAML. Omit it to browse packaged fault, "
            "block and GNSS layers."
        ),
    )
    parser.add_argument(
        "--catalog",
        help=(
            "Earthquake-client CSV for a one-command review. Packaged "
            "background layers remain available in the sidebar."
        ),
    )
    parser.add_argument(
        "--basemap",
        help="Temporarily override the initial basemap.",
    )
    parser.add_argument(
        "--region",
        nargs=4,
        type=float,
        metavar=("MIN_LON", "MAX_LON", "MIN_LAT", "MAX_LAT"),
        help=(
            "Initial map region. In project mode this temporarily overrides "
            "view.region without changing the YAML."
        ),
    )
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Local Dash host (default: 127.0.0.1).",
    )
    parser.add_argument(
        "--port",
        default=8050,
        type=int,
        help="Local Dash port (default: 8050).",
    )
    parser.add_argument(
        "--debug",
        default=False,
        type=_parse_bool,
        metavar="BOOL",
        help="Enable Dash debug/reloader mode (default: false).",
    )
    return parser


def _catalog_region(path):
    """Return a padded finite region from a canonical earthquake CSV."""

    longitude = []
    latitude = []
    with Path(path).open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        required = {"longitude", "latitude"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                "Earthquake CSV requires longitude and latitude columns."
            )
        for row in reader:
            try:
                lon = float(row["longitude"])
                lat = float(row["latitude"])
            except (TypeError, ValueError):
                continue
            if -360.0 <= lon <= 360.0 and -90.0 <= lat <= 90.0:
                longitude.append(lon)
                latitude.append(lat)
    if not longitude:
        raise ValueError(
            "Earthquake CSV contains no finite longitude/latitude rows."
        )
    west, east = min(longitude), max(longitude)
    south, north = min(latitude), max(latitude)
    lon_pad = max((east - west) * 0.08, 0.25)
    lat_pad = max((north - south) * 0.08, 0.25)
    return (
        west - lon_pad,
        east + lon_pad,
        south - lat_pad,
        north + lat_pad,
    )


def project_from_catalog(path, *, basemap="open-street-map"):
    """Build one explicit viewer project from an earthquake-client CSV."""

    source = Path(path).resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Earthquake catalog does not exist: {source}.")
    project = ViewerProject(
        name=source.stem,
        path=None,
        layers=(
            LayerSpec(
                id="earthquakes",
                name=f"Earthquakes: {source.stem}",
                kind="earthquake_catalog",
                source=source,
                visible=True,
            ),
        ),
        region=_catalog_region(source),
        basemap=basemap,
    )
    return with_packaged_backgrounds(project)


def main(args=None):
    """Run a local read-only ECAT research map viewer."""

    parser = build_parser()
    parsed = parser.parse_args(sys.argv[1:] if args is None else args)
    try:
        if parsed.project and parsed.catalog:
            raise ValueError(
                "Choose either a project YAML or --catalog, not both."
            )
        if parsed.catalog:
            project = project_from_catalog(
                parsed.catalog,
                basemap=parsed.basemap or "open-street-map",
            )
        elif parsed.project:
            project = with_packaged_backgrounds(
                load_viewer_project(parsed.project)
            )
        else:
            project = create_background_project(
                basemap=parsed.basemap or "open-street-map"
            )
        if parsed.region is not None:
            project = replace(project, region=tuple(parsed.region))
        if parsed.basemap is not None:
            project = replace(project, basemap=parsed.basemap)
    except (FileNotFoundError, ModuleNotFoundError, TypeError, ValueError) as exc:
        parser.error(str(exc))
    from .app import create_app

    app = create_app(project)
    app.run(
        host=parsed.host,
        port=parsed.port,
        debug=parsed.debug,
    )


__all__ = ["build_parser", "main", "project_from_catalog"]
