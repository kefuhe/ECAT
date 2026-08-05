"""Command-line entry point for the optional ECAT trace editor."""

import argparse
from pathlib import Path
import sys

from ..loaders import load_layer
from ..models import LayerSpec
from .adapters import background_from_payload
from .models import InteractiveWorkspace, TraceEditorSession
from .trace_io import read_reference_paths


def build_parser():
    """Build the direct standard-observation trace-editor parser."""

    parser = argparse.ArgumentParser(
        description=(
            "Adjust one working fault trace over an ECAT standard observation "
            "without modifying the observation or reference traces."
        )
    )
    parser.add_argument(
        "observation",
        help=(
            "ECAT standard .nc/.h5 observation, georeferenced GeoTIFF, or "
            "CSI varres prefix when --kind csi_varres is explicit."
        ),
    )
    parser.add_argument(
        "--kind",
        choices=("observation_grid", "raster", "csi_varres"),
        help="Scientific source kind; inferred for .nc/.h5/.tif when omitted.",
    )
    parser.add_argument(
        "--variable",
        help=(
            "Displayed variable. Standard observation grids default to "
            "observation; use corrected_observation explicitly when needed."
        ),
    )
    parser.add_argument(
        "--mask",
        choices=("source_valid", "analysis_valid", "finite"),
        default="source_valid",
        help="Observation-grid display mask (default: source_valid).",
    )
    parser.add_argument(
        "--data-type",
        choices=("sar", "optical"),
        default="sar",
        help="CSI varres row contract (default: sar).",
    )
    parser.add_argument(
        "--trace",
        action="append",
        default=[],
        help=(
            "Read-only TXT/GMT/GeoJSON reference trace. Repeat to show more "
            "than one reference."
        ),
    )
    parser.add_argument(
        "--output",
        default="adjusted_trace.txt",
        help="Initial Save As path (default: adjusted_trace.txt).",
    )
    parser.add_argument("--title", default="ECAT trace editor")
    parser.add_argument("--cmap", default="RdBu_r")
    parser.add_argument("--vmin", type=float)
    parser.add_argument("--vmax", type=float)
    parser.add_argument(
        "--auto-percentile",
        type=float,
        default=99.0,
        help="Central percentile used for automatic color limits (default: 99).",
    )
    parser.add_argument(
        "--display-factor",
        type=float,
        default=1.0,
        help="Display-only value multiplier (default: 1).",
    )
    parser.add_argument("--display-unit")
    parser.add_argument(
        "--basemap",
        choices=("gray", "street", "terrain", "satellite", "none"),
        default="gray",
        help="Initial basemap; it remains switchable in the editor.",
    )
    parser.add_argument(
        "--opacity",
        type=float,
        default=0.82,
        help="Initial observation opacity within [0, 1] (default: 0.82).",
    )
    parser.add_argument(
        "--no-symmetry",
        action="store_true",
        help="Do not force automatic color limits to be symmetric about zero.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5006)
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Start the server without opening the default browser.",
    )
    return parser


def infer_source_kind(path, explicit=None):
    """Return one unambiguous editor source kind."""

    if explicit:
        return explicit
    suffix = Path(path).suffix.lower()
    if suffix in {".nc", ".h5", ".hdf5"}:
        return "observation_grid"
    if suffix in {".tif", ".tiff"}:
        return "raster"
    raise ValueError(
        "Cannot infer the trace-editor source kind. Set --kind explicitly."
    )


def build_session(parsed):
    """Load one detached background and immutable reference collection."""

    source = Path(parsed.observation).resolve()
    kind = infer_source_kind(source, parsed.kind)
    variable = parsed.variable
    if kind == "observation_grid":
        variable = variable or "observation"
    elif kind == "raster":
        variable = variable or "band_1"
    style = {
        "cmap": parsed.cmap,
        "display_factor": parsed.display_factor,
        "symmetry": not parsed.no_symmetry,
        "auto_percentile": parsed.auto_percentile,
        "basemap": parsed.basemap,
        "alpha": parsed.opacity,
    }
    if parsed.display_unit is not None:
        style["display_unit"] = parsed.display_unit
    if parsed.vmin is not None or parsed.vmax is not None:
        style.update({"vmin": parsed.vmin, "vmax": parsed.vmax})
    loader_style = {
        name: value
        for name, value in style.items()
        if name not in {"auto_percentile", "basemap"}
    }
    spec = LayerSpec(
        id="trace_editor_background",
        name=source.stem,
        kind=kind,
        source=source,
        variable=variable,
        mask=parsed.mask if kind == "observation_grid" else None,
        data_type=parsed.data_type if kind == "csi_varres" else None,
        style=loader_style,
    )
    payload = load_layer(spec)
    references = []
    for trace_path in parsed.trace:
        references.extend(read_reference_paths(trace_path))
    workspace = InteractiveWorkspace(references)
    return TraceEditorSession(
        background=background_from_payload(payload, style=style),
        workspace=workspace,
        output_path=Path(parsed.output),
        title=parsed.title,
    )


def main(args=None):
    """Load a standard source and run the local optional Bokeh editor."""

    parser = build_parser()
    parsed = parser.parse_args(sys.argv[1:] if args is None else args)
    try:
        if (parsed.vmin is None) != (parsed.vmax is None):
            raise ValueError("Set both --vmin and --vmax, or neither.")
        if not 0.0 < parsed.auto_percentile <= 100.0:
            raise ValueError("--auto-percentile must be within (0, 100].")
        if not 0.0 <= parsed.opacity <= 1.0:
            raise ValueError("--opacity must be within [0, 1].")
        session = build_session(parsed)
        from .bokeh_trace_editor import run_trace_editor

        run_trace_editor(
            session,
            host=parsed.host,
            port=parsed.port,
            open_browser=not parsed.no_browser,
        )
    except (
        FileNotFoundError,
        ModuleNotFoundError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        parser.error(str(exc))


__all__ = ["build_parser", "build_session", "infer_source_kind", "main"]
