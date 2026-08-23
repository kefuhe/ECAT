"""Command-line fault-trace inspection and deterministic preprocessing.

The subcommand interface provides inspect, locate, clean, orient, reverse,
trim, extend, resample, simplify, smooth, and ordered YAML ``apply`` steps.
Distance calculations use a local CSI projection and the numerical operations
are delegated to :mod:`eqtools.csiExtend.trace_ops` through ``TracePath``.

The historical ``INPUT --algo {vw,rdp,bspline}`` invocation remains available
and keeps its original three-output behavior for compatibility.
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import sys
import os
import logging
from pathlib import Path

# Ensure csi is installed in the environment
from csi import SourceInv
from ..csiExtend.trace_io import read_trace, write_trace
from ..csiExtend.trace_processing import TracePath, process_trace
from ..csiExtend.trace_ops import simplify_trace, smooth_trace
from ..viztools import sci_plot_style, set_degree_formatter

# Initialize logger for this module
logger = logging.getLogger(__name__)

class FaultTraceProcessor(SourceInv):
    """
    Fault Trace Processor class inheriting from csi.SourceInv.
    Leverages the projection engine (ll2xy/xy2ll) to perform geometric 
    simplification in a metric space (km).
    """
    
    def __init__(self, name, lon0, lat0, utmzone=None, ellps='WGS84'):
        """
        Initialize the processor and the underlying SourceInv projection.
        
        Args:
            name (str): Name of the fault/project.
            lon0 (float): Reference Longitude for projection center.
            lat0 (float): Reference Latitude for projection center.
            utmzone (int, optional): UTM zone. Defaults to None (auto).
            ellps (str): Ellipsoid. Defaults to 'WGS84'.
        """
        super(FaultTraceProcessor, self).__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0)
        
        self.raw_lonlat = None      # Original Longitude/Latitude
        self.trace_xy = None        # Projected XY coordinates (km)
        self.processed_xy = None    # Processed (Simplified/Smoothed) XY coordinates (km)
        self.algorithm_info = "None"

        logger.info(f"[Init] Processor initialized. Projection Center: ({lon0:.4f}, {lat0:.4f})")

    def generate_demo_data(self):
        """
        Generates synthetic fault trace data for testing purposes.
        Creates a noisy sine wave to simulate a natural fault trace.
        """
        logger.info("[Demo] Generating synthetic fault trace data...")
        # Generate data in XY space (km)
        t = np.linspace(0, 50, 200) # 50km long
        x = t
        # A sine wave with random noise to simulate rough topography/trace
        y = 2.0 * np.sin(t / 5.0) + np.random.normal(0, 0.1, 200)
        
        self.trace_xy = np.column_stack((x, y))
        
        # Reverse project to Lat/Lon to simulate "raw input"
        lons, lats = self.xy2ll(x, y)
        self.raw_lonlat = np.column_stack((lons, lats))
        
        logger.info(f"[Demo] Generated {len(self.trace_xy)} points.")

    def load_and_project(self, input_source):
        """
        Loads Longitude/Latitude data and projects to XY (km).
        Args:
            input_source: Filepath (str) or numpy array (N, 2).
        """
        try:
            # Load data (handles space/tab delimiters automatically)
            if isinstance(input_source, str):
                data = np.loadtxt(input_source)
            else:
                data = input_source

            if data.ndim != 2 or data.shape[1] < 2:
                raise ValueError("Data must have at least 2 columns (Lon Lat)")
            
            self.raw_lonlat = data[:, :2]
            
            # Core Step: Project Lon/Lat -> X/Y (km) using SourceInv
            x, y = self.ll2xy(self.raw_lonlat[:, 0], self.raw_lonlat[:, 1])
            self.trace_xy = np.column_stack((x, y))
            
            logger.info(f"[Data] Loaded {len(self.trace_xy)} points. Projected to XY plane.")
            
        except Exception as e:
            logger.error(f"[Error] Failed to load data: {e}")
            sys.exit(1)

    def simplify_rdp(self, epsilon_km=1.0):
        """
        Executes RDP simplification.
        Args:
            epsilon_km (float): Max distance deviation in km.
        """
        if self.trace_xy is None: return
        logger.info(f"[Algo] Running RDP (Tolerance={epsilon_km} km)...")
        self.processed_xy = simplify_trace(self.trace_xy, method="rdp", tolerance=epsilon_km)
        self.algorithm_info = f"RDP (eps={epsilon_km} km)"

    # =========================================================
    # Algorithm 2: Visvalingam-Whyatt (VW)
    # =========================================================
    def simplify_vw(self, area_threshold=1.0):
        """
        Executes Visvalingam-Whyatt simplification.
        Args:
            area_threshold (float): Minimum effective area in km^2.
        """
        if self.trace_xy is None: return
        logger.info(f"[Algo] Running Visvalingam-Whyatt (Area Threshold={area_threshold} km^2)...")
        self.processed_xy = simplify_trace(self.trace_xy, method="vw", tolerance=area_threshold)
        self.algorithm_info = f"Visvalingam-Whyatt (area={area_threshold})"

    # =========================================================
    # Algorithm 3: B-Spline Smoothing
    # =========================================================
    def smooth_bspline(self, smooth_factor=1.0, num_points=None):
        """
        Executes B-Spline smoothing.
        Args:
            smooth_factor (float): Smoothing factor 's'. Larger = smoother.
            num_points (int): Number of output points.
        """
        if self.trace_xy is None: return
        logger.info(f"[Algo] Running B-Spline Smoothing (s={smooth_factor})...")

        if len(self.trace_xy) < 4:
            logger.warning("[Warning] Not enough points for B-Spline (need > 3).")
            self.processed_xy = self.trace_xy
            return

        try:
            n_out = num_points if num_points else len(self.trace_xy)
            self.processed_xy = smooth_trace(
                self.trace_xy,
                method="bspline",
                smoothing=smooth_factor,
                num_points=n_out,
            )
            self.algorithm_info = f"B-Spline (s={smooth_factor})"
        except Exception as e:
            logger.error(f"[Error] B-Spline failed: {e}")
            self.processed_xy = self.trace_xy

    # =========================================================
    # Geometry Calculation & Export
    # =========================================================
    def _compute_segment_geometry(self):
        """
        Computes geometry (Length, Strike, Midpoint) for each segment 
        of the processed trace.
        Returns: List of dictionaries.
        """
        if self.processed_xy is None or len(self.processed_xy) < 2:
            return []

        segments = []
        points = self.processed_xy
        
        for i in range(len(points) - 1):
            p1 = points[i]
            p2 = points[i+1]
            
            # 1. Length (km)
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            length = np.sqrt(dx**2 + dy**2)
            
            # 2. Strike (Degrees, 0-360, Clockwise from North)
            # Math angle is counter-clockwise from East.
            # Strike = 90 - math_angle
            math_angle = np.degrees(np.arctan2(dy, dx))
            strike = 90 - math_angle
            if strike < 0: strike += 360
            
            # 3. Midpoint (XY -> Lon/Lat)
            mid_x = (p1[0] + p2[0]) / 2.0
            mid_y = (p1[1] + p2[1]) / 2.0
            mid_lon, mid_lat = self.xy2ll(mid_x, mid_y)
            
            segments.append({
                'id': i + 1,
                'name': f"Seg#{i+1}",
                'lon': mid_lon,
                'lat': mid_lat,
                'length': length,
                'strike': strike,
                'mid_xy': (mid_x, mid_y)
            })
            
        return segments

    def save_fixed_params(self, output_prefix):
        """
        Generates the 'fixed_params' YAML-like file for inversion.
        """
        segments = self._compute_segment_geometry()
        if not segments: return

        filename = f"{output_prefix}_fixed_params.txt"
        
        with open(filename, 'w') as f:
            f.write("fixed_params:\n")
            for seg in segments:
                f.write(f"  {seg['name']}:\n")
                f.write(f"    lon: {seg['lon']:.3f}\n")
                f.write(f"    lat: {seg['lat']:.3f}\n")
                f.write(f"    depth: 0.00\n")
                f.write(f"    length: {seg['length']:.2f}\n")
                f.write(f"    strike: {seg['strike']:.2f}\n")
        
        logger.info(f"[Output] Saved fixed parameters to: {filename}")

    def save_trace_file(self, output_prefix):
        """Saves the simplified trace coordinates."""
        if self.processed_xy is None: return
        lons, lats = self.xy2ll(self.processed_xy[:, 0], self.processed_xy[:, 1])
        txt_filename = f"{output_prefix}_trace.txt"
        np.savetxt(txt_filename, np.column_stack((lons, lats)), fmt='%.6f', header="Lon Lat")
        logger.info(f"[Output] Saved simplified trace to: {txt_filename}")

    def plot_comparison(self, output_prefix, style=['science', 'no-latex'], figsize='single', is_lonlat=False):
        """
        Plots the original vs. processed trace in the projected XY plane.

        Args:
        figsize : str, float, or tuple, optional
            Figure size specification:
            - str: predefined column width name ('single', 'double', 'nature', 'ieee', 'ieee_double', 'a4')
            - float: custom width (height computed via figsize_aspect)
            - tuple: (width, height) in figsize_unit
            Default is None (use rcParams default).
        is_lonlat : bool, optional
            If True, converts Lon/Lat back to XY for plotting. Default is False.
        """
        with sci_plot_style(style, figsize=figsize):
            if is_lonlat:
                # Fix: Plot actual Lon/Lat, not projected XY
                x_orig, y_orig = self.raw_lonlat[:, 0], self.raw_lonlat[:, 1]
                x_proc, y_proc = self.xy2ll(self.processed_xy[:, 0], self.processed_xy[:, 1])
            else:
                x_orig, y_orig = self.trace_xy[:, 0], self.trace_xy[:, 1]
                x_proc, y_proc = self.processed_xy[:, 0], self.processed_xy[:, 1]
            # Plot Original (Grey)
            plt.plot(x_orig, y_orig, 'k.', alpha=0.2, label='Original Points')
            plt.plot(x_orig, y_orig, 'k-', alpha=0.1, linewidth=1)
            
            # Plot Processed (Red)
            if self.processed_xy is not None:
                pts_count = len(self.processed_xy)
                ratio = (1 - pts_count / len(self.trace_xy)) * 100
                label = f"{self.algorithm_info}\nPoints: {pts_count} (Reduced {ratio:.1f}%)"
                
                plt.plot(x_proc, y_proc, '-', color='#0c5da5', linewidth=1.5, label=label)
                plt.scatter(x_proc, y_proc, c='#0c5da5', marker='x', zorder=5) # , c='red', s=15

                # Label Segments
                segments = self._compute_segment_geometry()
                for seg in segments:
                    if is_lonlat:
                        mx, my = seg['lon'], seg['lat']
                    else:
                        mx, my = seg['mid_xy']
                    # Add text label with a small offset or box
                    plt.text(mx, my, seg['name'], fontsize=9, color='blue', fontweight='bold',
                            ha='center', va='bottom', zorder=10,
                            bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

            if is_lonlat:
                plt.title(f"Fault Trace Analysis \nCenter: {self.lon0:.2f}, {self.lat0:.2f}")
                plt.xlabel("Longitude (°)")
                plt.ylabel("Latitude (°)")
            else:
                plt.title(f"Fault Trace Analysis (Projected XY)\nCenter: {self.lon0:.2f}, {self.lat0:.2f}")
                plt.xlabel("Easting (km)")
                plt.ylabel("Northing (km)")
            plt.axis('equal') # Maintain geometric aspect ratio
            if is_lonlat:
                set_degree_formatter(plt.gca(), axis='both')
            plt.legend()
            plt.grid(True, linestyle=':', alpha=0.5)
            
            img_filename = f"{output_prefix}_plot.png"
            plt.savefig(img_filename, dpi=600)
            logger.info(f"[Output] Saved plot to: {img_filename}")
            plt.close()

# =========================================================
# Main Execution
# =========================================================
def _legacy_main(argv=None):
    parser = argparse.ArgumentParser(
        description="Fault Trace Processor (CSI Integrated)",
        epilog="Use --demo to run without an input file."
    )
    
    # Input file is optional if --demo is used
    parser.add_argument('input_file', nargs='?', help="Input file path (Lon Lat columns)")
    
    # Mode selection
    parser.add_argument('--demo', action='store_true', help="Run in demo mode with synthetic data")
    
    # Algorithm selection
    parser.add_argument('--algo', choices=['rdp', 'vw', 'bspline'], default='vw', 
                        help="Algorithm: rdp (distance), vw (area, default), bspline (smooth)")
    parser.add_argument('--param', type=float, 
                        help="Parameter: RDP(dist km) / VW(area km2) / BSpline(smooth factor)")
    
    # Output and Projection
    parser.add_argument('--output', default='output', help="Output filename prefix")
    parser.add_argument('--lon0', type=float, help="Projection Center Lon (Optional, auto-calculated from data)")
    parser.add_argument('--lat0', type=float, help="Projection Center Lat (Optional, auto-calculated from data)")
    
    args = parser.parse_args(argv)
    
    # Configure logging for CLI execution
    # format='%(message)s' keeps the output clean, preserving your [Tag] style
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    # 1. Determine Mode (File vs Demo)
    if args.demo:
        # Use arbitrary center for demo
        lon0 = args.lon0 if args.lon0 is not None else 100.0
        lat0 = args.lat0 if args.lat0 is not None else 30.0
        processor = FaultTraceProcessor(name=args.output, lon0=lon0, lat0=lat0)
        processor.generate_demo_data()
    else:
        # File mode
        if not args.input_file or not os.path.exists(args.input_file):
            parser.print_help()
            logger.error("\n[Error] Input file required unless --demo is specified.")
            return

        # Pre-read to determine center if not provided
        # Optimization: Read once, pass data to processor
        raw_data = np.loadtxt(args.input_file)
        lon0 = args.lon0 if args.lon0 is not None else np.mean(raw_data[:, 0])
        lat0 = args.lat0 if args.lat0 is not None else np.mean(raw_data[:, 1])
        
        processor = FaultTraceProcessor(name=args.output, lon0=lon0, lat0=lat0)
        processor.load_and_project(raw_data)
    
    # 2. Execute Algorithm (in XY space)
    if args.algo == 'rdp':
        val = args.param if args.param is not None else 1.0
        processor.simplify_rdp(epsilon_km=val)
    elif args.algo == 'vw':
        val = args.param if args.param is not None else 5.0 # Default area threshold
        processor.simplify_vw(area_threshold=val)
    elif args.algo == 'bspline':
        val = args.param if args.param is not None else 10.0
        processor.smooth_bspline(smooth_factor=val)
        
    # 3. Save Results and Plot
    processor.save_trace_file(args.output)
    processor.save_fixed_params(args.output) # Generate fixed_params file
    processor.plot_comparison(args.output)   # Plot in projected XY plane


_TRACE_COMMANDS = {
    "inspect",
    "locate",
    "clean",
    "orient",
    "reverse",
    "trim",
    "extend",
    "resample",
    "simplify",
    "smooth",
    "apply",
}


def _add_trace_source_arguments(parser):
    parser.add_argument("input", help="Input TXT/GMT/GeoJSON lon/lat trace")
    parser.add_argument(
        "--segment",
        type=int,
        help="Multipart trace segment index (required only when input has several parts)",
    )
    parser.add_argument("--lon0", type=float, help="Projection center longitude")
    parser.add_argument("--lat0", type=float, help="Projection center latitude")
    parser.add_argument("--utmzone", type=int, help="Explicit UTM zone")
    parser.add_argument("--ellps", default="WGS84", help="Projection ellipsoid (default: WGS84)")


def _add_trace_output_arguments(parser):
    parser.add_argument("-o", "--output", required=True, help="Output trace path")
    parser.add_argument("--overwrite", action="store_true", help="Explicitly replace output files")
    parser.add_argument(
        "--plot",
        nargs="?",
        const="auto",
        help="Write a comparison PNG; omit the path to derive it from --output",
    )
    parser.add_argument("--report", help="Optional JSON processing report path")


def _add_marker_arguments(parser, prefix):
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{prefix}-km", type=float, dest=f"{prefix}_km")
    group.add_argument(f"--{prefix}-fraction", type=float, dest=f"{prefix}_fraction")
    group.add_argument(f"--{prefix}-lon", type=float, dest=f"{prefix}_lon")
    group.add_argument(f"--{prefix}-lat", type=float, dest=f"{prefix}_lat")
    group.add_argument(
        f"--{prefix}-nearest",
        nargs=2,
        type=float,
        metavar=("LON", "LAT"),
        dest=f"{prefix}_nearest",
    )
    parser.add_argument(
        f"--{prefix}-which",
        default="first",
        dest=f"{prefix}_which",
        help="Coordinate intersection: first, last, or zero-based index",
    )


def _build_command_parser():
    parser = argparse.ArgumentParser(
        prog="ecat-fault-trace-tool",
        description="Inspect and preprocess one ECAT fault trace without modifying a fault object.",
        epilog=(
            "Legacy 'INPUT --algo rdp|vw|bspline --param VALUE' calls remain supported. "
            "New scripts should use the explicit subcommands above."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    inspect_parser = subparsers.add_parser("inspect", help="Report trace points, length, and endpoints")
    _add_trace_source_arguments(inspect_parser)
    inspect_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    locate_parser = subparsers.add_parser("locate", help="Preview one marker on the trace")
    _add_trace_source_arguments(locate_parser)
    locate_group = locate_parser.add_mutually_exclusive_group(required=True)
    locate_group.add_argument("--lon", type=float)
    locate_group.add_argument("--lat", type=float)
    locate_group.add_argument("--nearest", nargs=2, type=float, metavar=("LON", "LAT"))
    locate_group.add_argument("--distance-km", type=float)
    locate_group.add_argument("--fraction", type=float)
    locate_parser.add_argument("--which", default="first")
    locate_parser.add_argument("--max-snap-km", type=float)
    locate_parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")

    clean_parser = subparsers.add_parser("clean", help="Remove consecutive duplicate vertices")
    _add_trace_source_arguments(clean_parser)
    _add_trace_output_arguments(clean_parser)
    clean_parser.add_argument("--atol-km", type=float, default=0.0)

    orient_parser = subparsers.add_parser("orient", help="Force the first endpoint to a map side")
    _add_trace_source_arguments(orient_parser)
    _add_trace_output_arguments(orient_parser)
    orient_parser.add_argument(
        "--start",
        choices=("west", "east", "south", "north"),
        default="west",
    )

    reverse_parser = subparsers.add_parser("reverse", help="Reverse trace point order")
    _add_trace_source_arguments(reverse_parser)
    _add_trace_output_arguments(reverse_parser)

    trim_parser = subparsers.add_parser(
        "trim",
        help="Keep an along-trace interval resolved from distance, coordinate, or nearest point",
    )
    _add_trace_source_arguments(trim_parser)
    _add_trace_output_arguments(trim_parser)
    _add_marker_arguments(trim_parser, "start")
    _add_marker_arguments(trim_parser, "end")
    trim_parser.add_argument(
        "--max-snap-km",
        type=float,
        help="Reject nearest-point markers farther than this distance",
    )

    extend_parser = subparsers.add_parser("extend", help="Extend endpoints along local tangents")
    _add_trace_source_arguments(extend_parser)
    _add_trace_output_arguments(extend_parser)
    extend_parser.add_argument("--start-km", type=float, default=0.0)
    extend_parser.add_argument("--end-km", type=float, default=0.0)
    extend_parser.add_argument("--tangent-window", type=int, default=1)

    resample_parser = subparsers.add_parser("resample", help="Resample by arc-length spacing or count")
    _add_trace_source_arguments(resample_parser)
    _add_trace_output_arguments(resample_parser)
    resample_group = resample_parser.add_mutually_exclusive_group(required=True)
    resample_group.add_argument("--every-km", type=float)
    resample_group.add_argument("--num-points", type=int)

    simplify_parser = subparsers.add_parser("simplify", help="Simplify with RDP or VW")
    _add_trace_source_arguments(simplify_parser)
    _add_trace_output_arguments(simplify_parser)
    simplify_parser.add_argument("--method", choices=("rdp", "vw"), default="vw")
    simplify_parser.add_argument(
        "--tolerance",
        type=float,
        required=True,
        help="RDP distance in km or VW effective area in km^2",
    )

    smooth_parser = subparsers.add_parser("smooth", help="Smooth with B-Spline or Savitzky-Golay")
    _add_trace_source_arguments(smooth_parser)
    _add_trace_output_arguments(smooth_parser)
    smooth_parser.add_argument("--method", choices=("bspline", "savgol"), default="bspline")
    smooth_parser.add_argument("--smoothing", type=float, default=1.0)
    smooth_parser.add_argument("--num-points", type=int)
    smooth_parser.add_argument("--window", type=int, default=5)
    smooth_parser.add_argument("--polyorder", type=int, default=2)
    smooth_parser.add_argument("--move-endpoints", action="store_true")

    apply_parser = subparsers.add_parser("apply", help="Apply ordered operations from YAML")
    _add_trace_source_arguments(apply_parser)
    _add_trace_output_arguments(apply_parser)
    apply_parser.add_argument("--config", required=True, help="YAML with an operations list")
    return parser


def _load_trace_path(args):
    segment = read_trace(args.input, segment=args.segment)
    return TracePath.from_lonlat(
        segment.coordinates,
        lon0=args.lon0,
        lat0=args.lat0,
        utmzone=args.utmzone,
        ellps=args.ellps,
    )


def _marker_from_arguments(args, prefix):
    which = getattr(args, f"{prefix}_which")
    max_snap_km = getattr(args, "max_snap_km", None)
    value = getattr(args, f"{prefix}_km")
    if value is not None:
        return {"trace_distance_km": value}
    value = getattr(args, f"{prefix}_fraction")
    if value is not None:
        return {"fraction": value}
    value = getattr(args, f"{prefix}_lon")
    if value is not None:
        return {"longitude": value, "which": which}
    value = getattr(args, f"{prefix}_lat")
    if value is not None:
        return {"latitude": value, "which": which}
    value = getattr(args, f"{prefix}_nearest")
    if value is not None:
        marker = {"nearest": value, "coord_system": "lonlat"}
        if max_snap_km is not None:
            marker["max_distance_km"] = max_snap_km
        return marker
    return None


def _locate_marker_from_arguments(args):
    if args.lon is not None:
        return {"longitude": args.lon, "which": args.which}
    if args.lat is not None:
        return {"latitude": args.lat, "which": args.which}
    if args.nearest is not None:
        marker = {"nearest": args.nearest, "coord_system": "lonlat"}
        if args.max_snap_km is not None:
            marker["max_distance_km"] = args.max_snap_km
        return marker
    if args.distance_km is not None:
        return {"trace_distance_km": args.distance_km}
    return {"fraction": args.fraction}


def _resolved_plot_path(args):
    if not args.plot:
        return None
    if args.plot != "auto":
        return Path(args.plot)
    output = Path(args.output)
    return output.with_suffix(".png")


def _preflight_output(path, *, overwrite):
    target = Path(path)
    if not target.parent.exists():
        raise FileNotFoundError(f"output directory does not exist: {target.parent}.")
    if target.exists() and not overwrite:
        raise FileExistsError(f"output already exists: {target}.")


def _plot_trace_result(original, result, path):
    with sci_plot_style(["science", "no-latex"], figsize="single"):
        figure, axes = plt.subplots()
        axes.plot(
            original.lonlat[:, 0],
            original.lonlat[:, 1],
            "k.-",
            alpha=0.35,
            linewidth=0.8,
            label="Input",
        )
        axes.plot(
            result.lonlat[:, 0],
            result.lonlat[:, 1],
            color="#0c5da5",
            marker="x",
            linewidth=1.4,
            label="Processed",
        )
        axes.set_xlabel("Longitude (°)")
        axes.set_ylabel("Latitude (°)")
        axes.set_aspect("equal", adjustable="datalim")
        set_degree_formatter(axes, axis="both")
        axes.legend()
        axes.grid(True, linestyle=":", alpha=0.5)
        figure.savefig(path, dpi=300, bbox_inches="tight")
        plt.close(figure)


def _save_trace_result(args, original, result):
    plot_path = _resolved_plot_path(args)
    targets = [Path(args.output)]
    if args.report:
        targets.append(Path(args.report))
    if plot_path is not None:
        targets.append(plot_path)
    for target in targets:
        _preflight_output(target, overwrite=args.overwrite)

    output = write_trace(
        args.output,
        result.lonlat,
        name=Path(args.output).stem,
        role="processed_trace",
        overwrite=args.overwrite,
    )
    if args.report:
        report = result.report()
        report.update({"input": str(Path(args.input).resolve()), "output": str(output)})
        with Path(args.report).open(
            "w" if args.overwrite else "x",
            encoding="utf-8",
            newline="\n",
        ) as stream:
            json.dump(report, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    if plot_path is not None:
        _plot_trace_result(original, result, plot_path)
    operation = result.history[-1].name if result.history else "none"
    logger.info(
        "[Output] %s: %d -> %d points, %.6g -> %.6g km",
        operation,
        original.point_count,
        result.point_count,
        original.length_km,
        result.length_km,
    )
    logger.info("[Output] Saved trace to: %s", output)
    return 0


def _print_trace_report(report, *, as_json):
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print(f"points: {report['point_count']}")
    print(f"length_km: {report['length_km']:.8g}")
    print(f"start_lonlat: {report['start_lonlat'][0]:.8f}, {report['start_lonlat'][1]:.8f}")
    print(f"end_lonlat: {report['end_lonlat'][0]:.8f}, {report['end_lonlat'][1]:.8f}")
    projection = report["projection"]
    if "lon0" in projection:
        print(f"projection_center: {projection['lon0']:.8f}, {projection['lat0']:.8f}")


def _run_trace_command(args):
    original = _load_trace_path(args)
    if args.command == "inspect":
        _print_trace_report(original.report(), as_json=args.json)
        return 0
    if args.command == "locate":
        marker_spec = _locate_marker_from_arguments(args)
        candidates = original.resolve_markers(marker_spec)
        selected = original.resolve_marker(marker_spec)
        report = {
            "candidate_count": len(candidates),
            "selected": selected.to_dict(),
            "candidates": [
                {**item.to_dict(), "candidate_index": index, "candidate_count": len(candidates)}
                for index, item in enumerate(candidates)
            ],
        }
        if args.json:
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(f"matches: {len(candidates)}")
            print(f"selected: {selected.candidate_index}")
            print(f"method: {selected.method}")
            print(f"trace_distance_km: {selected.trace_distance_km:.8g}")
            if selected.lonlat is not None:
                print(f"resolved_lonlat: {selected.lon:.8f}, {selected.lat:.8f}")
            print(f"distance_to_trace_km: {selected.distance_to_trace_km:.8g}")
        return 0
    if args.command == "clean":
        result = original.clean(atol_km=args.atol_km)
    elif args.command == "orient":
        result = original.orient(start=args.start)
    elif args.command == "reverse":
        result = original.reverse()
    elif args.command == "trim":
        start = _marker_from_arguments(args, "start")
        end = _marker_from_arguments(args, "end")
        if start is None and end is None:
            raise ValueError("trim requires at least one --start-* or --end-* marker.")
        result = original.trim(start=start, end=end)
    elif args.command == "extend":
        if args.start_km == 0.0 and args.end_km == 0.0:
            raise ValueError("extend requires a positive --start-km or --end-km.")
        result = original.extend(
            start_km=args.start_km,
            end_km=args.end_km,
            tangent_window=args.tangent_window,
        )
    elif args.command == "resample":
        result = original.resample(every_km=args.every_km, num_points=args.num_points)
    elif args.command == "simplify":
        result = original.simplify(method=args.method, tolerance=args.tolerance)
    elif args.command == "smooth":
        result = original.smooth(
            method=args.method,
            smoothing=args.smoothing,
            num_points=args.num_points,
            window=args.window,
            polyorder=args.polyorder,
            preserve_endpoints=not args.move_endpoints,
        )
    elif args.command == "apply":
        import yaml

        with Path(args.config).open("r", encoding="utf-8-sig") as stream:
            config = yaml.safe_load(stream) or {}
        operations = config if isinstance(config, list) else config.get("operations")
        if not isinstance(operations, list) or not operations:
            raise ValueError("trace config must define a non-empty operations list.")
        result = process_trace(original, operations)
    else:
        raise ValueError(f"unsupported trace command: {args.command}.")
    return _save_trace_result(args, original, result)


def _command_main(argv):
    parser = _build_command_parser()
    if not argv:
        parser.print_help()
        return 0
    args = parser.parse_args(argv)
    if args.command is None:
        parser.print_help()
        return 0
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    try:
        return _run_trace_command(args)
    except (OSError, TypeError, ValueError, IndexError) as exc:
        logger.error("[Error] %s", exc)
        return 2


def main(argv=None):
    """Run the explicit trace CLI or the backward-compatible legacy form."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments or arguments[0] in _TRACE_COMMANDS or arguments[0] in {"-h", "--help"}:
        return _command_main(arguments)
    return _legacy_main(arguments)


if __name__ == "__main__":
    raise SystemExit(main())
