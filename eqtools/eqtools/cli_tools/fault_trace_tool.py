"""
Fault Trace Processor (CSI Integrated)
======================================

A tool for simplifying and smoothing fault traces using the `csi.SourceInv` projection engine.
It converts Longitude/Latitude data to Cartesian (XY) coordinates (km) for accurate geometric 
processing, then converts back for output.

Algorithms Supported:
1. RDP (Ramer-Douglas-Peucker): Simplifies based on perpendicular distance. Good for reducing points while keeping corners.
2. VW (Visvalingam-Whyatt): Simplifies based on effective area. Excellent for natural features like faults.
3. B-Spline: Smooths the trace using spline interpolation. Good for noisy data.

Usage Examples:
---------------
1. Run with demo data (no input file needed):
   python fault_trace_tool.py --demo --algo vw --param 2.0

2. Process a file using Visvalingam-Whyatt (VW) algorithm (Recommended for faults):
   python fault_trace_tool.py trace.txt --algo vw --param 0.5 --output result_vw

3. Process a file using RDP algorithm:
   python fault_trace_tool.py trace.txt --algo rdp --param 1.0

4. Smooth a trace using B-Spline:
   python fault_trace_tool.py trace.txt --algo bspline --param 5.0

Input Format:
-------------
Text file with at least two columns: Longitude Latitude
(Space or Tab separated)
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys
import os
import logging

# Ensure csi is installed in the environment
from csi import SourceInv
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
def main():
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
    
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main()
