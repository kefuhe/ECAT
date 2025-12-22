import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.tri as mtri
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA

# Import base class from CSI
from csi.SourceInv import SourceInv
from .AdaptiveLayeredDipTriangularPatches import AdaptiveLayeredDipTriangularPatches
from .AdaptiveLayeredDipRectangularPatches import AdaptiveLayeredDipRectangularPatches
from ..plottools import sci_plot_style
from ..plottools import set_degree_formatter

class FaultGeometryEngine(SourceInv):
    """
    Fault Geometry Processing Engine.
    
    Capabilities:
    1. Manage and process depth contour layers (Layers).
    2. Perform geometric operations: Extrapolation, Interpolation, Trimming, Resampling.
    3. Generate Depth-Dip profiles required for rectangular fault construction.
    4. Inverse extraction of depth contours from existing fault models (Fault Mesh).
    
    Inherits from SourceInv to utilize projection methods like ll2xy, xy2ll, utmzone.
    """

    def __init__(self, name="GeometryEngine", utmzone=None, ellps='WGS84', 
                 lon0=None, lat0=None, verbose=True):
        
        # Initialize base class and coordinate system
        super(FaultGeometryEngine, self).__init__(name, utmzone=utmzone, ellps=ellps, 
                                                  lon0=lon0, lat0=lat0)
        self.verbose = verbose
        
        # Core data container: {depth_km: np.array([[x, y, z, lon, lat], ...])}
        # Stores 5 columns for easy access.
        self.layers = {} 
        
    # =================================================================
    # 1. Basic Data Management (Add & Resample)
    # =================================================================
    
    def add_layer(self, coords, depth, coords_type='ll', sort_by=None, ascending=True):
        """
        Add a single depth contour layer.
        
        Args:
            coords: (N, 2) or (N, 3) array.
            depth: Depth value (positive, km).
            coords_type: 'll' (Longitude/Latitude) or 'xy' (Kilometers).
            sort_by: str, optional sorting key ['lat', 'lon', 'x', 'y']. Default is None (keep order).
            ascending: bool, True for ascending, False for descending. Default is True.
        """
        depth = abs(depth)
        n_points = len(coords)
        
        # 1. Coordinate conversion and completion
        if coords_type == 'll':
            lon, lat = coords[:, 0], coords[:, 1]
            x, y = self.ll2xy(lon, lat)
        elif coords_type == 'xy':
            x, y = coords[:, 0], coords[:, 1]
            lon, lat = self.xy2ll(x, y)
        else:
            raise ValueError("coords_type must be 'll' or 'xy'")
            
        # 2. Unified storage format: [x, y, z, lon, lat]
        # Column Indices: 0:x, 1:y, 2:z, 3:lon, 4:lat
        z = np.full(n_points, depth)
        data = np.column_stack([x, y, z, lon, lat])
        
        # 3. Sorting (if specified)
        if sort_by is not None:
            # Map column names to indices
            col_map = {'x': 0, 'y': 1, 'lon': 3, 'lat': 4}
            
            if sort_by in col_map:
                col_idx = col_map[sort_by]
                sort_values = data[:, col_idx]
                
                # Get sort indices (default ascending)
                sort_indices = np.argsort(sort_values)
                
                # Reverse if descending
                if not ascending:
                    sort_indices = sort_indices[::-1]
                
                # Apply sort
                data = data[sort_indices]
                if self.verbose:
                    direction = "ascending" if ascending else "descending"
                    print(f"  Layer sorted by {sort_by} ({direction})")
            else:
                print(f"Warning: Invalid sort_by key '{sort_by}'. Options: {list(col_map.keys())}")

        # 4. Store
        self.layers[depth] = data
        
        if self.verbose:
            print(f"Added layer at {depth} km ({n_points} points)")

    def load_from_slab2(self, grd_file, target_levels, min_points=50, stitch_mode='lat'):
        """
        Batch load depth contours from a Slab2 GRD file.
        Automatically handles negative-positive depth conversion:
        User inputs positive depths -> Internal negative slicing -> Stored as positive depths.
        
        Args:
            grd_file (str): Path to .grd file.
            target_levels (list): List of depths to extract (positive, km), e.g., [20, 40, 60, 80].
            min_points (int): Ignore segments with fewer points than this.
            stitch_mode (str): 'lat' (Sort North-South) or 'lon' (Sort West-East).
        """
        if not os.path.exists(grd_file):
            raise FileNotFoundError(f"{grd_file} not found")
            
        import xarray as xr
        ds = xr.open_dataset(grd_file)
        z_grid = ds['z'] # Slab2 data is typically negative (e.g., -20.5)
        
        # --- Pre-process target depths ---
        # Users provide positive values (20, 40), but contour needs negative values (-40, -20)
        # Levels must be sorted ascending.
        
        if target_levels is None or len(target_levels) == 0:
            raise ValueError("Must provide target_levels (e.g., [20, 40, 60]).")
            
        # 1. Convert to negative for querying
        query_levels = [-abs(d) for d in target_levels]
        # 2. Sort (matplotlib requires ascending levels: -80, -60, ...)
        query_levels = sorted(query_levels)
        
        # --- Execute Contouring ---
        # Use matplotlib backend extraction, no display
        fig, ax = plt.subplots()
        try:
            contours = ax.contour(ds['x'], ds['y'], z_grid, levels=query_levels)
        except Exception as e:
            plt.close(fig)
            print(f"Error during contouring: {e}")
            return
            
        plt.close(fig) # Close immediately to prevent popups/memory leaks
        
        if self.verbose:
            print(f"Loading Slab2 contours from {os.path.basename(grd_file)}...")
            print(f"  Target depths (km): {[abs(l) for l in query_levels]}")

        count = 0
        # --- Parse contour results ---
        for i, level in enumerate(contours.levels):
            # level is negative here (e.g., -20.0)
            paths = contours.collections[i].get_paths()
            
            # 1. Collect valid segments
            valid_segments = []
            for path in paths:
                if len(path.vertices) > min_points:
                    valid_segments.append(path.vertices)
            
            if not valid_segments:
                continue
                
            # 2. Stitching
            if stitch_mode == 'lat':
                # N-S strike: Sort by latitude (North to South)
                valid_segments.sort(key=lambda seg: np.mean(seg[:, 1]), reverse=True)
            elif stitch_mode == 'lon':
                # E-W strike: Sort by longitude (West to East)
                valid_segments.sort(key=lambda seg: np.mean(seg[:, 0]), reverse=False)
                
            merged_xy = np.vstack(valid_segments)
            
            # --- Store back as Positive Value ---
            # Use positive depth (20.0) as key for easier access
            positive_depth = abs(level)
            
            self.add_layer(merged_xy, positive_depth, coords_type='ll')
            count += 1
            
        if self.verbose:
            print(f"Successfully loaded {count} layers into engine.")

    def inspect_slab2_in_3d(self, grd_file, stride=10, z_exaggeration=0.5):
        """
        [Utility] Interactive 3D preview of Slab2 data.
        
        Args:
            grd_file (str): Path to .grd file.
            stride (int): Downsampling stride. Larger value = smoother but less detail. Default 10.
            z_exaggeration (float): Vertical exaggeration factor (0.1~1.0).
                                    Used to stretch the Z-axis to visualize subduction geometry better.
        """
        if not os.path.exists(grd_file):
            print(f"File not found: {grd_file}")
            return

        import xarray as xr
        
        try:
            print(f"Loading 3D preview for {os.path.basename(grd_file)} (stride={stride})...")
            ds = xr.open_dataset(grd_file)
            z_data = ds['z']
            
            # 1. Data Prep (Downsampling)
            # Use slicing [::stride] to reduce data volume
            x = ds['x'].values[::stride]
            y = ds['y'].values[::stride]
            z = z_data.values[::stride, ::stride]
            
            # Generate meshgrid
            X, Y = np.meshgrid(x, y)
            
            # 2. Plot Setup
            fig = plt.figure(figsize=(12, 8))
            ax = fig.add_subplot(111, projection='3d')
            
            # 3. Plot Surface
            # rstride/cstride further controls grid sparsity during rendering
            surf = ax.plot_surface(X, Y, z, cmap='viridis_r', 
                                   rstride=1, cstride=1, 
                                   linewidth=0, antialiased=False, alpha=0.8)
            
            # 4. Add Bottom Projection
            # Plot contours at the bottom of Z-axis for map view reference
            min_z = np.nanmin(z)
            ax.contourf(X, Y, z, zdir='z', offset=min_z - 100, cmap='viridis_r', alpha=0.4)
            
            # 5. Axis and View
            ax.set_xlabel('Longitude')
            ax.set_ylabel('Latitude')
            ax.set_zlabel('Depth (km)')
            ax.set_title(f"3D Inspection: {os.path.basename(grd_file)}")
            
            # Set Z limits (leave space for projection)
            ax.set_zlim(min_z - 100, np.nanmax(z))
            
            # Adjust Box Aspect (Vertical Exaggeration)
            # Matplotlib normalizes by default, manually stretch Z here
            ax.set_box_aspect((1, 1, z_exaggeration)) 
            
            # Add Colorbar
            fig.colorbar(surf, ax=ax, shrink=0.6, aspect=15, label='Depth (km)')
            
            print("Displaying 3D plot... You can rotate and zoom.")
            plt.show()

        except Exception as e:
            print(f"Error inspecting 3D: {e}")

    def inspect_shallow_trench_structure(self, grd_file, bbox=None, 
                                       shallow_cutoff=60.0, fine_step=5.0,
                                       stride=5):
        """
        [Utility] High-performance inspection of shallow trench structure.
        
        Args:
            grd_file (str): Path to .grd file.
            bbox (tuple): (lon_min, lon_max, lat_min, lat_max). Strongly recommended to focus view.
            shallow_cutoff (float): Shallow depth limit to focus on (km).
            fine_step (float): Contour interval for shallow region (km).
            stride (int): [Optimization] Downsampling stride.
                          Default 5 (take 1 point every 5).
                          Increase to 10 for speed, decrease to 1 for max precision.
        """
        if not os.path.exists(grd_file):
            print(f"File not found: {grd_file}")
            return

        import xarray as xr
        
        try:
            print(f"Inspecting trench (stride={stride})...")
            ds = xr.open_dataset(grd_file)
            
            # 1. Spatial Cropping (BBox) - First layer acceleration
            if bbox:
                # Note: Check if Slab2 longitudes are 0-360 or -180-180 to match bbox
                ds = ds.sel(x=slice(bbox[0], bbox[1]), y=slice(bbox[2], bbox[3]))
            
            # 2. Downsampling (Striding) - Second layer acceleration (Critical!)
            # Read only partial data to reduce memory/compute load
            x = ds['x'].values[::stride]
            y = ds['y'].values[::stride]
            z = ds['z'].values[::stride, ::stride]
            
            # Stats
            min_z = np.nanmin(z) # Slab2 depth is negative
            max_z = np.nanmax(z)
            
            # 3. Plotting
            fig, ax = plt.subplots(figsize=(12, 8))
            
            # A. Background Heatmap (Quick Overview)
            # Use shading='nearest' to avoid interpolation overhead
            mesh = ax.pcolormesh(x, y, z, cmap='Greys', alpha=0.2, shading='auto')
            
            # B. Deep Background Lines (> shallow_cutoff) - Minimalist
            # Only draw a few reference lines (e.g., every 20km)
            deep_levels = np.arange(np.floor(min_z/20)*20, -shallow_cutoff, 20)
            if len(deep_levels) > 0:
                ax.contour(x, y, z, levels=deep_levels, 
                           colors='gray', linestyles=':', linewidths=0.5, alpha=0.5)

            # C. Shallow Fine Lines (< shallow_cutoff) - Focus Area
            # Construct levels: from -60 to 0, every fine_step
            start_shallow = -abs(shallow_cutoff)
            shallow_levels = np.arange(start_shallow, 0.1, fine_step)
            
            # Filter levels outside data range
            valid_levels = [l for l in shallow_levels if l >= min_z and l <= max_z]
            
            if len(valid_levels) > 0:
                # Use vivid colors (Jet/Turbo/Autumn)
                cs = ax.contour(x, y, z, levels=valid_levels, 
                                cmap='jet_r', linewidths=1.2)
                # Labels
                ax.clabel(cs, inline=True, fontsize=9, fmt='%.0f', colors='black')
                
                # [New] Highlight shallowest data edge
                # Extract the shallowest contour and thicken it for easy identification
                shallowest_lvl = max(valid_levels)
                ax.contour(x, y, z, levels=[shallowest_lvl], 
                           colors='red', linewidths=2.5, linestyles='-')

            # Decoration
            ax.set_title(f"Trench Inspection (Stride={stride})\n"
                         f"Shallowest Data found: ~{max_z:.1f} km")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.axis('equal')
            
            # Print Advice
            print("="*40)
            print(f"Data range in view: {min_z:.1f} km ~ {max_z:.1f} km")
            print(f"Red thick line indicates the shallowest contour at {max(valid_levels):.0f} km")
            print("="*40)
            
            plt.show()

        except Exception as e:
            print(f"Error inspecting trench: {e}")

    def resample_layers(self, interval_km=2.0):
        """Uniformly resample all internal layers."""
        if self.verbose: print(f"Resampling all layers to {interval_km} km interval...")
        
        for d in list(self.layers.keys()):
            layer = self.layers[d]
            # Extract XY for resampling calculation
            xy = layer[:, 0:2]
            
            # Calculate cumulative distance
            dists = np.sqrt(np.diff(xy[:,0])**2 + np.diff(xy[:,1])**2)
            cum_dist = np.concatenate(([0], np.cumsum(dists)))
            total_len = cum_dist[-1]
            
            # Interpolation
            num_points = int(total_len / interval_km) + 1
            new_dists = np.linspace(0, total_len, num_points)
            
            new_x = np.interp(new_dists, cum_dist, layer[:, 0])
            new_y = np.interp(new_dists, cum_dist, layer[:, 1])
            new_z = np.full(len(new_x), d)
            
            # Update Lon/Lat
            new_lon, new_lat = self.xy2ll(new_x, new_y)
            
            self.layers[d] = np.column_stack([new_x, new_y, new_z, new_lon, new_lat])

    # =================================================================
    # 2. Geometric Operations: Universal Extrapolation (Extrapolate)
    # =================================================================

    def extrapolate_layer(self, shallow_depth, deep_depth, target_depth, 
                          calc_interval=1.0, output_interval=None, 
                          in_place=False):
        """
        Universal layer extrapolation: Linearly extrapolate based on two existing layers 
        to generate a new target depth layer.
        
        Args:
            shallow_depth (float): Depth of the shallower reference layer.
            deep_depth (float): Depth of the deeper reference layer.
            target_depth (float): Desired target depth to generate.
            calc_interval (float): High-precision resampling interval for calculation (km). Default 1.0.
            output_interval (float): Downsampling interval for output (km). Default None (no downsampling).
            in_place (bool): Whether to add the result directly to self.layers.
                             Default False (returns data only, no internal modification).

        Returns:
            np.ndarray: Array containing full info (N, 5) -> [x, y, z, lon, lat]
        """
        # 1. Get source data
        layer_s = self.layers.get(shallow_depth)
        layer_d = self.layers.get(deep_depth)
        
        if layer_s is None or layer_d is None:
            raise ValueError(f"Missing reference layers: Need {shallow_depth}km and {deep_depth}km")

        # 2. Prepare high-density points for calculation (XY space)
        interval = calc_interval if calc_interval is not None else 1.0
        p_shallow = self._resample_path_xy(layer_s[:, 0:2], interval)
        p_deep_ref = self._resample_path_xy(layer_d[:, 0:2], interval)

        # 3. Find corresponding vectors (KDTree Matching)
        tree = cKDTree(p_deep_ref)
        dists, indices = tree.query(p_shallow)
        p_matched = p_deep_ref[indices]
        
        # 4. Calculate extrapolation ratio
        dz_segment = shallow_depth - deep_depth
        dz_target = target_depth - shallow_depth
        
        if dz_segment == 0:
            raise ValueError("Reference layers have same depth, cannot calculate gradient.")
            
        ratio = dz_target / dz_segment 
        
        # 5. Execute vector calculation (XY space)
        vec = p_shallow - p_matched
        p_target_dense = p_shallow + vec * ratio
        
        # 6. Output resampling (Downsampling)
        if output_interval is not None and output_interval > interval:
            p_target_xy = self._resample_path_xy(p_target_dense, output_interval)
            if self.verbose:
                print(f"  Downsampling extrapolated layer to {output_interval} km")
        else:
            p_target_xy = p_target_dense
            
        # 7. Construct full data format [x, y, z, lon, lat]
        n_points = len(p_target_xy)
        new_x, new_y = p_target_xy[:, 0], p_target_xy[:, 1]
        new_lon, new_lat = self.xy2ll(new_x, new_y)
        new_z = np.full(n_points, abs(target_depth))
        
        full_layer_data = np.column_stack([new_x, new_y, new_z, new_lon, new_lat])
        
        # 8. Modify in-place if requested
        if in_place:
            self.layers[abs(target_depth)] = full_layer_data
            if self.verbose:
                print(f"Extrapolated layer at {target_depth} km added to engine (Ratio={ratio:.2f})")
        
        return full_layer_data

    def generate_surface_trace(self, shallow_depth=20.0, deep_depth=40.0, 
                               calc_interval=1.0, output_interval=5.0,
                               in_place=False):
        """
        Shortcut to generate surface trace (0km).
        """
        if self.verbose: print(f"Generating surface trace (0 km)...")
        
        return self.extrapolate_layer(
            shallow_depth, deep_depth, target_depth=0.0, 
            calc_interval=calc_interval,
            output_interval=output_interval,
            in_place=in_place
        )

    # ---------------------------------------------------------------
    # 2.1 Helper: Temporary resampling for XY arrays only
    # ---------------------------------------------------------------
    def _resample_path_xy(self, xy_coords, interval):
        """
        Internal tool: Resample (N, 2) XY array.
        Does not involve Lon/Lat conversion, used only for intermediate geometric calculations.
        """
        if len(xy_coords) < 2: return xy_coords
        
        dx = np.diff(xy_coords[:, 0])
        dy = np.diff(xy_coords[:, 1])
        dists = np.sqrt(dx**2 + dy**2)
        cum_dist = np.concatenate(([0], np.cumsum(dists)))
        
        total_len = cum_dist[-1]
        num_points = int(total_len / interval) + 1
        
        new_dists = np.linspace(0, total_len, num_points)
        new_x = np.interp(new_dists, cum_dist, xy_coords[:, 0])
        new_y = np.interp(new_dists, cum_dist, xy_coords[:, 1])
        
        return np.column_stack([new_x, new_y])

    # =================================================================
    # 3. Geometric Operations: Advanced Spatial Filter (Curve Projection)
    # =================================================================

    def apply_spatial_filter(self, bbox_ll, buffer_km=200.0, ref_depth=0.0, 
                             calc_interval=1.0, output_interval=None):
        """
        Curvilinear Spatial Filter based on curve projection.
        
        Principle:
        Project deep layers into the curvilinear coordinate system (s, d) of the reference layer.
        - s (Arc Length): Along-strike coordinate, used for trimming ends.
        - d (Distance): Normal distance, used for buffer filtering.
        This is more robust for curved faults (e.g., arcuate subduction zones) than simple dot products.

        Args:
            bbox_ll: [lon_min, lon_max, lat_min, lat_max] Bounding box for ref layer (0km).
            buffer_km: Max distance perpendicular to ref trace (km).
            ref_depth: Reference layer depth (usually 0.0).
            calc_interval: (float) High-precision resampling interval for calculation (km). Default 1.0.
            output_interval: (float, optional) Output resampling interval (km). Default None.
        """
        if ref_depth not in self.layers:
            raise ValueError(f"Reference layer {ref_depth}km not found")

        if self.verbose:
            print(f"Applying spatial filter (Buffer: {buffer_km}km)...")

        # 1. [Pre-calc] Upsampling
        if calc_interval is not None:
            if self.verbose: print(f"  Upsampling for calculation (Interval: {calc_interval} km)")
            self.resample_layers(calc_interval)

        # 2. Crop Reference Layer (Ref Layer) - Using BBox
        ref_layer = self.layers[ref_depth]
        lon_min, lon_max, lat_min, lat_max = bbox_ll
        lons, lats = ref_layer[:, 3], ref_layer[:, 4]
        
        mask_bbox = (lons >= lon_min) & (lons <= lon_max) & \
                    (lats >= lat_min) & (lats <= lat_max)
        
        cropped_ref = ref_layer[mask_bbox]
        if len(cropped_ref) < 2:
            raise ValueError("Reference layer has insufficient points after BBox cropping.")
            
        self.layers[ref_depth] = cropped_ref
        
        # 3. Build Reference Curvilinear System
        ref_xy = cropped_ref[:, 0:2]
        
        # Build KDTree for fast initial screening
        ref_tree = cKDTree(ref_xy)
        
        # Calculate cumulative arc length s_ref
        dists_seg = np.sqrt(np.sum(np.diff(ref_xy, axis=0)**2, axis=1))
        s_ref = np.concatenate(([0], np.cumsum(dists_seg)))
        total_len = s_ref[-1] 

        if self.verbose:
            print(f"  Ref layer clipped: {len(cropped_ref)} points (Length: {total_len:.1f} km)")

        # 4. Filter other layers (Deep Layers)
        keys_to_process = [k for k in self.layers.keys() if k != ref_depth]
        
        for d in keys_to_process:
            target = self.layers[d]
            target_xy = target[:, 0:2]
            
            # --- A. Coarse Filter (Fast rejection) ---
            dists_raw, idxs = ref_tree.query(target_xy)
            
            # --- B. Precise Projection (Calculate arc length s) ---
            s_proj = np.zeros(len(target))
            
            for i in range(len(target)):
                k = idxs[i] # Nearest Ref point index
                P = target_xy[i]
                
                # Determine local tangent
                if k < len(ref_xy) - 1:
                    R_base = ref_xy[k]
                    vec_t = ref_xy[k+1] - R_base 
                    seg_len = dists_seg[k]       
                    base_s = s_ref[k]            
                else:
                    R_base = ref_xy[k-1]
                    vec_t = ref_xy[k] - R_base
                    seg_len = dists_seg[k-1]
                    base_s = s_ref[k-1]
                
                # Vector projection
                vec_pr = P - R_base
                
                if seg_len > 1e-6:
                    proj_ratio = np.dot(vec_pr, vec_t) / (seg_len**2)
                    local_s = proj_ratio * seg_len
                else:
                    local_s = 0
                
                s_proj[i] = base_s + local_s

            # --- C. Apply Filter Conditions ---
            tolerance = 2.0 # 2km tolerance
            
            mask = (dists_raw <= buffer_km) & \
                   (s_proj >= -tolerance) & \
                   (s_proj <= total_len + tolerance)
            
            # --- D. Update Layer ---
            if np.sum(mask) > 1:
                self.layers[d] = target[mask]
            else:
                if self.verbose: print(f"    Layer {d}km empty after filter, removed.")
                del self.layers[d]

        # 5. [Post-calc] Downsampling
        if output_interval is not None and output_interval > calc_interval:
            if self.verbose: 
                print(f"  Downsampling output layers to {output_interval} km...")
            self.resample_layers(output_interval)

    # =================================================================
    # 4. Data Getters
    # =================================================================

    def get_layer_coords(self, target_depth, coords_type='ll', 
                         calc_interval=None, output_interval=None):
        """
        Get coordinates for a specific layer. Supports dynamic calculation.
        
        Args:
            target_depth (float): Target depth (km).
            coords_type (str): 'll' or 'xy'.
            calc_interval (float, optional): Calculation interval for extrapolation.
            output_interval (float, optional): Output resampling interval.

        Returns:
            np.ndarray: (N, 3) array [lon, lat, z] or [x, y, z].
        """
        target_depth = abs(target_depth)
        
        raw_data = None
        
        # 1. Try to find existing layer
        if target_depth in self.layers:
            raw_data = self.layers[target_depth]
        else:
            # Fuzzy match
            for d in self.layers.keys():
                if abs(d - target_depth) < 1e-5:
                    raw_data = self.layers[d]
                    break
        
        # 2. Dynamic Extrapolation/Interpolation if not found
        if raw_data is None:
            available_depths = sorted(self.layers.keys())
            if len(available_depths) < 2:
                print(f"Error: Not enough layers ({len(available_depths)}) to infer depth {target_depth} km")
                return None
            
            import bisect
            idx = bisect.bisect_left(available_depths, target_depth)
            
            if idx == 0:
                d1, d2 = available_depths[0], available_depths[1]
                mode_str = "Extrapolating (Up)"
            elif idx == len(available_depths):
                d1, d2 = available_depths[-2], available_depths[-1]
                mode_str = "Extrapolating (Down)"
            else:
                d1, d2 = available_depths[idx-1], available_depths[idx]
                mode_str = "Interpolating"
                
            if self.verbose:
                print(f"Notice: Layer {target_depth}km not found. {mode_str} from {d1}km and {d2}km...")
            
            c_interval = calc_interval if calc_interval is not None else 1.0
            
            raw_data = self.extrapolate_layer(
                shallow_depth=d1, deep_depth=d2, target_depth=target_depth,
                calc_interval=c_interval, 
                output_interval=None, # Keep full precision
                in_place=False
            )

        # 3. Output Resampling
        final_xy = raw_data[:, 0:2]
        
        if output_interval is not None:
            final_xy = self._resample_path_xy(final_xy, output_interval)

        # 4. Format Output
        n_pts = len(final_xy)
        z_col = np.full(n_pts, target_depth)
        
        if coords_type == 'xy':
            return np.column_stack([final_xy[:, 0], final_xy[:, 1], z_col])
        elif coords_type == 'll':
            new_lon, new_lat = self.xy2ll(final_xy[:, 0], final_xy[:, 1])
            return np.column_stack([new_lon, new_lat, z_col])
        else:
            raise ValueError("coords_type must be 'll' or 'xy'")

    def extract_depth_dip_profiles(self, num_profiles=5, profile_locations=None, 
                                   location_type='ll'):
        """
        Extract Depth-Dip Profiles.
        Essential for building Adaptive Rectangular Faults.
        
        Functionality:
        1. Auto-Uniform: If no locations specified, sample uniformly along strike.
        2. Specified: Sample at nearest points on Top Layer to specified locations.
        
        Args:
            num_profiles (int): Number of profiles (if profile_locations is None).
            profile_locations (list/array): Specified start points (N, 2).
            location_type (str): 'll' or 'xy'.

        Returns:
            dict: {
                'reference_nodes': [[lon, lat], ...], 
                'depth_dip_profiles': [np.array([[depth, dip], ...]), ...]
            }
        """
        # 1. Prepare sorted layers
        sorted_depths = sorted(self.layers.keys())
        if len(sorted_depths) < 2:
            raise ValueError("Need at least 2 layers to extract profiles.")
            
        sorted_layers = [self.layers[d] for d in sorted_depths]
        top_layer = sorted_layers[0]
        
        # 2. Determine indices
        target_indices = []
        
        if profile_locations is not None:
            locs = np.array(profile_locations)
            if location_type == 'll':
                user_x, user_y = self.ll2xy(locs[:, 0], locs[:, 1])
                user_xy = np.column_stack([user_x, user_y])
            else:
                user_xy = locs[:, 0:2]
            
            top_xy = top_layer[:, 0:2]
            tree = cKDTree(top_xy)
            _, idxs = tree.query(user_xy)
            target_indices = sorted(list(idxs))
            
            if self.verbose:
                print(f"Extracting {len(target_indices)} profiles at user-specified locations...")
                
        else:
            # Uniform sampling
            dx = np.diff(top_layer[:, 0])
            dy = np.diff(top_layer[:, 1])
            dists = np.sqrt(dx**2 + dy**2)
            cum_dist = np.concatenate(([0], np.cumsum(dists)))
            total_len = cum_dist[-1]

            sample_dists = np.linspace(0, total_len, num_profiles)
            
            for d in sample_dists:
                idx = (np.abs(cum_dist - d)).argmin()
                target_indices.append(idx)
            
            target_indices = sorted(list(set(target_indices)))
            
            if self.verbose:
                print(f"Extracting {len(target_indices)} profiles uniformly along strike...")

        return self._extract_profiles_internal(sorted_layers, indices=target_indices)

    # =================================================================
    # 5. Inverse Extraction (Contours from Fault Mesh)
    # =================================================================

    def extract_contours_from_fault(self, fault_obj, target_depths, 
                                    update_engine=False, min_len=5,
                                    sort_by=None, reverse=False,
                                    merge_tol=0.1, subdivision=3):
        """
        Extract precise depth contours from an existing Fault object.
        
        [Optimization for Rectangular Mesh] Uses "Bilinear Subdivision":
        Subdivides coarse rectangular patches into dense micro-grids (subdivision^2)
        before contour extraction. This perfectly reconstructs the hyperbolic paraboloid
        surface and eliminates depth drift caused by linear approximation errors.
        
        Args:
            subdivision (int): Subdivision level.
                               1 = Keep original (Not recommended).
                               3 = Split rect into 3x3=9 points (Recommended, fast & smooth).
                               5 = High precision.
            merge_tol (float): Vertex welding tolerance (km). Use to weld adjacent patches.
        """
        if self.verbose:
            print(f"Extracting contours (subdiv={subdivision}, merge={merge_tol}km)...")
        
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        
        x, y, z = None, None, None
        triangles = None 

        # -----------------------------------------------------------
        # 1. Parsing & Subdivision
        # -----------------------------------------------------------
        
        # --- A: Triangular Mesh (Use directly) ---
        if hasattr(fault_obj, 'Vertices') and hasattr(fault_obj, 'Faces'):
            pts = fault_obj.Vertices
            x, y, z = pts[:, 0], pts[:, 1], pts[:, 2]
            triangles = fault_obj.Faces.astype(int)

        # --- B: Rectangular Mesh (Bilinear Subdivision) ---
        else:
            # 1. Collect Patches
            raw_patches = [] 
            for p in fault_obj.patch:
                if len(p) == 4: raw_patches.append(p)
            
            if not raw_patches:
                return {}

            patches = np.array(raw_patches) 
            n_patches = len(patches)
            
            # 2. Vectorized Bilinear Interpolation
            if subdivision > 1:
                # Generate local coords u, v
                u = np.linspace(0, 1, subdivision)
                v = np.linspace(0, 1, subdivision)
                U, V = np.meshgrid(u, v)
                U = U.flatten()
                V = V.flatten()
                
                # Patch corners
                P0 = patches[:, 0][:, None, :] 
                P1 = patches[:, 1][:, None, :]
                P2 = patches[:, 2][:, None, :]
                P3 = patches[:, 3][:, None, :]
                
                # Weights
                W0 = ((1-U) * (1-V))[None, :, None]
                W1 = (U * (1-V))[None, :, None]
                W2 = (U * V)[None, :, None]
                W3 = ((1-U) * V)[None, :, None]
                
                # Interpolate
                points_interp = W0*P0 + W1*P1 + W2*P2 + W3*P3
                flat_verts = points_interp.reshape(-1, 3)
                
                # Generate local topology
                rows = subdivision
                cols = subdivision
                local_tris = []
                for r in range(rows - 1):
                    for c in range(cols - 1):
                        i0 = r * cols + c
                        i1 = r * cols + (c + 1)
                        i2 = (r + 1) * cols + (c + 1)
                        i3 = (r + 1) * cols + c
                        local_tris.append([i0, i1, i2])
                        local_tris.append([i0, i2, i3])
                
                local_tris = np.array(local_tris)
                num_points_per_patch = rows * cols
                offsets = np.arange(n_patches) * num_points_per_patch
                all_tris = local_tris[None, :, :] + offsets[:, None, None]
                triangles_raw = all_tris.reshape(-1, 3)
                
            else:
                flat_verts = patches.reshape(-1, 3)
                # Fallback to simple subdivision=2
                return self.extract_contours_from_fault(fault_obj, target_depths, 
                                                      subdivision=2, 
                                                      update_engine=update_engine,
                                                      merge_tol=merge_tol, sort_by=sort_by)

            # -------------------------------------------------------
            # 2. Vertex Welding
            # -------------------------------------------------------
            flat_verts_rounded = np.round(flat_verts, decimals=5)
            unique_v, inv_idx = np.unique(flat_verts_rounded, axis=0, return_inverse=True)
            
            if merge_tol > 0:
                tree = cKDTree(unique_v)
                pairs = tree.query_pairs(r=merge_tol)
                
                if len(pairs) > 0:
                    rows, cols = np.array(list(pairs))[:, 0], np.array(list(pairs))[:, 1]
                    N = len(unique_v)
                    graph = coo_matrix((np.ones(len(pairs)), (rows, cols)), shape=(N, N))
                    n_comp, labels = connected_components(graph, directed=False)
                    
                    final_indices = labels[inv_idx]
                    
                    # Compute averaged centroids for welded groups (using pandas for speed)
                    import pandas as pd
                    df = pd.DataFrame(unique_v, columns=['x','y','z'])
                    df['label'] = labels
                    grouped = df.groupby('label').mean()
                    x = grouped['x'].values
                    y = grouped['y'].values
                    z = grouped['z'].values
                    triangles = final_indices[triangles_raw]
                    
                else:
                    x, y, z = unique_v[:, 0], unique_v[:, 1], unique_v[:, 2]
                    triangles = inv_idx[triangles_raw]
            else:
                x, y, z = unique_v[:, 0], unique_v[:, 1], unique_v[:, 2]
                triangles = inv_idx[triangles_raw]

        z = np.abs(z)

        # -----------------------------------------------------------
        # 3. Tricontour Extraction
        # -----------------------------------------------------------
        triang = mtri.Triangulation(x, y, triangles=triangles)
        
        fig, ax = plt.subplots()
        sorted_targets = sorted(target_depths)
        extracted_layers = {}
        
        try:
            cs = ax.tricontour(triang, z, levels=sorted_targets)
            
            for i, level in enumerate(cs.levels):
                paths = cs.collections[i].get_paths()
                if not paths: continue
                
                segments = [p.vertices for p in paths if len(p.vertices) >= min_len]
                if not segments: continue
                
                segments.sort(key=len, reverse=True)
                final_xy = segments[0]
                
                if sort_by is not None:
                    sx, sy = final_xy[:, 0], final_xy[:, 1]
                    slon, slat = self.xy2ll(sx, sy)
                    if sort_by == 'x': sort_val = sx
                    elif sort_by == 'y': sort_val = sy
                    elif sort_by == 'lon': sort_val = slon
                    elif sort_by == 'lat': sort_val = slat
                    else: sort_val = None
                    if sort_val is not None:
                        idx = np.argsort(sort_val)
                        if reverse: idx = idx[::-1]
                        final_xy = final_xy[idx]

                n_pts = len(final_xy)
                lons, lats = self.xy2ll(final_xy[:, 0], final_xy[:, 1])
                z_vals = np.full(n_pts, level)
                layer_data = np.column_stack([final_xy[:, 0], final_xy[:, 1], z_vals, lons, lats])
                extracted_layers[level] = layer_data
                
        except Exception as e:
            print(f"Error extracting contour {level}km: {e}")
        finally:
            plt.close(fig)
            
        if update_engine:
            for d, data in extracted_layers.items():
                self.layers[d] = data
            if self.verbose:
                print(f"Updated engine with {len(extracted_layers)} extracted layers.")
                
        return extracted_layers

    # =================================================================
    # 6. Factory Methods
    # =================================================================

    def build_rectangular_model(self, name, total_width=20.0, numz=None, mesh_len=15.0, 
                                num_profiles=5):
        """
        Build an Adaptive Rectangular Fault Model (Rectangular Mesh).

        Uses extracted Dip Profiles to control fault geometry vs depth.

        Args:
            name (str): Fault model name.
            total_width (float): Total down-dip width (km).
            numz (int, optional): Number of down-dip patches.
                                  - None (Default): Use depth levels from engine.layers.
                                  - Integer: Force uniform division into numz segments.
            mesh_len (float): Along-strike patch length (km).
            num_profiles (int): Number of profiles to extract for geometry control.

        Returns:
            csi.AdaptiveLayeredDipRectangularPatches: Initialized fault object.
        """
        if AdaptiveLayeredDipRectangularPatches is None:
            raise ImportError("AdaptiveLayeredDipRectangularPatches not available.")

        print(f"\n[Factory] Building Rectangular Fault: {name}")
        
        rect_fault = AdaptiveLayeredDipRectangularPatches(name, lon0=self.lon0, lat0=self.lat0)
        
        prof_data = self.extract_depth_dip_profiles(num_profiles=num_profiles)
        
        top_key = sorted(self.layers.keys())[0]
        trace_data = self.layers[top_key]
            
        rect_fault.trace(trace_data[:, 3], trace_data[:, 4])
        rect_fault.set_depth_dip_from_profiles(prof_data)
        
        if numz is None:
            z_levels = sorted(self.layers.keys())
            rect_fault.z_patches = z_levels
            print(f"  Using defined Z levels: {z_levels}")
            rect_fault.buildPatches(width=total_width, numz=len(z_levels)-1, every=mesh_len)
        else:
            rect_fault.buildPatches(width=total_width, numz=numz, every=mesh_len)
            
        return rect_fault

    def build_triangular_model(self, name, 
                               field_size_dict={'min_dx': 10, 'bias': 1.1},
                               top_size=10.0, bottom_size=20.0,
                               sparse_factor=0.5):
        """
        Build an Adaptive Triangular Fault Model (Unstructured Mesh).

        Uses sorted layers stored in engine to generate mesh via Gmsh.

        Args:
            field_size_dict (dict): Mesh density parameters for PyGMSH.
            top_size (float): Target mesh size at top edge (km).
            bottom_size (float): Target mesh size at bottom edge (km).
            sparse_factor (float): Point cloud downsampling factor (0.0 ~ 1.0).

        Returns:
            csi.AdaptiveLayeredDipTriangularPatches: Initialized triangular fault object.
        """
        if AdaptiveLayeredDipTriangularPatches is None:
            raise ImportError("AdaptiveLayeredDipTriangularPatches not available.")
            
        if len(self.layers) < 2:
            raise ValueError("Insufficient layers. Need at least 2.")

        print(f"\n[Factory] Building Triangular Fault: {name}")
        
        tri_fault = AdaptiveLayeredDipTriangularPatches(name, lon0=self.lon0, lat0=self.lat0)
        
        top_key = sorted(self.layers.keys())[0]
        bottom_key = sorted(self.layers.keys())[-1]
        top_layer = self.layers[top_key][:, [3, 4, 2]]    
        bottom_layer = self.layers[bottom_key][:, [3, 4, 2]] 
        
        tri_fault.top = top_layer[0, 2]
        tri_fault.depth = bottom_layer[0, 2]
        
        tri_fault.set_top_coords(top_layer)       
        tri_fault.set_bottom_coords(bottom_layer)
        
        inter_layer_keys = sorted(self.layers.keys())[1:-1]
        intermediate = [self.layers[k][:, [3, 4, 2]] for k in inter_layer_keys]
        if intermediate:
            tri_fault.set_layer_coords(intermediate, lonlat=True)
            print(f"  Added {len(intermediate)} intermediate layers.")
            
        print("  Running Gmsh generation...")
        tri_fault.generate_multilayer_mesh(
            None, 
            nodes_on_layers=True,
            field_size_dict=field_size_dict,
            mesh_func=True, 
            verbose=False,
            top_size=top_size, 
            bottom_size=bottom_size,
            remove_entities=True, 
            sparse_points=True, 
            sparse_factor=sparse_factor, 
            occ_method='filling',
            show=False
        )
        
        return tri_fault

    # =================================================================
    # 7. Visualization Utils
    # =================================================================

    def plot_layers(self, ax=None, show=True, title="Fault Layers Geometry", 
                    figsize=None, style=['science', 'no-latex'], 
                    pdf_fonttype=42, style_opts=None, 
                    marker_size=2, **plot_kwargs):
        """
        Plot all stored depth contour layers. 
        Integrated with sci_plot_style for publication-quality figures.
        """
        if not self.layers: return

        s_opts = style_opts if style_opts else {}
        
        with sci_plot_style(style=style, figsize=figsize, pdf_fonttype=pdf_fonttype, **s_opts):
            if ax is None:
                fig, ax = plt.subplots()
            
            depths = sorted(self.layers.keys())
            norm = mcolors.Normalize(vmin=min(depths), vmax=max(depths))
            cmap = plt.get_cmap('viridis_r') 

            for d in depths:
                data = self.layers[d]
                lons, lats = data[:, 3], data[:, 4]
                color = cmap(norm(d))
                
                ax.plot(lons, lats, '.-', markersize=marker_size, color=color, 
                        label=f"{d:.0f} km", **plot_kwargs)
                ax.text(lons[0], lats[0], f"{d:.0f}", color=color, 
                        fontsize='small', fontweight='bold', ha='right', va='center')

            set_degree_formatter(ax, axis='both')
            ax.set_title(title)
            ax.axis('equal') 
            if len(depths) <= 10: ax.legend(loc='best')
            if show: plt.show()
            return ax

    def plot_dip_profiles(self, profile_data, ax=None, show=True, 
                          figsize=None, style=['science', 'no-latex'],
                          pdf_fonttype=42, style_opts=None, **plot_kwargs):
        """Plot depth-dip profiles."""
        profiles = profile_data.get('depth_dip_profiles', [])
        s_opts = style_opts if style_opts else {}

        with sci_plot_style(style=style, figsize=figsize, pdf_fonttype=pdf_fonttype, **s_opts):
            if ax is None: fig, ax = plt.subplots()

            for i, prof in enumerate(profiles):
                ax.plot(prof[:, 1], prof[:, 0], 'o-', label=f"Prof {i}", **plot_kwargs)
            
            ax.invert_yaxis() 
            ax.set_xlabel(r"Dip Angle ($^\circ$)") 
            ax.set_ylabel("Depth (km)")
            ax.set_title("Depth vs. Dip Profiles")
            if len(profiles) <= 10: ax.legend(fontsize='small')
            if show: plt.show()
            return ax

    def plot_profile_locations(self, profile_data, ax=None, show=True, 
                               figsize=None, style=['science', 'no-latex'],
                               pdf_fonttype=42, style_opts=None, **scatter_kwargs):
        """Overlay profile start locations on map view."""
        ref_nodes = np.array(profile_data['reference_nodes'])
        s_opts = style_opts if style_opts else {}
        
        with sci_plot_style(style=style, figsize=figsize, pdf_fonttype=pdf_fonttype, **s_opts):
            if ax is None:
                ax = self.plot_layers(ax=None, show=False, title="Profile Locations",
                                      figsize=figsize, style=style, 
                                      pdf_fonttype=pdf_fonttype, style_opts=style_opts)
            
            kwargs = {'c': 'red', 's': 60, 'marker': 'x', 'label': 'Profile Starts', 'zorder': 10}
            kwargs.update(scatter_kwargs)
            
            ax.scatter(ref_nodes[:, 0], ref_nodes[:, 1], **kwargs)
            
            handles, labels = ax.get_legend_handles_labels()
            by_label = dict(zip(labels, handles))
            ax.legend(by_label.values(), by_label.keys(), loc='best')

            if show: plt.show()
            return ax

    # =================================================================
    # Private Helpers
    # =================================================================
    
    def _extract_profiles_internal(self, layers_data, indices=None, num_profiles=5):
        """Internal logic for calculating dip profiles."""
        if len(layers_data) < 2:
            raise ValueError("Need at least 2 layers.")

        top_layer = layers_data[0] 

        if indices is not None:
            sample_indices = indices
        else:
            dx = np.diff(top_layer[:, 0])
            dy = np.diff(top_layer[:, 1])
            dists = np.sqrt(dx**2 + dy**2)
            cum_dist = np.concatenate(([0], np.cumsum(dists)))
            total_len = cum_dist[-1]
            sample_dists = np.linspace(0, total_len, num_profiles)
            sample_indices = []
            for d in sample_dists:
                idx = (np.abs(cum_dist - d)).argmin()
                sample_indices.append(idx)
            sample_indices = sorted(list(set(sample_indices)))

        if self.verbose:
            print(f"Internal: Processing {len(sample_indices)} profiles...")

        ref_nodes = [] 
        profiles_list = [] 

        for idx in sample_indices:
            dip_profile = [] 
            current_pt = top_layer[idx] 
            ref_nodes.append([current_pt[3], current_pt[4]])

            p_prev = top_layer[max(0, idx - 1)]
            p_next = top_layer[min(len(top_layer) - 1, idx + 1)]
            strike_vec = p_next[0:2] - p_prev[0:2]
            norm = np.linalg.norm(strike_vec)
            if norm < 1e-6: strike_vec = np.array([0.0, 1.0])
            else: strike_vec /= norm

            for i in range(len(layers_data) - 1):
                next_layer_data = layers_data[i+1]
                nl_x, nl_y = next_layer_data[:, 0], next_layer_data[:, 1]
                nl_z = next_layer_data[0, 2] 
                
                nl_dists = np.sqrt(np.diff(nl_x)**2 + np.diff(nl_y)**2)
                nl_cum = np.concatenate(([0], np.cumsum(nl_dists)))
                dense_steps = np.arange(0, nl_cum[-1], 0.5)
                if len(dense_steps) < 2: dense_steps = np.array([0, nl_cum[-1]])
                
                dense_x = np.interp(dense_steps, nl_cum, nl_x)
                dense_y = np.interp(dense_steps, nl_cum, nl_y)
                dense_xy = np.column_stack([dense_x, dense_y])

                tree = cKDTree(dense_xy)
                _, idx_next = tree.query(current_pt[0:2])
                p_next_xy = dense_xy[idx_next]
                
                vec_v = p_next_xy - current_pt[0:2]
                normal_vec = np.array([-strike_vec[1], strike_vec[0]])
                d_h_proj = abs(np.dot(vec_v, normal_vec))
                if d_h_proj < 0.01: d_h_proj = 0.01
                
                d_z = abs(nl_z - current_pt[2])
                dip_rad = np.arctan(d_z / d_h_proj)
                dip_deg = np.degrees(dip_rad)
                
                if dip_deg < 5.0:
                    if len(dip_profile) > 0: dip_deg = dip_profile[-1][1] 
                    else: dip_deg = 10.0 
                
                dip_profile.append([abs(current_pt[2]), dip_deg])
                current_pt = np.array([p_next_xy[0], p_next_xy[1], nl_z, 0, 0])

            last_depth = abs(current_pt[2])
            last_dip = dip_profile[-1][1] if len(dip_profile) > 0 else 0
            dip_profile.append([last_depth, last_dip])
            profiles_list.append(np.array(dip_profile))

        return {
            'reference_nodes': ref_nodes, 
            'depth_dip_profiles': profiles_list
        }
    

def main():
    import numpy as np
    import matplotlib.pyplot as plt
    # =================================================================
    # 1. Configuration Parameters
    # =================================================================
    CONFIG = {
        # --- Data Source ---
        'grd_file': 'kur_slab2_dep_02.24.18.grd',
        
        # [CRITICAL] Raw Data Layers: Start from 20km to avoid noise at 10km depth.
        # The 0km trace (Trench) will be generated via extrapolation.
        'raw_depths': [15.0, 20.0, 30.0, 40.0, 60.0, 80.0], 
        
        'bbox': (156, 164, 48, 53),     # Study area BBox
        'ref_center': (160.0, 50.0),    # Projection center
        
        # --- Rectangular Mesh Parameters ---
        'rect_width': 200.0,            # Total down-dip width (km)
        'rect_len': 15.0,               # Along-strike patch length (km)
        'num_profiles': 5,              # Number of automatically extracted profiles
        
        # --- Triangular Mesh Parameters ---
        'tri_field': {'min_dx': 15, 'bias': 1.1}, # Mesh density control
        'tri_sparse': 0.5               # Point cloud sparsification factor (0~1)
    }
    
    # =================================================================
    # 2. Inspection Phase
    # =================================================================
    # Initialize Engine (Temporary center, used only for inspection)
    engine = FaultGeometryEngine(lon0=CONFIG['ref_center'][0], 
                                 lat0=CONFIG['ref_center'][1], 
                                 verbose=True)

    print("=== Step 0: Data Inspection ===")
    print("Tip: Check the popup plots to confirm the starting depth for 'raw_depths'.")
    
    # 2.1 3D Preview: Check Slab morphology and undulation
    engine.inspect_slab2_in_3d(CONFIG['grd_file'], stride=10, z_exaggeration=0.5)
    
    # 2.2 Trench Zoom: Determine where to cut off shallow data
    # If the red contour at 10km is broken, start loading from 20km.
    engine.inspect_shallow_trench_structure(
        CONFIG['grd_file'], 
        bbox=CONFIG['bbox'], 
        shallow_cutoff=30.0, 
        fine_step=2.0, 
        stride=5
    )

    # =================================================================
    # 3. Data Loading & Geometry Processing
    # =================================================================
    print("\n=== Step 1: Loading & Geometry Processing ===")

    # 3.1 Load Deep Data
    engine.load_from_slab2(CONFIG['grd_file'], CONFIG['raw_depths'], stitch_mode='lat')

    # 3.2 [Geometric Extrapolation] Generate Trench Trace (0km)
    # Extrapolate 0km using the two shallowest configured layers (e.g., 20km & 30km)
    d1, d2 = CONFIG['raw_depths'][0], CONFIG['raw_depths'][1]
    if 0.0 not in engine.layers:
        print(f"Generating Trench (0km) from {d1}km and {d2}km trends...")
        engine.generate_surface_trace(shallow_depth=d1, deep_depth=d2, 
                                      calc_interval=1.0, output_interval=2.0, in_place=True)
    
    # 3.3 [Spatial Filter] Keep data only within study area
    engine.apply_spatial_filter(bbox_ll=CONFIG['bbox'], buffer_km=250.0, 
                                ref_depth=0.0) # Reference to trench
    
    # 3.4 [Visual QC] Check final geometry used for modeling
    print("QC: Visualizing final geometry...")
    engine.plot_layers(title="Final Geometry for Modeling", figsize='single')
    
    # (Optional) Manually check profile morphology
    prof_qc = engine.extract_depth_dip_profiles(num_profiles=5)
    engine.plot_profile_locations(prof_qc)
    engine.plot_dip_profiles(prof_qc)

    # =================================================================
    # 4. Demo A: Build Rectangular Mesh
    # =================================================================
    print("\n" + "="*50)
    print("Model A: Rectangular Mesh (Structured)")
    print("Use Case: Simple slip inversions requiring regular patch alignment.")
    print("="*50)
    
    try:
        # [Factory Mode] One-click Build
        # Engine automatically: 1. Extracts profiles 2. Fits dip 3. Generates mesh
        rect_fault = engine.build_rectangular_model(
            name='Kamchatka_Rect',
            total_width=CONFIG['rect_width'],
            mesh_len=CONFIG['rect_len'],
            num_profiles=CONFIG['num_profiles'], # Let factory auto-extract 5 profiles
            numz=15  # Force 15 layers down-dip
        )
        
        # Post-processing
        if hasattr(rect_fault, 'initializeslip'):
            rect_fault.initializeslip(values='depth') 
            # rect_fault.slip[:, 0] = np.degrees(rect_fault.slip[:, 0])
            
            # Save & Plot
            rect_fault.writePatches2File('kamchatka_rect.gmt', add_slip='strikeslip')
            print("Saved: kamchatka_rect.gmt")
            rect_fault.plot(drawCoastlines=False, slip='strikeslip', equiv=True, title='Rectangular Mesh')
        
        # --- Inverse Contour Extraction & Validation ---
        my_layers_rect = engine.extract_contours_from_fault(rect_fault, 
                                                            target_depths=[10.0, 15.0, 20.0, 30.0, 40.0, 55.0],
                                                            sort_by='y', reverse=True)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        for depth, layer in my_layers_rect.items():
            ax.plot(layer[:,0], layer[:,1], 'x-', label=f'{depth} km')
        
        if hasattr(rect_fault, 'patch'):
            for quad in rect_fault.patch:
                quad = np.asarray(quad)
                # Plot X/Y projection, closed quadrilateral
                x = np.append(quad[:,0], quad[0,0])
                y = np.append(quad[:,1], quad[0,1])
                ax.plot(x, y, 'r-', lw=1, alpha=0.7)
                # Annotate depth at vertices
                for pt in quad:
                    ax.text(pt[0], pt[1], f"{pt[2]:.1f}", fontsize=7, color='b', ha='left', va='bottom')
        
        ax.legend()
        ax.set_title("Extracted Contours from Rectangular Model")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.show()
            
    except Exception as e:
        print(f"Skipping Rectangular Model: {e}")

    # =================================================================
    # 5. Demo B: Build Triangular Mesh
    # =================================================================
    print("\n" + "="*50)
    print("Model B: Triangular Mesh (Unstructured)")
    print("Use Case: Complex slab geometry requiring surface fitting.")
    print("="*50)
    
    try:
        # [Factory Mode] One-click Build
        # Engine uses internal cleaned layers to generate mesh directly
        tri_fault = engine.build_triangular_model(
            name='Kamchatka_Tri',
            field_size_dict=CONFIG['tri_field'],
            top_size=15.0,     # Top mesh fineness
            bottom_size=25.0,  # Bottom mesh coarseness
            sparse_factor=CONFIG['tri_sparse']
        )
        
        # Post-processing
        if hasattr(tri_fault, 'initializeslip'):
            # Triangular mesh usually colored by depth to check morphology
            tri_fault.initializeslip(values='depth') 
            
            # Save
            tri_fault.writePatches2File('kamchatka_tri.gmt', add_slip='strikeslip')
            print("Saved: kamchatka_tri.gmt")
            
            # Quick preview of node distribution
            tri_fault.plot(drawCoastlines=False, slip='strikeslip', title='Triangular Mesh') 

        # --- Inverse Contour Extraction & Validation ---
        my_layers_tri = engine.extract_contours_from_fault(tri_fault, 
                                                           target_depths=[10.0, 15.0, 20.0, 30.0, 40.0, 60.0],
                                                           sort_by='y', reverse=True)
        fig = plt.figure(figsize=(8,6))
        ax = fig.add_subplot(111)
        for depth, layer in my_layers_tri.items():
            ax.plot(layer[:,0], layer[:,1], 'x-', label=f'{depth} km')
        
        # === New: Plot Triangular Mesh of tri_fault ===
        if hasattr(tri_fault, 'Vertices') and hasattr(tri_fault, 'Faces'):
            verts = tri_fault.Vertices
            faces = tri_fault.Faces
            for tri in faces:
                tri = np.asarray(tri, dtype=int)
                # Get x/y of three vertices
                x = np.append(verts[tri, 0], verts[tri[0], 0])
                y = np.append(verts[tri, 1], verts[tri[0], 1])
                ax.plot(x, y, color='g', lw=0.5, alpha=0.5)
        
        ax.legend()
        ax.set_title("Extracted Contours from Rectangular Model + Triangular Mesh")
        ax.set_xlabel("X (km)")
        ax.set_ylabel("Y (km)")
        plt.show()

    except Exception as e:
        print(f"Skipping Triangular Model: {e}")

if __name__ == "__main__":
    main()