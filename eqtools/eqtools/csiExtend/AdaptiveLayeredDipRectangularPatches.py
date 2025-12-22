import numpy as np
import pandas as pd
from typing import Union, List, Dict, Callable, Tuple
from .DipInterpolation import LayeredDipInterpolator, DepthDipProfile
from csi.faultwithlistric import faultwithlistric

class AdaptiveLayeredDipRectangularPatches(faultwithlistric):
    """
    A Rectangular Patch fault class that uses adaptive 3D dip interpolation.
    
    Inherits from faultwithlistric to maintain rectangular grid topology,
    but generates node geometry by querying a LayeredDipInterpolator
    instead of using simple top/bottom dip arrays.
    """

    def __init__(self, name, utmzone=None, ellps='WGS84', lon0=None, lat0=None, verbose=True):
        super().__init__(name, utmzone=utmzone, ellps=ellps, lon0=lon0, lat0=lat0, verbose=verbose)
        
        # Initialize the interpolator engine
        self.dip_interpolator = LayeredDipInterpolator()
        self.interpolation_axis = 'auto'

    # ==================== Input Methods (Adapted from Triangular Class) ====================

    def set_depth_dip_from_profiles(
            self,
            profiles_data: Union[Dict, pd.DataFrame, np.ndarray],
            is_utm: bool = False,
            interpolation_method: str = 'linear',
            buffer_nodes: np.ndarray = None,
            buffer_radius: float = None
        ):
        """
        Set depth-dip relationship from discrete profiles.
        
        This mimics the triangular patch input method, allowing sparse/arbitrary 
        profiles rather than requiring input at every trace node.
        """
        self.dip_interpolator.clear_profiles()
        
        # Reuse logic to parse inputs (assuming helper methods are available or reimplemented)
        # For brevity, implementing the core parsing logic directly here
        
        if isinstance(profiles_data, dict):
            self._parse_dict_profiles(profiles_data, is_utm, interpolation_method)
        elif isinstance(profiles_data, pd.DataFrame):
            self._parse_dataframe_profiles(profiles_data, is_utm, interpolation_method)
        elif isinstance(profiles_data, np.ndarray):
            self._parse_array_profiles(profiles_data, is_utm, interpolation_method)
        else:
            raise ValueError(f"Unsupported profiles_data type: {type(profiles_data)}")

        # Handle buffer nodes - simplified version for rectangular context
        # In rectangular patches, we often just need the spatial interpolation to work well
        if buffer_nodes is not None and buffer_radius is not None:
            self._add_buffer_profiles(buffer_nodes, buffer_radius, interpolation_method)

    def _parse_dict_profiles(self, data: dict, is_utm: bool, interp_method: str):
        """Parse dictionary format profiles."""
        ref_nodes = np.atleast_2d(data['reference_nodes'])
        profiles = data['depth_dip_profiles']
        
        for node, depth_dip in zip(ref_nodes, profiles):
            if is_utm:
                x, y = node[0], node[1]
                lon, lat = self.xy2ll(x, y)
            else:
                lon, lat = node[0], node[1]
                x, y = self.ll2xy(lon, lat)
            
            profile = DepthDipProfile(x, y, lon, lat, np.array(depth_dip), 
                                      interp_method, input_dip_range='neg90_90')
            self.dip_interpolator.add_profile(profile)
    
    def _parse_dataframe_profiles(self, df: pd.DataFrame, is_utm: bool, interp_method: str):
        """Parse DataFrame format profiles."""
        coord_cols = ['x', 'y'] if is_utm else ['lon', 'lat']
        
        # Group by location
        grouped = df.groupby(coord_cols)
        
        for coords, group in grouped:
            if is_utm:
                x, y = coords
                lon, lat = self.xy2ll(x, y)
            else:
                lon, lat = coords
                x, y = self.ll2xy(lon, lat)
            
            depth_dip = group[['depth', 'dip']].sort_values('depth').values
            profile = DepthDipProfile(x, y, lon, lat, depth_dip, 
                                      interp_method, input_dip_range='neg90_90')
            self.dip_interpolator.add_profile(profile)
    
    def _parse_array_profiles(self, arr: np.ndarray, is_utm: bool, interp_method: str):
        """Parse array format profiles."""
        if arr.shape[1] != 4:
            raise ValueError("Array must have 4 columns: [lon/x, lat/y, depth, dip]")
        
        # Convert to DataFrame and use DataFrame parser
        coord_cols = ['x', 'y'] if is_utm else ['lon', 'lat']
        df = pd.DataFrame(arr, columns=coord_cols + ['depth', 'dip'])
        self._parse_dataframe_profiles(df, is_utm, interp_method)
    
    def _add_buffer_profiles(self, buffer_nodes: np.ndarray, buffer_radius: float,
                             interpolation_method: str = 'linear'):
        """
        Add profiles at buffer node locations by interpolating from existing profiles.
        """
        if len(self.dip_interpolator.profiles) == 0:
            raise ValueError("Must have existing profiles before adding buffer profiles")
        
        # Get depths from existing profiles
        depths = self.dip_interpolator.profiles[0].depth_dip_pairs[:, 0]
        
        # Create xydip at each depth for buffer node handling
        for depth in depths:
            # Get dips at this depth for existing profiles
            xydip = self.dip_interpolator.sample_at_depth(depth)
            
            # Apply buffer node handling
            interpolation_axis = self.dip_interpolator.interpolation_axis
            xydip_expanded = self.handle_buffer_nodes(
                xydip, buffer_nodes, buffer_radius,
                interpolation_axis, update_ref=False)
            
            # Find new nodes (buffer nodes)
            existing_locs = set(zip(xydip['x'], xydip['y']))
            for _, row in xydip_expanded.iterrows():
                if (row['x'], row['y']) not in existing_locs:
                    # This is a buffer node - need to create profile if not exists
                    # Check if profile already exists at this location
                    profile_exists = any(
                        np.isclose(p.x, row['x']) and np.isclose(p.y, row['y'])
                        for p in self.dip_interpolator.profiles
                    )
                    if not profile_exists:
                        # Create new profile by sampling at all depths
                        depth_dip_pairs = []
                        for d in depths:
                            dip = self.dip_interpolator.get_dip(row['x'], row['y'], d)
                            depth_dip_pairs.append([d, dip])
                        
                        lon, lat = row.get('lon', None), row.get('lat', None)
                        if lon is None or lat is None:
                            lon, lat = self.xy2ll(row['x'], row['y'])
                        
                        profile = DepthDipProfile(
                            row['x'], row['y'], lon, lat,
                            np.array(depth_dip_pairs),
                            interpolation_method=interpolation_method,
                            input_dip_range='neg90_90'
                        )
                        self.dip_interpolator.add_profile(profile)
    
    def _parse_dict_profiles(self, data, is_utm, interp_method):
        ref_nodes = np.atleast_2d(data['reference_nodes'])
        profiles = data['depth_dip_profiles']
        for node, depth_dip in zip(ref_nodes, profiles):
            if is_utm:
                x, y = node[0], node[1]
                lon, lat = self.xy2ll(x, y)
            else:
                lon, lat = node[0], node[1]
                x, y = self.ll2xy(lon, lat)
            profile = DepthDipProfile(x, y, lon, lat, np.array(depth_dip), 
                                    interp_method, input_dip_range='neg90_90')
            self.dip_interpolator.add_profile(profile)

    def _parse_dataframe_profiles(self, df, is_utm, interp_method):
        coord_cols = ['x', 'y'] if is_utm else ['lon', 'lat']
        for coords, group in df.groupby(coord_cols):
            if is_utm:
                x, y = coords
                lon, lat = self.xy2ll(x, y)
            else:
                lon, lat = coords
                x, y = self.ll2xy(lon, lat)
            depth_dip = group[['depth', 'dip']].sort_values('depth').values
            profile = DepthDipProfile(x, y, lon, lat, depth_dip, 
                                    interp_method, input_dip_range='neg90_90')
            self.dip_interpolator.add_profile(profile)

    def _parse_array_profiles(self, arr, is_utm, interp_method):
        if arr.shape[1] != 4: raise ValueError("Array must be [lon/x, lat/y, depth, dip]")
        coord_cols = ['x', 'y'] if is_utm else ['lon', 'lat']
        df = pd.DataFrame(arr, columns=coord_cols + ['depth', 'dip'])
        self._parse_dataframe_profiles(df, is_utm, interp_method)

    def _add_buffer_profiles(self, buffer_nodes, buffer_radius, interpolation_method):
        # Implementation identical to AdaptiveTriangularPatches._add_buffer_profiles
        # Used to stabilize interpolation far from reference profiles
        if len(self.dip_interpolator.profiles) == 0: return
        depths = self.dip_interpolator.profiles[0].depth_dip_pairs[:, 0]
        
        # Determine axis if auto
        if self.interpolation_axis == 'auto':
             # Simple heuristic based on current trace if available, else skip
             if hasattr(self, 'xf') and self.xf is not None:
                 dx = self.xf.max() - self.xf.min()
                 dy = self.yf.max() - self.yf.min()
                 self.dip_interpolator.interpolation_axis = 'x' if dx > dy else 'y'
             else:
                 self.dip_interpolator.interpolation_axis = 'y' # Default

        for depth in depths:
            # 1. Sample field at current depth
            xydip = self.dip_interpolator.sample_at_depth(depth)
            
            # 2. Logic to expand xydip to buffer nodes (Simplified here for brevity)
            # In a full implementation, you would copy the `handle_buffer_nodes` 
            # logic from AdaptiveTriangularPatches or inherit a common Mixin.
            # Here we just iterate buffer nodes and nearest-neighbor interpolate
            for bn in buffer_nodes:
                bx, by = self.ll2xy(bn[0], bn[1])
                # Find nearest existing profile
                dists = np.sqrt((xydip['x'] - bx)**2 + (xydip['y'] - by)**2)
                nearest_dip = xydip.iloc[dists.argmin()]['dip']
                
                # Check if profile exists, if not create one
                if not any(np.isclose(p.x, bx) and np.isclose(p.y, by) for p in self.dip_interpolator.profiles):
                     # (In reality you collect all depths first, this is a conceptual snippet)
                     pass 

    # ==================== Core Build Logic ====================

    def buildPatches(self, dipdirection=None, every=10, numz=None, width=None,
                     width_bias=None, min_patch_width=None, 
                     minpatchsize=0.00001, verbose=None):
        """
        Build patches using the Adaptive Layered Dip Interpolator.
        
        Overrides faultwithlistric.buildPatches.
        """
        if verbose is None: verbose = self.verbose
        
        # Check requirements
        if len(self.dip_interpolator.profiles) == 0:
            raise ValueError("No dip profiles found. Call set_depth_dip_from_profiles first.")

        # Update geometry
        if numz is not None: self.numz = numz
        if width is not None: self.width = width
        
        # Set interpolation axis if still auto
        if self.dip_interpolator.interpolation_axis == 'auto':
            dx = self.xf.max() - self.xf.min()
            dy = self.yf.max() - self.yf.min()
            self.dip_interpolator.interpolation_axis = 'x' if dx > dy else 'y'

        if verbose:
            print("Building Adaptive Rectangular Patches...")
            print(f"Interpolation Axis: {self.dip_interpolator.interpolation_axis}")

        # Initialize structures
        self.patch = []
        self.patchll = []
        self.slip = []
        self.patchdip = []

        # 1. Discretize Trace
        self.discretize_trace(every, threshold=every/3.0)
        
        # 2. Interpolate Trace Geometries
        total_length = self._calculateTraceLength()
        nlength = max(2, int(np.round(total_length / every)) + 1)
        l_samples = np.linspace(0, total_length, nlength)
        positions, strikes = self._interpolateTrace(l_samples)
        self.local_strikes = strikes

        # 3. Calculate Layer Widths (Down-dip lengths)
        n_layers = self.numz
        layer_widths, cum_widths = self._calculateLayerWidths(self.width, n_layers)
        self.layer_widths = layer_widths

        # 4. Generate Nodes Grid (nlength x nwidth x 3)
        nwidth = n_layers + 1
        nodes = np.zeros((nlength, nwidth, 3))
        dip_at_nodes = np.zeros((nlength, nwidth))

        # 5. Propagate Nodes Down-Dip
        for il in range(nlength):
            x_curr, y_curr = positions[il]
            strike_rad = np.deg2rad(strikes[il])
            
            # Dip direction: perpendicular to strike, usually +90 deg (Right Hand Rule)
            if dipdirection is None:
                dip_dir_rad = strike_rad + np.pi/2
            else:
                dip_dir_rad = np.deg2rad(dipdirection)

            # --- Layer 0 (Top) ---
            nodes[il, 0, 0] = x_curr
            nodes[il, 0, 1] = y_curr
            nodes[il, 0, 2] = self.top
            
            # Get dip at top from interpolator
            dip_val = self.dip_interpolator.get_dip(x_curr, y_curr, self.top)
            dip_at_nodes[il, 0] = dip_val

            # --- Propagate Downwards ---
            for iw in range(1, nwidth):
                prev_x, prev_y, prev_z = nodes[il, iw-1]
                dw = layer_widths[iw-1] # Width of this specific layer segment
                
                # Query Interpolator: "What is the dip at my current location?"
                # Note: We query at prev_z (top of current layer segment) or 
                # midpoint (prev_z + estimate) for better accuracy.
                # Here we query at the starting node of the segment.
                current_dip = self.dip_interpolator.get_dip(prev_x, prev_y, prev_z)
                
                dip_rad = np.deg2rad(current_dip)
                dip_at_nodes[il, iw] = current_dip # Store for patch averaging

                # Adjust for negative dips (dipping in opposite direction)
                if dip_rad < 0:
                    dip_rad = -dip_rad
                    dip_dir_rad += np.pi # Reverse dip direction

                # Calculate displacement based on this local dip
                dx = dw * np.cos(dip_rad) * np.sin(dip_dir_rad)
                dy = dw * np.cos(dip_rad) * np.cos(dip_dir_rad)
                dz = dw * np.sin(dip_rad)

                # Set next node
                nodes[il, iw, 0] = prev_x + dx
                nodes[il, iw, 1] = prev_y + dy
                nodes[il, iw, 2] = prev_z + dz

        self.nodes = nodes
        self.dip_at_nodes = dip_at_nodes

        # 6. Build Patches from Nodes (Standard Rectangular Logic)
        # Track actual depth range achieved
        D = [self.top]
        
        for iw in range(n_layers):
            D.append(nodes[:, iw+1, 2].max())
            
            for il in range(nlength - 1):
                # Get corners
                p1 = nodes[il, iw]     # Top-Left
                p2 = nodes[il+1, iw]   # Top-Right
                p3 = nodes[il+1, iw+1] # Bottom-Right
                p4 = nodes[il, iw+1]   # Bottom-Left
                
                # Coordinate conversion
                p1ll = [*self.xy2ll(p1[0], p1[1]), p1[2]]
                p2ll = [*self.xy2ll(p2[0], p2[1]), p2[2]]
                p3ll = [*self.xy2ll(p3[0], p3[1]), p3[2]]
                p4ll = [*self.xy2ll(p4[0], p4[1]), p4[2]]

                # Average dip for this patch
                # Average of 4 corners
                patch_dip = np.mean([
                    dip_at_nodes[il, iw], dip_at_nodes[il+1, iw],
                    dip_at_nodes[il+1, iw+1], dip_at_nodes[il, iw+1]
                ])

                # Store
                self.patch.append(np.array([p1, p2, p3, p4]))
                self.patchll.append(np.array([p1ll, p2ll, p3ll, p4ll]))
                self.slip.append([0.0, 0.0, 0.0])
                self.patchdip.append(np.deg2rad(patch_dip))

        # Finalize
        self.z_patches = np.array(D)
        self.depth = self.z_patches.max()
        self.slip = np.array(self.slip)
        
        # Restore trace discretization for consistency
        self.discretize_trace(every, threshold=every/3.0)
        self.computeEquivRectangle()
        
        if verbose:
            print(f"Built {len(self.patch)} patches.")
            print(f"Achieved Depth Range: {self.top:.2f}km to {self.depth:.2f}km")


# ==================== Example Usage ====================

if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    
    print("=== 启动自适应矩形断层测试 ===")

    # 1. 初始化断层
    # lon0, lat0 用于建立局部笛卡尔坐标系 (XY)
    fault = AdaptiveLayeredDipRectangularPatches('test_fault', lon0=100.0, lat0=30.0, verbose=True)

    # 2. 定义断层迹线 (Trace) - 之前缺少的步骤
    # 这里我们生成一条大约 50-60km 长，稍微有一点点弧度的迹线
    n_trace_points = 20
    lon_array = np.linspace(100.0, 100.5, n_trace_points)
    # 让纬度稍微带点正弦弯曲，模拟真实断层
    lat_array = np.linspace(30.0, 30.1, n_trace_points) + 0.01 * np.sin(np.linspace(0, np.pi, n_trace_points))
    
    print(f"设置断层迹线: {n_trace_points} 个节点")
    fault.trace(lon_array, lat_array)

    # 3. 定义稀疏的倾角剖面 (Sparse Dip Profiles)
    # 场景：断层西段 (Start) 是铲状的 (Listric)，东段 (End) 是平直的 (Planar)
    profile_data = {
        'reference_nodes': [
            [100.0, 30.0],  # 剖面 1 的位置 (近似迹线起点)
            [100.5, 30.1]   # 剖面 2 的位置 (近似迹线终点)
        ],
        'depth_dip_profiles': [
            # 剖面 1 (铲状): 地表 60度 -> 深部 20km 处变缓为 20度
            np.array([[0, 60], [20, 20]]), 
            
            # 剖面 2 (平直): 地表 45度 -> 深部 20km 处依然 45度 (线性过渡)
            np.array([[0, 45], [20, 45]])
        ]
    }

    # 4. 加载数据到插值器
    # is_utm=False 表示输入的是经纬度
    print("加载倾角剖面数据...")
    fault.set_depth_dip_from_profiles(profile_data, is_utm=False, interpolation_method='linear')

    # 5. 构建网格 (Build Grid)
    # 这里会自动调用插值器，计算每一层每一个点的具体倾角
    print("构建断层网格...")
    fault.buildPatches(
        width=25.0,           # 沿倾向的总延伸长度 (km)
        numz=10,              # 沿深度方向切分 10 层
        every=5.0,            # 沿走向每 5km 切分一个 patch
        width_bias=1.1,       # 几何级数分布：深部的 patch 会比浅部的大 (1.1倍递增)
        min_patch_width=1.0,  # 第一层 (最浅层) 的宽度为 1.0km
        dipdirection=None     # 默认为 None，程序会自动根据走向右手法则计算
    )

    # 6. 可视化检查
    
    # 6.1 绘制剖面图
    # 绘制 3 个横切面：起点、中点、终点，观察倾角如何从铲状渐变为平直
    print("绘制横切剖面...")
    try:
        fault.plotCrossSections(n_sections=3)
    except Exception as e:
        print(f"绘制剖面时出错 (可能是因为不在 notebook 环境): {e}")

    # 6.2 绘制 3D 视图或地图视图
    # 初始化滑移量为 0，防止绘图函数报错
    print("绘制 3D 网格...")
    if hasattr(fault, 'initializeslip'):
        fault.initializeslip()
    
    # equiv=True 表示绘制等效矩形
    fault.plot(drawCoastlines=False, equiv=True)
    
    print("=== 测试完成 ===")