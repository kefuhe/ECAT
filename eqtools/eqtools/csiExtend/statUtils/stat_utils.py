import numpy as np
import matplotlib.pyplot as plt
from ..BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches as TriangularPatches
import seaborn as sns
from ...plottools import sci_plot_style


class DepthStatistics:
    """
    Statistical analysis of slip distribution along fault depth direction
    """
    
    def __init__(self, source):
        """
        Initialize depth statistics analyzer
        
        Parameters:
        -----------
        source : TriangularPatches object
            Fault geometry object containing vertices, faces and slip data
        """
        self.source = source
        self.vertices = source.Vertices 
        self.faces = source.Faces
        
    def get_slip_data(self, slip='total'):
        """
        Extract slip data based on type
        
        Parameters:
        -----------
        slip : str or array
            Slip type ('total', 'strike', 'dip') or slip array
            
        Returns:
        --------
        slip_data : array
            Slip data array
        """
        if isinstance(slip, str):
            if slip == 'total':
                slip_data = np.sqrt(self.source.slip[:, 0]**2 + self.source.slip[:, 1]**2)
            elif slip == 'strike':
                slip_data = self.source.slip[:, 0]
            elif slip == 'dip':
                slip_data = self.source.slip[:, 1]
            else:
                raise ValueError("slip must be 'total', 'strike', 'dip' or array")
        else:
            slip_data = np.array(slip)
        return slip_data
    
    def calculate_patch_depths(self, slip_data):
        """
        Calculate depth for each triangular patch
        
        Parameters:
        -----------
        slip_data : array
            Slip data for each patch
            
        Returns:
        --------
        patch_depths : array
            Center depths of each patch
        patch_slips : array
            Slip values for each patch
        """
        patch_depths = []
        patch_slips = []
        
        for i, face in enumerate(self.faces):
            # Get coordinates of three vertices of the triangle
            v1, v2, v3 = self.vertices[face[0]], self.vertices[face[1]], self.vertices[face[2]]
            # Calculate center depth (absolute value of Z coordinate)
            center_depth = abs((v1[2] + v2[2] + v3[2]) / 3.0)
            patch_depths.append(center_depth)
            patch_slips.append(slip_data[i])
        
        return np.array(patch_depths), np.array(patch_slips)
    
    def compute_statistics(self, slip='total', bins=None, depth_interval=None, 
                          depth_min=None, depth_max=None, depth_edges=None, method='mean', 
                          normalize=False, norm_method='max', normalize_stats=False,
                          auto_depth_layers=False, layer_tolerance=0.1, min_layer_thickness=0.5,
                          clustering_method='simple_clustering', max_layers=30):
        """
        Compute statistical analysis of slip distribution along depth
        
        Parameters:
        -----------
        slip : str or array
            Slip type ('total', 'strike', 'dip') or slip array
        bins : int, optional
            Number of depth bins (ignored if depth_interval or depth_edges is specified)
        depth_interval : float, optional
            Depth interval size in km (ignored if depth_edges is specified)
        depth_min, depth_max : float, optional
            Depth range in km, auto-calculated if None (ignored if depth_edges is specified)
        depth_edges : array-like, optional
            Custom depth bin edges in km, e.g., [0, 2, 5, 9, 13, 18, 25]
            Takes priority over all other binning parameters
        method : str
            Statistical method ('mean', 'max', 'median')
        normalize : bool, optional
            Whether to normalize the slip values before statistics (default: False)
        norm_method : str or float, optional
            Normalization method ('max', 'sum', 'l2', 'percentile', or custom value)
        normalize_stats : bool, optional
            Whether to normalize the statistical results after computation (default: False)
        auto_depth_layers : bool, optional
            Whether to automatically detect depth layers from vertices (default: False)
        layer_tolerance : float, optional
            Tolerance for grouping vertices into layers in km (default: 0.5)
        min_layer_thickness : float, optional
            Minimum thickness between layers in km (default: 1.0)
        clustering_method : str, optional
            Method for clustering depths ('histogram', 'kmeans', 'density') (default: 'histogram')
        max_layers : int, optional
            Maximum number of layers to detect (default: 20)
            
        Returns:
        --------
        dict containing:
            - depth_centers: Center depths of each bin
            - stat_values: Statistical values for each depth bin
            - std_values: Standard deviation for each depth bin
            - patch_depths: Individual patch depths
            - patch_slips: Individual patch slips
            - depth_edges: Bin edges
            - params: Analysis parameters
            - norm_factor: Normalization factor (if normalized)
            - stats_norm_factor: Statistical normalization factor (if normalize_stats=True)
        """
        # Auto-detect depth layers if requested
        if auto_depth_layers and depth_edges is None:
            depth_edges = self._detect_depth_layers(layer_tolerance=layer_tolerance, 
                                                min_layer_thickness=min_layer_thickness,
                                                clustering_method=clustering_method,
                                                max_layers=max_layers)
            print(f"Auto-detected {len(depth_edges)-1 if len(depth_edges)>1 else 0} depth layers with boundaries: {depth_edges}")
        
        # Get slip data
        slip_data = self.get_slip_data(slip)
        
        # Calculate patch depths and slips
        patch_depths, patch_slips = self.calculate_patch_depths(slip_data)
        
        # Apply normalization if requested
        norm_factor = 1.0
        if normalize:
            patch_slips, norm_factor = self._normalize_data(patch_slips, norm_method)
        
        # Determine depth binning strategy
        custom_bins = depth_edges is not None
        
        if custom_bins:
            # Use custom depth edges
            depth_edges = np.array(depth_edges)
            if len(depth_edges) < 2:
                raise ValueError("depth_edges must contain at least 2 values")
            
            # Sort edges to ensure proper order
            depth_edges = np.sort(depth_edges)
            bins = len(depth_edges) - 1
            depth_min = depth_edges[0]
            depth_max = depth_edges[-1]
            
            print(f"Using custom depth edges: {depth_edges}")
            print(f"Number of bins: {bins}")
            
        else:
            # Use existing logic for uniform binning
            if depth_min is None:
                depth_min = np.abs(self.source.Vertices[:, 2]).min()
            if depth_max is None:
                depth_max = np.abs(self.source.Vertices[:, 2]).max()
            
            # Determine bins based on interval or bin count
            if depth_interval is not None:
                bins = int((depth_max - depth_min) / depth_interval)
                actual_interval = (depth_max - depth_min) / bins
            else:
                if bins is None:
                    bins = 10  # default value
                actual_interval = (depth_max - depth_min) / bins
            
            depth_edges = np.linspace(depth_min, depth_max, bins + 1)
        
        # ... rest of the method remains the same as before ...
        # Statistical analysis for each depth interval
        depth_centers = []
        stat_values = []
        std_values = []
        
        for i in range(bins):
            if custom_bins:
                depth_start = depth_edges[i]
                depth_end = depth_edges[i + 1]
                depth_center = (depth_start + depth_end) / 2.0
                bin_width = depth_end - depth_start
            else:
                depth_start = depth_edges[i]
                depth_end = depth_edges[i + 1]
                depth_center = (depth_start + depth_end) / 2.0
                bin_width = actual_interval
            
            # Find patches within current depth interval
            mask = (patch_depths >= depth_start) & (patch_depths < depth_end)
            if i == bins - 1:  # Include right boundary for last interval
                mask = (patch_depths >= depth_start) & (patch_depths <= depth_end)
            
            if np.any(mask):
                slips_in_bin = patch_slips[mask]
                if method == 'mean':
                    stat_value = np.mean(slips_in_bin)
                elif method == 'max':
                    stat_value = np.max(slips_in_bin)
                elif method == 'median':
                    stat_value = np.median(slips_in_bin)
                else:
                    stat_value = np.mean(slips_in_bin)
                std_value = np.std(slips_in_bin)
            else:
                stat_value = 0.0
                std_value = 0.0
            
            depth_centers.append(depth_center)
            stat_values.append(stat_value)
            std_values.append(std_value)
        
        depth_centers = np.array(depth_centers)
        stat_values = np.array(stat_values)
        std_values = np.array(std_values)
        
        # Apply normalization to statistics if requested
        stats_norm_factor = 1.0
        if normalize_stats:
            stats_max = np.max(np.abs(stat_values))
            stats_norm_factor = stats_max if stats_max != 0 else 1.0
            stat_values = stat_values / stats_norm_factor
            # Also normalize standard deviations with the same factor
            std_values = std_values / stats_norm_factor
            print(f"Applied post-statistics normalization with factor: {stats_norm_factor:.6f}")
        
        # Store parameters
        params = {
            'bins': bins,
            'depth_min': depth_min,
            'depth_max': depth_max,
            'method': method,
            'slip_type': slip if isinstance(slip, str) else 'custom',
            'normalized': normalize,
            'norm_method': norm_method if normalize else None,
            'normalize_stats': normalize_stats,
            'custom_bins': custom_bins,
            'auto_depth_layers': auto_depth_layers
        }
        
        # Add bin-specific information
        if custom_bins:
            params['custom_depth_edges'] = depth_edges.tolist()
            params['bin_widths'] = [depth_edges[i+1] - depth_edges[i] for i in range(bins)]
            params['actual_interval'] = 'variable'
        else:
            params['actual_interval'] = actual_interval
        
        result_dict = {
            'depth_centers': depth_centers,
            'stat_values': stat_values,
            'std_values': std_values,
            'patch_depths': patch_depths,
            'patch_slips': patch_slips,
            'depth_edges': depth_edges,
            'params': params
        }
        
        # Add normalization factors if applied
        if normalize:
            result_dict['norm_factor'] = norm_factor
        if normalize_stats:
            result_dict['stats_norm_factor'] = stats_norm_factor
        
        return result_dict

    def _detect_depth_layers(self, layer_tolerance=0.1, min_layer_thickness=0.5, 
                            clustering_method='simple_clustering', max_layers=20):
        """
        Automatically detect depth layers from fault vertices (optimized for rectangular patches)
        
        Methods available:
        - 'simple_clustering': Sort and merge by tolerance, use group centers as boundaries (recommended)
        """
        # Get all vertex depths
        all_depths = np.abs(self.source.Vertices[:, 2])
        depth_min, depth_max = all_depths.min(), all_depths.max()
        
        print(f"Vertex depth range: {depth_min:.2f} - {depth_max:.2f} km")
        print(f"Total vertices: {len(all_depths)}")
        
        if clustering_method == 'simple_clustering':
            # New method: Simple clustering, use group centers as boundaries
            depth_edges = self._detect_layers_simple_clustering(all_depths, layer_tolerance, 
                                                               min_layer_thickness, max_layers)
        
        # Ensure depth_edges includes boundaries and is sorted
        if len(depth_edges) > 0:
            depth_edges = np.unique(np.concatenate([[depth_min], depth_edges, [depth_max]]))
        else:
            depth_edges = np.array([depth_min, depth_max])
        depth_edges = np.sort(depth_edges)
        
        return depth_edges
    
    def _detect_layers_simple_clustering(self, all_depths, layer_tolerance, min_layer_thickness, max_layers):
        """
        Simple clustering method - Sort and merge by tolerance, use group centers as boundaries
        """
        print("Using simple clustering method (group centers as boundaries)...")
        
        # 1. Remove duplicates and sort
        unique_depths = np.unique(np.round(all_depths, 4))  # Keep 4 decimal precision
        print(f"Found {len(unique_depths)} unique depths")
        
        # 2. Merge nearby depth values by tolerance
        depth_groups = []
        current_group = [unique_depths[0]]
        
        for i in range(1, len(unique_depths)):
            depth_diff = unique_depths[i] - unique_depths[i-1]
            
            # If depth difference is smaller than tolerance, merge into current group
            if depth_diff <= layer_tolerance:
                current_group.append(unique_depths[i])
            else:
                # Otherwise save current group and start new group
                depth_groups.append(current_group)
                current_group = [unique_depths[i]]
        
        # Save the last group
        depth_groups.append(current_group)
        
        print(f"Merged into {len(depth_groups)} depth groups")
        
        if len(depth_groups) <= 1:
            print("Only one depth group found, using uniform binning")
            return np.linspace(all_depths.min(), all_depths.max(), 6)[1:-1]
        
        # 3. Calculate representative depth for each group (median) - directly as boundaries!
        group_centers = []
        for group in depth_groups:
            center = np.median(group)
            group_centers.append(center)
        
        print(f"Group centers (direct boundaries): {group_centers}")
        
        # 4. Check layer thickness constraint - remove too close layer centers
        filtered_centers = [group_centers[0]]
        for i in range(1, len(group_centers)):
            if (group_centers[i] - filtered_centers[-1]) >= min_layer_thickness:
                filtered_centers.append(group_centers[i])
        
        # 5. Limit number of layers
        if len(filtered_centers) > max_layers:
            filtered_centers = filtered_centers[:max_layers]
        
        print(f"Final layer centers (as boundaries): {filtered_centers}")
        
        # 6. Return group centers as boundaries directly (exclude first and last, as they will be added in parent function)
        if len(filtered_centers) <= 2:
            return np.array([])
        
        # Return middle group centers as boundaries
        boundaries = filtered_centers[1:-1]  # Exclude first and last
        
        print(f"Internal boundaries: {boundaries}")
        return np.array(boundaries)
    
    def _normalize_data(self, data, norm_method='max'):
        """
        Normalize data based on specified method.
        
        Parameters:
        -----------
        data : array
            Data to normalize
        norm_method : str or float
            Normalization method ('max', 'sum', 'l2', 'percentile', or custom value)
            
        Returns:
        --------
        tuple: (normalized_data, norm_factor)
        """
        data = np.array(data)
        
        if norm_method == 'max':
            # Normalize by maximum value
            norm_factor = np.max(np.abs(data))
        elif norm_method == 'sum':
            # Normalize by sum of absolute values
            norm_factor = np.sum(np.abs(data))
        elif norm_method == 'l2':
            # Normalize by L2 norm
            norm_factor = np.linalg.norm(data)
        elif norm_method == 'percentile':
            # Normalize by 95th percentile (robust to outliers)
            norm_factor = np.percentile(np.abs(data), 95)
        elif isinstance(norm_method, (int, float)):
            # Normalize by custom value
            norm_factor = float(norm_method)
        else:
            print(f"Warning: Unknown normalization method '{norm_method}', using max")
            norm_factor = np.max(np.abs(data))
        
        if norm_factor == 0:
            print("Warning: Normalization factor is zero, skipping normalization")
            return data, 1.0
        
        normalized_data = data / norm_factor
        
        print(f"Applied {norm_method} normalization with factor: {norm_factor:.6f}")
        print(f"Data range after normalization: {normalized_data.min():.6f} - {normalized_data.max():.6f}")
        
        return normalized_data, norm_factor
    
    def plot_depth_histogram(self, results, plot_curve=True, plot_patch_slip=True, figsize=(4, 5)):
        """
        Plot depth histogram with scientific publication style
        
        Parameters:
        -----------
        results : dict
            Results from compute_statistics
        plot_curve : bool
            Whether to plot statistical curves
        plot_patch_slip : bool
            Whether to plot individual patch slips
        figsize : tuple
            Figure size (A4 by default)
            
        Returns:
        --------
        fig, axes : matplotlib objects
        """
        # Extract data
        depth_centers = results['depth_centers']
        stat_values = results['stat_values']
        std_values = results['std_values']
        patch_depths = results['patch_depths']
        patch_slips = results['patch_slips']
        depth_edges = results['depth_edges']
        params = results['params']
        
        with sci_plot_style():
            # Get color palette with enough colors
            colors = sns.color_palette("husl", 8)  # Ensure we have at least 8 colors
            
            # Create figure with scientific style
            fig, axes = plt.subplots(2, 1, figsize=figsize, facecolor='white')
            
            # Main plot: Depth vs statistical values
            ax1 = axes[0]
            
            # Create continuous histogram coordinates
            hist_x = []
            hist_y = []
            for i in range(params['bins']):
                hist_x.extend([0, stat_values[i], stat_values[i], 0])
                hist_y.extend([depth_edges[i], depth_edges[i], depth_edges[i+1], depth_edges[i+1]])
            
            # Plot histogram
            ax1.plot(hist_x, hist_y, linewidth=2, color=colors[0], 
                    label='Histogram', alpha=0.8)
            
            # Fill areas
            for i in range(params['bins']):
                ax1.fill_betweenx([depth_edges[i], depth_edges[i+1]], 0, stat_values[i], 
                                alpha=0.3, color=colors[0])
            
            # Scatter plot of individual patches
            if plot_patch_slip:
                ax1.scatter(patch_slips, patch_depths, alpha=0.4, s=9, 
                        color=colors[1], label='Individual patches', zorder=3)
            
            # Statistical center points
            # ax1.scatter(stat_values, depth_centers, # s=60, 
            #             color=colors[2], 
            #         marker='o', edgecolors='white', # linewidth=1.5, 
            #         label='Bin centers', zorder=5)
            
            # Optional statistical curve
            if plot_curve:
                ax1.plot(stat_values, depth_centers, # linewidth=2.5, 
                        color=colors[3], linestyle='--', marker='o', 
                        markersize=5, label=f'{params["method"].title()} curve', zorder=4)
            
            ax1.set_ylabel('Depth (km)')
            ax1.set_xlabel(f'{params["method"].title()} Slip (m)')
            ax1.set_title(f'Slip Distribution vs Depth ({params["method"].title()})', 
                        fontweight='bold')
            ax1.legend(loc='best', framealpha=0.9)
            ax1.grid(True, alpha=0.3)
            
            # Subplot: Standard deviation
            ax2 = axes[1]
            
            # Create standard deviation histogram
            std_hist_x = []
            std_hist_y = []
            for i in range(params['bins']):
                std_hist_x.extend([0, std_values[i], std_values[i], 0])
                std_hist_y.extend([depth_edges[i], depth_edges[i], depth_edges[i+1], depth_edges[i+1]])
            
            ax2.plot(std_hist_x, std_hist_y, linewidth=2, color=colors[4], 
                    label='Std Histogram', alpha=0.8)
            
            # Fill areas
            for i in range(params['bins']):
                ax2.fill_betweenx([depth_edges[i], depth_edges[i+1]], 0, std_values[i], 
                                alpha=0.3, color=colors[4])
            
            # Standard deviation center points
            # ax2.scatter(std_values, depth_centers, s=60, color=colors[5], 
            #         marker='s', edgecolors='white', linewidth=1.5, 
            #         label='Std centers', zorder=5)
            
            # Optional standard deviation curve
            if plot_curve:
                ax2.plot(std_values, depth_centers, # linewidth=2.5, 
                        color=colors[6], linestyle='--', marker='s', 
                        markersize=5, label='Std curve', zorder=4)
            
            ax2.set_ylabel('Depth (km)')
            ax2.set_xlabel('Slip Std (m)')
            ax2.set_title('Slip Variability vs Depth', fontweight='bold')
            ax2.legend(loc='best', framealpha=0.9)
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            return fig, axes
    
    def save_outputs(self, results, outfile=None, outfile_curve=None):
        """
        Save results to files
        
        Parameters:
        -----------
        results : dict
            Results from compute_statistics
        outfile : str, optional
            GMT output filename for histogram rectangles
        outfile_curve : str, optional
            Output filename for depth-slip curve data
        """
        # Extract data
        depth_centers = results['depth_centers']
        stat_values = results['stat_values']
        std_values = results['std_values']
        depth_edges = results['depth_edges']
        params = results['params']
        
        # Output GMT format file for histogram rectangles
        if outfile:
            with open(outfile, 'w') as f:
                for i in range(params['bins']):
                    depth_start = depth_edges[i]
                    depth_end = depth_edges[i + 1]
                    slip_val = stat_values[i]
                    
                    # Write four vertices of rectangle (slip as X-axis, depth as Y-axis)
                    f.write('>\n')
                    f.write(f'{0:.3f} {depth_start:.3f}\n')
                    f.write(f'{slip_val:.3f} {depth_start:.3f}\n')
                    f.write(f'{slip_val:.3f} {depth_end:.3f}\n')
                    f.write(f'{0:.3f} {depth_end:.3f}\n')
        
        # Output curve data
        if outfile_curve:
            with open(outfile_curve, 'w') as f:
                f.write('# Depth(km) Mean_Slip(m) Std_Slip(m) Bin_Width(km)\n')
                for i in range(len(depth_centers)):
                    bin_width = depth_edges[i+1] - depth_edges[i]
                    f.write(f'{depth_centers[i]:.3f} {stat_values[i]:.3f} {std_values[i]:.3f} {bin_width:.3f}\n')
    
    def print_summary(self, results):
        """
        Print statistical summary
        
        Parameters:
        -----------
        results : dict
            Results from compute_statistics
        """
        params = results['params']
        stat_values = results['stat_values']
        depth_centers = results['depth_centers']
        depth_edges = results['depth_edges']
        patch_depths = results['patch_depths']
        
        print(f"Depth Statistical Summary:")
        print(f"Depth range: {params['depth_min']:.2f} - {params['depth_max']:.2f} km")
        print(f"Number of bins: {params['bins']}")
        
        if params['custom_bins']:
            print(f"Using custom depth edges: {params['custom_depth_edges']}")
            print(f"Bin widths: {[f'{w:.2f}' for w in params['bin_widths']]} km")
        else:
            print(f"Uniform depth interval: {params['actual_interval']:.2f} km")
        
        print(f"Maximum {params['method']} slip: {np.max(stat_values):.3f} m at depth {depth_centers[np.argmax(stat_values)]:.2f} km")
        print(f"Total patches: {len(patch_depths)}")
        
        if params['custom_bins']:
            print("\nBin details:")
            for i in range(params['bins']):
                print(f"  Bin {i+1}: {depth_edges[i]:.1f}-{depth_edges[i+1]:.1f} km, "
                      f"center: {depth_centers[i]:.1f} km, "
                      f"{params['method']}: {stat_values[i]:.3f} m")


# Usage example in main section
if __name__ == '__main__':
    
    # 使用DepthStatistics类进行自动深度分层
    from .stat_utils import DepthStatistics
    
    # 初始化分析器
    analyzer = DepthStatistics(source)
    
    # 方法1: 直方图峰值检测（推荐，不需要额外依赖）
    results = analyzer.compute_statistics(
        slip='total', 
        method='mean',
        auto_depth_layers=True,           # 启用自动分层
        layer_tolerance=0.5,              # 分层容差 0.5km
        min_layer_thickness=1.0,          # 最小分层厚度 1km
        clustering_method='histogram',    # 使用直方图方法
        normalize_stats=True,             # 标准化统计结果
    )
    
    # 方法2: K-means聚类（需要sklearn）
    results = analyzer.compute_statistics(
        slip='total', 
        method='mean',
        auto_depth_layers=True,
        layer_tolerance=0.3,
        min_layer_thickness=0.5,
        clustering_method='kmeans',       # 使用K-means聚类
        max_layers=15,                    # 最大分层数
    )
    
    # 方法3: 密度聚类（需要sklearn）
    results = analyzer.compute_statistics(
        slip='total', 
        method='mean',
        auto_depth_layers=True,
        layer_tolerance=0.4,
        clustering_method='density',      # 使用DBSCAN密度聚类
    )
    
    # 绘图和保存结果
    fig, axes = analyzer.plot_depth_histogram(results, plot_curve=True, plot_patch_slip=True)
    plt.savefig('auto_depth_layers_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    analyzer.save_outputs(results, 
                         outfile='auto_depth_hist.gmt',
                         outfile_curve='auto_depth_curve.dat')
    analyzer.print_summary(results)