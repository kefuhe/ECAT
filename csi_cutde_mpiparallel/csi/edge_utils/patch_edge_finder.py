import numpy as np
from scipy.spatial.distance import cdist

class RectangularPatchAnalyzer:
    """
    Utility class for analyzing rectangular patches structure
    """
    
    def __init__(self, patches, merge_threshold=1e-2):
        """
        Initialize patch analyzer for rectangular patches
        
        Parameters:
        -----------
        patches : array or list
            Rectangular patches, each patch contains 4 corner points (N, 4, 3)
            or list of patches where each patch is (4, 3) array
        merge_threshold : float
            Distance threshold for merging duplicate vertices
        """
        self.patches = np.array(patches)
        self.merge_threshold = merge_threshold
        
        # Process patches to get unique vertices and connectivity
        self.vertices, self.patch_vertex_indices = self._process_patches()
        
    def _process_patches(self):
        """
        Process patches to extract unique vertices and remove duplicates
        
        Returns:
        --------
        vertices : array
            Unique vertices (N, 3)
        patch_vertex_indices : list
            List of vertex indices for each patch
        """
        # Flatten all patch vertices
        if self.patches.ndim == 3:
            # Shape: (num_patches, 4, 3)
            all_vertices = self.patches.reshape(-1, 3)
        else:
            # List of patches
            all_vertices = np.vstack([np.array(patch) for patch in self.patches])
        
        print(f"Total vertices before deduplication: {len(all_vertices)}")
        
        # Remove duplicate vertices
        unique_vertices, inverse_indices = self._remove_duplicate_vertices(all_vertices)
        
        print(f"Unique vertices after deduplication: {len(unique_vertices)}")
        
        # Build patch connectivity using unique vertex indices
        patch_vertex_indices = []
        vertex_idx = 0
        
        for i in range(len(self.patches)):
            patch_indices = inverse_indices[vertex_idx:vertex_idx+4]
            patch_vertex_indices.append(patch_indices.tolist())
            vertex_idx += 4
        
        return unique_vertices, patch_vertex_indices
    
    def _remove_duplicate_vertices(self, vertices):
        """
        Remove duplicate vertices based on distance threshold
        
        Parameters:
        -----------
        vertices : array
            All vertices including duplicates (N, 3)
            
        Returns:
        --------
        unique_vertices : array
            Unique vertices (M, 3) where M <= N
        inverse_indices : array
            Mapping from original to unique indices (N,)
        """
        unique_vertices = []
        inverse_indices = np.zeros(len(vertices), dtype=int)
        
        for i, vertex in enumerate(vertices):
            if len(unique_vertices) == 0:
                unique_vertices.append(vertex)
                inverse_indices[i] = 0
            else:
                # Calculate distances to existing unique vertices
                distances = np.linalg.norm(np.array(unique_vertices) - vertex, axis=1)
                min_distance_idx = np.argmin(distances)
                
                if distances[min_distance_idx] < self.merge_threshold:
                    # Use existing vertex
                    inverse_indices[i] = min_distance_idx
                else:
                    # Add new unique vertex
                    unique_vertices.append(vertex)
                    inverse_indices[i] = len(unique_vertices) - 1
        
        return np.array(unique_vertices), inverse_indices
    
    def find_fault_fouredge_vertices(self, depth_tolerance=0.1, min_layer_thickness=0.5, 
                                    sort_axis='auto', ascending=True):
        """
        Find the four edge vertices of rectangular fault patches
        
        Parameters:
        -----------
        depth_tolerance : float
            Tolerance for grouping vertices into depth layers (default: 0.1 km)
        min_layer_thickness : float
            Minimum thickness between layers (default: 0.5 km)
        sort_axis : str
            Axis for lateral sorting within each layer ('x', 'y', 'auto')
            'auto' uses PCA to find main horizontal direction
        ascending : bool
            Sort order for lateral arrangement (True: ascending, False: descending)
            
        Returns:
        --------
        Sets self.edge_vertex_indices and self.edge_vertices attributes
        Also returns dict containing detailed edge information
        """
        # Get all vertex depths (absolute Z values)
        vertex_depths = np.abs(self.vertices[:, 2])
        
        # Group vertices into depth layers
        depth_layers = self._group_vertices_by_depth(vertex_depths, depth_tolerance, min_layer_thickness)
        
        print(f"Found {len(depth_layers)} depth layers from {len(self.vertices)} unique vertices")
        
        # Get top and bottom edges
        top_layer = depth_layers[0]  # Shallowest
        bottom_layer = depth_layers[-1]  # Deepest
        
        top_inds = top_layer['indices']
        bottom_inds = bottom_layer['indices']
        top_pnts = self.vertices[top_inds]
        bottom_pnts = self.vertices[bottom_inds]
        
        # Determine sorting axis
        if sort_axis == 'auto':
            sort_coords, sort_method = self._get_pca_coordinates()
            print(f"Using PCA-based sorting along main horizontal direction")
        elif sort_axis.lower() == 'x':
            sort_coords = self.vertices[:, 0]
            sort_method = 'x-axis'
            print(f"Using X-axis for sorting")
        elif sort_axis.lower() == 'y':
            sort_coords = self.vertices[:, 1] 
            sort_method = 'y-axis'
            print(f"Using Y-axis for sorting")
        else:
            raise ValueError("sort_axis must be 'x', 'y', or 'auto'")
        
        # Find left and right edges by collecting extreme vertices from all layers
        left_indices = []
        right_indices = []
        
        for layer in depth_layers:
            layer_sort_coords = sort_coords[layer['indices']]
            layer_indices = np.array(layer['indices'])
            
            if ascending:
                # Left: minimum coordinates, Right: maximum coordinates
                left_idx = layer_indices[np.argmin(layer_sort_coords)]
                right_idx = layer_indices[np.argmax(layer_sort_coords)]
            else:
                # Reverse the order
                left_idx = layer_indices[np.argmax(layer_sort_coords)]
                right_idx = layer_indices[np.argmin(layer_sort_coords)]
            
            left_indices.append(left_idx)
            right_indices.append(right_idx)
        
        # Sort edge indices by depth for consistent ordering
        left_depths = vertex_depths[left_indices]
        right_depths = vertex_depths[right_indices]
        
        left_sort_order = np.argsort(left_depths)
        right_sort_order = np.argsort(right_depths)
        
        left_inds = np.array(left_indices)[left_sort_order].tolist()
        right_inds = np.array(right_indices)[right_sort_order].tolist()
        left_pnts = self.vertices[left_inds]
        right_pnts = self.vertices[right_inds]
        
        # Set class attributes in the required format
        self.edge_vertex_indices = {
            'top': top_inds,
            'bottom': bottom_inds,
            'left': left_inds, 
            'right': right_inds
        }
        
        self.edge_vertices = {
            'top': top_pnts,
            'bottom': bottom_pnts,
            'left': left_pnts, 
            'right': right_pnts
        }
        
        # Return detailed information (optional)
        return {
            'edge_vertex_indices': self.edge_vertex_indices,
            'edge_vertices': self.edge_vertices,
            'depth_layers': depth_layers,
            'sort_info': {
                'axis': sort_axis,
                'method': sort_method,
                'ascending': ascending
            },
            'patch_info': {
                'total_patches': len(self.patches),
                'unique_vertices': len(self.vertices),
                'original_vertices': len(self.patches) * 4
            }
        }
    
    def _group_vertices_by_depth(self, vertex_depths, tolerance, min_thickness):
        """
        Group vertices into depth layers
        """
        # Remove duplicates and sort
        unique_depths = np.unique(np.round(vertex_depths, 4))
        print(f"Found {len(unique_depths)} unique depths")
        
        # Group by tolerance
        depth_groups = []
        current_group = [unique_depths[0]]
        
        for i in range(1, len(unique_depths)):
            depth_diff = unique_depths[i] - unique_depths[i-1]
            
            if depth_diff <= tolerance:
                current_group.append(unique_depths[i])
            else:
                depth_groups.append(current_group)
                current_group = [unique_depths[i]]
        
        depth_groups.append(current_group)
        
        print(f"Merged into {len(depth_groups)} depth groups")
        
        # Calculate representative depth for each group
        layers = []
        for group in depth_groups:
            group_depth = np.median(group)
            
            # Find all vertices belonging to this depth group
            group_indices = []
            for depth_val in group:
                mask = np.abs(vertex_depths - depth_val) < tolerance/2
                group_indices.extend(np.where(mask)[0].tolist())
            
            # Remove duplicates and sort
            group_indices = sorted(list(set(group_indices)))
            
            if len(group_indices) > 0:
                layers.append({
                    'depth': group_depth,
                    'indices': group_indices,
                    'depth_range': [min(group), max(group)],
                    'vertex_count': len(group_indices)
                })
        
        # Sort layers by depth (shallow to deep)
        layers.sort(key=lambda x: x['depth'])
        
        # Apply minimum thickness constraint
        filtered_layers = [layers[0]] if layers else []
        for layer in layers[1:]:
            if layer['depth'] - filtered_layers[-1]['depth'] >= min_thickness:
                filtered_layers.append(layer)
        
        print(f"After thickness filtering: {len(filtered_layers)} layers")
        
        return filtered_layers
    
    def _get_pca_coordinates(self):
        """
        Get coordinates along main horizontal direction using PCA
        """
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            print("sklearn not available, falling back to X-axis")
            return self.vertices[:, 0], 'x-axis (PCA unavailable)'
        
        # Use only X and Y coordinates for horizontal PCA
        horizontal_coords = self.vertices[:, :2]
        
        # Perform PCA
        pca = PCA(n_components=2)
        pca_coords = pca.fit_transform(horizontal_coords)
        
        # Return coordinates along first principal component
        return pca_coords[:, 0], f'PCA (explained variance: {pca.explained_variance_ratio_[0]:.3f})'
    
    def print_edge_summary(self):
        """
        Print summary of detected edges and patch information
        """
        if not hasattr(self, 'edge_vertex_indices'):
            print("No edge information available. Run find_fault_fouredge_vertices() first.")
            return
        
        print("\n=== Rectangular Patch Analysis Summary ===")
        print(f"Total patches: {len(self.patches)}")
        print(f"Original vertices: {len(self.patches) * 4}")
        print(f"Unique vertices: {len(self.vertices)}")
        print(f"Duplicate vertices removed: {len(self.patches) * 4 - len(self.vertices)}")
        
        # Calculate edge depths for summary
        top_depths = np.abs(self.edge_vertices['top'][:, 2])
        bottom_depths = np.abs(self.edge_vertices['bottom'][:, 2])
        
        print(f"\nFault Four Edge Vertices:")
        print(f"Top edge (depth {np.mean(top_depths):.2f} km): {len(self.edge_vertex_indices['top'])} vertices")
        print(f"Bottom edge (depth {np.mean(bottom_depths):.2f} km): {len(self.edge_vertex_indices['bottom'])} vertices")
        print(f"Left edge: {len(self.edge_vertex_indices['left'])} vertices")
        print(f"Right edge: {len(self.edge_vertex_indices['right'])} vertices")
        
        print(f"\nVertex index ranges:")
        for edge, indices in self.edge_vertex_indices.items():
            print(f"  {edge.capitalize()} edge indices: {min(indices)} - {max(indices)}")
        
    def interpolate_curve_at_depth(self, target_depth, variable_axis='auto', ascending=True, 
                                  depth_tolerance=0.5, output_file=None, verbose=True,
                                  layer_depth_tolerance=None, layer_min_thickness=None):
        """
        Interpolate a curve at specified depth using depth layers
        
        This method generates a 3D curve at any target depth by interpolating between
        existing depth layers. If the target depth is close to an existing layer,
        it returns that layer directly. Otherwise, it performs bilinear interpolation
        between adjacent layers.
        
        Parameters:
        -----------
        target_depth : float
            Target depth for curve generation (positive value, in km)
        variable_axis : str, optional
            Independent variable axis ('x', 'y', or 'auto'). Default: 'auto'
            - 'x': Use X-coordinate as independent variable
            - 'y': Use Y-coordinate as independent variable  
            - 'auto': Use the axis with larger range
        ascending : bool, optional
            Sort order for the independent variable (default: True)
        depth_tolerance : float, optional
            Tolerance for considering target depth as existing layer (default: 0.5 km)
        output_file : str, optional
            Output GMT format file path (default: None, no file output)
        verbose : bool, optional
            Enable detailed output messages (default: True)
        layer_depth_tolerance : float, optional
            Tolerance for grouping vertices into depth layers (default: None, uses 0.1 km)
            Override this for different fault geometries
        layer_min_thickness : float, optional
            Minimum thickness between depth layers (default: None, uses 0.5 km)
            Override this for different fault geometries
            
        Returns:
        --------
        curve_points : ndarray
            Interpolated curve points (N, 3) in [x, y, z] format
        curve_info : dict
            Dictionary containing interpolation information:
            - 'method': str, interpolation method used
            - 'source_layers': list, source depth layers used
            - 'variable_axis': str, independent variable axis
            - 'point_count': int, number of points in curve
            - 'depth_range': list, actual depth range of curve points
            
        Examples:
        ---------
        >>> analyzer = RectangularPatchAnalyzer(patches)
        >>> analyzer.find_fault_fouredge_vertices()
        >>> 
        >>> # Get curve at specific depth with default layer parameters
        >>> curve, info = analyzer.interpolate_curve_at_depth(5.0, variable_axis='x')
        >>> 
        >>> # Custom layer parameters for fine-resolution faults
        >>> curve, info = analyzer.interpolate_curve_at_depth(
        ...     target_depth=7.5,
        ...     layer_depth_tolerance=0.05,
        ...     layer_min_thickness=0.2,
        ...     output_file='curve_7p5km.gmt'
        ... )
        
        Notes:
        ------
        - If target_depth matches an existing layer (within tolerance), returns
          that layer's points sorted by the independent variable
        - For interpolation between layers, endpoint coordinates come from linear
          interpolation, while intermediate points use bilinear interpolation
        - The method automatically handles edge cases and ensures smooth curves
        - Adjust layer_depth_tolerance and layer_min_thickness for different fault resolutions
        """
        if not hasattr(self, 'edge_vertex_indices'):
            raise ValueError("Must run find_fault_fouredge_vertices() first")
        
        target_depth = abs(target_depth)  # Ensure positive depth
        
        # Set default layer parameters if not provided
        if layer_depth_tolerance is None:
            layer_depth_tolerance = 0.1
        if layer_min_thickness is None:
            layer_min_thickness = 0.5
        
        # Get depth layers from previous analysis with adjustable parameters
        vertex_depths = np.abs(self.vertices[:, 2])
        depth_layers = self._group_vertices_by_depth(vertex_depths, layer_depth_tolerance, layer_min_thickness)
        
        if verbose:
            print(f"Interpolating curve at depth {target_depth:.2f} km")
            print(f"Layer parameters: tolerance={layer_depth_tolerance:.3f} km, min_thickness={layer_min_thickness:.3f} km")
            layer_depths_out = [f"{layer['depth']:.2f} km" for layer in depth_layers]
            print(f"Available depth layers: {layer_depths_out}")
    
        # Check if target depth is close to an existing layer
        for layer in depth_layers:
            if abs(layer['depth'] - target_depth) < depth_tolerance:
                if verbose:
                    print(f"Target depth matches existing layer (depth: {layer['depth']:.2f} km)")
                return self._get_layer_curve(layer, variable_axis, ascending, output_file, verbose)
        
        # Find adjacent layers for interpolation
        layer_depths = [layer['depth'] for layer in depth_layers]
        
        # Handle edge cases
        if target_depth <= min(layer_depths):
            if verbose:
                print(f"Target depth above shallowest layer, using layer at {min(layer_depths):.2f} km")
            shallow_layer = depth_layers[0]
            return self._get_layer_curve(shallow_layer, variable_axis, ascending, output_file, verbose)
        
        if target_depth >= max(layer_depths):
            if verbose:
                print(f"Target depth below deepest layer, using layer at {max(layer_depths):.2f} km")
            deep_layer = depth_layers[-1]
            return self._get_layer_curve(deep_layer, variable_axis, ascending, output_file, verbose)
        
        # Find bracketing layers
        upper_layer = None
        lower_layer = None
        
        for i, layer in enumerate(depth_layers):
            if layer['depth'] < target_depth:
                upper_layer = layer
            elif layer['depth'] > target_depth and lower_layer is None:
                lower_layer = layer
                break
        
        if upper_layer is None or lower_layer is None:
            raise ValueError("Cannot find bracketing layers for interpolation")
        
        if verbose:
            print(f"Interpolating between layers at {upper_layer['depth']:.2f} km and {lower_layer['depth']:.2f} km")
        
        # Perform bilinear interpolation
        curve_points = self._bilinear_interpolate_curve(
            upper_layer, lower_layer, target_depth, variable_axis, ascending, verbose
        )
        
        # Create curve info
        curve_info = {
            'method': 'bilinear_interpolation',
            'source_layers': [upper_layer['depth'], lower_layer['depth']],
            'variable_axis': variable_axis,
            'point_count': len(curve_points),
            'depth_range': [curve_points[:, 2].min(), curve_points[:, 2].max()],
            'target_depth': target_depth,
            'layer_parameters': {
                'depth_tolerance': layer_depth_tolerance,
                'min_thickness': layer_min_thickness
            }
        }
        
        # Output to file if requested
        if output_file is not None:
            self._write_curve_to_gmt(curve_points, output_file, curve_info, verbose)
        
        if verbose:
            print(f"Generated curve with {len(curve_points)} points")
            print(f"Actual depth range: {curve_info['depth_range'][0]:.3f} - {curve_info['depth_range'][1]:.3f} km")
        
        return curve_points, curve_info
    
    def _get_layer_curve(self, layer, variable_axis, ascending, output_file, verbose):
        """
        Get curve from existing depth layer
        """
        layer_points = self.vertices[layer['indices']]
        
        # Determine variable axis
        if variable_axis == 'auto':
            x_range = layer_points[:, 0].max() - layer_points[:, 0].min()
            y_range = layer_points[:, 1].max() - layer_points[:, 1].min()
            variable_axis = 'x' if x_range >= y_range else 'y'
            if verbose:
                print(f"Auto-selected {variable_axis}-axis as independent variable")
        
        # Sort by independent variable
        if variable_axis.lower() == 'x':
            sort_indices = np.argsort(layer_points[:, 0])
        elif variable_axis.lower() == 'y':
            sort_indices = np.argsort(layer_points[:, 1])
        else:
            raise ValueError("variable_axis must be 'x', 'y', or 'auto'")
        
        if not ascending:
            sort_indices = sort_indices[::-1]
        
        curve_points = layer_points[sort_indices]
        
        # Create curve info
        curve_info = {
            'method': 'existing_layer',
            'source_layers': [layer['depth']],
            'variable_axis': variable_axis,
            'point_count': len(curve_points),
            'depth_range': [curve_points[:, 2].min(), curve_points[:, 2].max()],
            'target_depth': layer['depth']
        }
        
        # Output to file if requested
        if output_file is not None:
            self._write_curve_to_gmt(curve_points, output_file, curve_info, verbose)
        
        return curve_points, curve_info
    
    def _bilinear_interpolate_curve(self, upper_layer, lower_layer, target_depth, 
                                   variable_axis, ascending, verbose):
        """
        Perform bilinear interpolation between two depth layers
        """
        upper_points = self.vertices[upper_layer['indices']]
        lower_points = self.vertices[lower_layer['indices']]
        
        # Determine variable axis
        if variable_axis == 'auto':
            all_points = np.vstack([upper_points, lower_points])
            x_range = all_points[:, 0].max() - all_points[:, 0].min()
            y_range = all_points[:, 1].max() - all_points[:, 1].min()
            variable_axis = 'x' if x_range >= y_range else 'y'
            if verbose:
                print(f"Auto-selected {variable_axis}-axis as independent variable")
        
        axis_idx = 0 if variable_axis.lower() == 'x' else 1
        other_idx = 1 if variable_axis.lower() == 'x' else 0
        
        # Sort both layers by independent variable axis
        upper_sort_indices = np.argsort(upper_points[:, axis_idx])
        lower_sort_indices = np.argsort(lower_points[:, axis_idx])
        
        if not ascending:
            upper_sort_indices = upper_sort_indices[::-1]
            lower_sort_indices = lower_sort_indices[::-1]
        
        upper_sorted = upper_points[upper_sort_indices]
        lower_sorted = lower_points[lower_sort_indices]
        
        # Get endpoint coordinates from linear interpolation
        upper_min_coord = upper_sorted[0, axis_idx]
        upper_max_coord = upper_sorted[-1, axis_idx]
        lower_min_coord = lower_sorted[0, axis_idx]
        lower_max_coord = lower_sorted[-1, axis_idx]
        
        # Linear interpolation factor based on depth
        depth_factor = (target_depth - upper_layer['depth']) / (lower_layer['depth'] - upper_layer['depth'])
        
        # Interpolated endpoints
        min_coord = upper_min_coord + depth_factor * (lower_min_coord - upper_min_coord)
        max_coord = upper_max_coord + depth_factor * (lower_max_coord - upper_max_coord)

        print('Upper layer min/max:', upper_min_coord, upper_max_coord)
        print('Lower layer min/max:', lower_min_coord, lower_max_coord)
        print('Interpolated min/max:', min_coord, max_coord)
        # Determine the number of points to generate (use more points from the layer with more vertices)
        num_points = max(len(upper_points), len(lower_points))
        
        # Generate interpolated coordinates along the independent axis
        if num_points > 1:
            interp_coords = np.linspace(min_coord, max_coord, num_points)
        else:
            interp_coords = np.array([(min_coord + max_coord) / 2])
        
        # Create interpolated curve points
        curve_points = []
        
        for interp_coord in interp_coords:
            # Interpolate the other coordinate using both layers
            
            # Interpolate from upper layer
            if len(upper_sorted) > 1:
                upper_other_coord = np.interp(interp_coord, upper_sorted[:, axis_idx], upper_sorted[:, other_idx])
            else:
                upper_other_coord = upper_sorted[0, other_idx]
            
            # Interpolate from lower layer
            if len(lower_sorted) > 1:
                lower_other_coord = np.interp(interp_coord, lower_sorted[:, axis_idx], lower_sorted[:, other_idx])
            else:
                lower_other_coord = lower_sorted[0, other_idx]
            
            # Linear interpolation between layers for the other coordinate
            interp_other_coord = upper_other_coord + depth_factor * (lower_other_coord - upper_other_coord)
            
            # Create point with correct coordinate order
            if variable_axis.lower() == 'x':
                point = [interp_coord, interp_other_coord, target_depth]
            else:
                point = [interp_other_coord, interp_coord, target_depth]
            
            curve_points.append(point)
        
        return np.array(curve_points)
    
    def _write_curve_to_gmt(self, curve_points, output_file, curve_info, verbose):
        """
        Write curve to GMT format file
        """
        try:
            with open(output_file, 'w') as f:
                # Write header
                f.write(f"# Interpolated fault curve at depth {curve_info['target_depth']:.2f} km\n")
                f.write(f"# Method: {curve_info['method']}\n")
                f.write(f"# Source layers: {curve_info['source_layers']}\n")
                f.write(f"# Variable axis: {curve_info['variable_axis']}\n")
                f.write(f"# Point count: {curve_info['point_count']}\n")
                f.write(f"# Format: X Y Z\n")
                
                # Write data points
                for point in curve_points:
                    f.write(f"{point[0]:.6f} {point[1]:.6f} {point[2]:.3f}\n")
            
            if verbose:
                print(f"Curve written to GMT file: {output_file}")
                
        except Exception as e:
            print(f"Error writing GMT file: {e}")

# Example usage
if __name__ == "__main__":
    # Example: Create some rectangular patch data
    patches = [
        # First rectangular patch's 4 corner points
        [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]],
        # Second rectangular patch's 4 corner points (with overlapping points)
        [[1, 0, 0], [2, 0, 0], [2, 1, 0], [1, 1, 0]],
        # Third rectangular patch's 4 corner points
        [[0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]]
    ]

    # Create analyzer
    analyzer = RectangularPatchAnalyzer(patches, merge_threshold=1e-2)

    # Find four edge vertices
    result = analyzer.find_fault_fouredge_vertices()

    # Print summary
    analyzer.print_edge_summary()