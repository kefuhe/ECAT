"""
Utility functions for processing and visualizing curves.
This module provides functionality to reconstruct ordered curves from unordered scatter points,
including handling overlapping points, finding endpoints, and optimizing paths.
It is particularly useful for geological applications such as fault trace reconstruction.

This code is part of a larger project and is designed to work with point data files.
It includes functions for loading, saving, and visualizing point data.

Author: Kefeng He
Date: 2025-07-17
License: MIT License
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from sklearn.neighbors import NearestNeighbors
import networkx as nx
from typing import List, Tuple, Optional, Dict

# Set matplotlib font support for better visualization
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


class CurveReconstructor:
    """
    A class for reconstructing ordered curves from unordered scatter points.
    Suitable for geological applications such as fault trace reconstruction.
    """
    
    def __init__(self, points: np.ndarray, merge_threshold: float = 1e-6, verbose: bool = False):
        """
        Initialize the curve reconstructor.
        
        Args:
            points: Point coordinate array with shape (n, 2) or (n, 3)
            merge_threshold: Distance threshold for merging overlapping points
            verbose: Whether to print detailed information during processing
        """
        self.original_points = np.array(points)
        # If 3D points, use only the first two dimensions for 2D processing
        if self.original_points.shape[1] > 2:
            self.points_2d = self.original_points[:, :2]
        else:
            self.points_2d = self.original_points
        
        self.merge_threshold = merge_threshold
        
        # Process overlapping points
        self.points, self.overlap_info = self._merge_overlapping_points(self.points_2d)
        self.n_points = len(self.points)
        
        if verbose:
            print(f"Original points: {len(self.original_points)}, After merging: {self.n_points}")
            if self.overlap_info:
                print(f"Found {len(self.overlap_info)} groups of overlapping points")
    
    def _merge_overlapping_points(self, points: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Merge points that are closer than the threshold distance.
        
        Args:
            points: Original point array
            
        Returns:
            Tuple of (merged point array, overlap information dictionary)
        """
        if len(points) == 0:
            return np.array([]), {}
        
        # Calculate distance matrix between points
        dist_matrix = cdist(points, points)
        
        # Find point pairs with distance less than threshold (excluding self)
        close_pairs = np.where((dist_matrix < self.merge_threshold) & (dist_matrix > 0))
        
        # Build connected components to find all overlapping point groups
        G = nx.Graph()
        G.add_nodes_from(range(len(points)))
        
        for i, j in zip(close_pairs[0], close_pairs[1]):
            G.add_edge(i, j)
        
        # Find all connected components (overlapping point groups)
        overlap_groups = list(nx.connected_components(G))
        
        # Merge overlapping points
        merged_points = []
        overlap_info = {}
        point_mapping = {}  # Mapping from original index to new index
        
        new_idx = 0
        for group in overlap_groups:
            group_list = list(group)
            if len(group_list) > 1:
                # Multiple overlapping points, calculate average position
                group_points = points[group_list]
                merged_point = np.mean(group_points, axis=0)
                merged_points.append(merged_point)
                
                # Record overlap information
                overlap_info[new_idx] = {
                    'original_indices': group_list,
                    'original_points': group_points.copy(),
                    'merged_point': merged_point,
                    'count': len(group_list)
                }
                
                # Update mapping
                for orig_idx in group_list:
                    point_mapping[orig_idx] = new_idx
                    
                new_idx += 1
            else:
                # Single point, keep as is
                merged_points.append(points[group_list[0]])
                point_mapping[group_list[0]] = new_idx
                new_idx += 1
        
        self.point_mapping = point_mapping
        return np.array(merged_points), overlap_info
    
    def get_original_path(self, merged_path: List[int]) -> List[int]:
        """
        Map the merged path back to original point indices.
        
        Args:
            merged_path: Path indices after merging
            
        Returns:
            Original point path indices
        """
        original_path = []
        reverse_mapping = {v: k for k, v in self.point_mapping.items()}
        
        for merged_idx in merged_path:
            if merged_idx in self.overlap_info:
                # For overlapping point groups, use the first original point index
                original_indices = self.overlap_info[merged_idx]['original_indices']
                original_path.append(original_indices[0])  # Choose first as representative
            else:
                # Single point, map back directly
                original_path.append(reverse_mapping[merged_idx])
        
        return original_path
    
    def find_endpoints(self, k: int = 3) -> Tuple[int, int]:
        """
        Find the two endpoints of the curve.
        Endpoints are usually points that are relatively far from other points.
        
        Args:
            k: Number of nearest neighbors to consider
            
        Returns:
            Indices of the two endpoints
        """
        if self.n_points < 2:
            return 0, 0
        
        k = min(k, self.n_points - 1)  # Ensure k doesn't exceed available points
        
        # Calculate average distance to k nearest neighbors for each point
        nbrs = NearestNeighbors(n_neighbors=k+1).fit(self.points)
        distances, indices = nbrs.kneighbors(self.points)
        
        # Exclude self (0th neighbor)
        avg_distances = np.mean(distances[:, 1:], axis=1)
        
        # Select the two points with largest distances as endpoints
        endpoint_indices = np.argsort(avg_distances)[-2:]
        
        return endpoint_indices[0], endpoint_indices[1]
    
    def build_mst_path(self) -> List[int]:
        """
        Reconstruct path using Minimum Spanning Tree method.
        
        Returns:
            Reconstructed point index sequence
        """
        if self.n_points < 2:
            return list(range(self.n_points))
        
        # Build complete graph distance matrix
        dist_matrix = cdist(self.points, self.points)
        
        # Create graph
        G = nx.Graph()
        for i in range(self.n_points):
            for j in range(i+1, self.n_points):
                G.add_edge(i, j, weight=dist_matrix[i, j])
        
        # Calculate minimum spanning tree
        mst = nx.minimum_spanning_tree(G)
        
        # Find nodes with degree 1 (leaf nodes) as starting points
        degrees = dict(mst.degree())
        endpoints = [node for node, degree in degrees.items() if degree == 1]
        
        if len(endpoints) >= 2:
            # Choose one endpoint as starting point
            start_node = endpoints[0]
        else:
            # If no obvious endpoints, choose the outermost point
            start_node, _ = self.find_endpoints()
        
        # DFS traversal of MST to build path
        visited = set()
        path = []
        
        def dfs(node):
            visited.add(node)
            path.append(node)
            for neighbor in mst.neighbors(node):
                if neighbor not in visited:
                    dfs(neighbor)
        
        dfs(start_node)
        return path
    
    def greedy_nearest_neighbor(self) -> List[int]:
        """
        Greedy nearest neighbor algorithm for path reconstruction.
        
        Returns:
            Reconstructed point index sequence
        """
        if self.n_points < 2:
            return list(range(self.n_points))
        
        # Find endpoints
        start_idx, end_idx = self.find_endpoints()
        
        # Build path greedily starting from start point
        path = [start_idx]
        remaining = set(range(self.n_points)) - {start_idx}
        
        current = start_idx
        while remaining:
            # Find nearest neighbor of current point
            distances = [np.linalg.norm(self.points[current] - self.points[idx]) 
                        for idx in remaining]
            nearest_idx = list(remaining)[np.argmin(distances)]
            
            path.append(nearest_idx)
            remaining.remove(nearest_idx)
            current = nearest_idx
            
        return path
    
    def optimize_path_2opt(self, path: List[int], max_iterations: int = 1000) -> List[int]:
        """
        Optimize path using 2-opt algorithm.
        
        Args:
            path: Initial path
            max_iterations: Maximum number of iterations
            
        Returns:
            Optimized path
        """
        if len(path) < 4:
            return path
        
        def calculate_path_length(p):
            length = 0
            for i in range(len(p) - 1):
                length += np.linalg.norm(self.points[p[i]] - self.points[p[i+1]])
            return length
        
        best_path = path.copy()
        best_length = calculate_path_length(best_path)
        
        for iteration in range(max_iterations):
            improved = False
            for i in range(1, len(path) - 2):
                for j in range(i + 1, len(path)):
                    if j - i == 1: continue  # Skip adjacent edges
                    
                    # Create new path: reverse the segment between i and j
                    new_path = path[:i] + path[i:j][::-1] + path[j:]
                    new_length = calculate_path_length(new_path)
                    
                    if new_length < best_length:
                        best_path = new_path
                        best_length = new_length
                        path = new_path
                        improved = True
                        break
                if improved:
                    break
            
            if not improved:
                break
                
        return best_path
    
    def reconstruct(self, method: str = 'mst', optimize: bool = True) -> Tuple[List[int], np.ndarray, List[int]]:
        """
        Reconstruct the curve.
        
        Args:
            method: Reconstruction method ('mst', 'greedy', 'hybrid')
            optimize: Whether to use 2-opt optimization
            
        Returns:
            Tuple of (merged index sequence, reconstructed point coordinates, original point indices)
        """
        if method == 'mst':
            path = self.build_mst_path()
        elif method == 'greedy':
            path = self.greedy_nearest_neighbor()
        elif method == 'hybrid':
            # Hybrid method: try both MST and greedy, choose the shorter path
            mst_path = self.build_mst_path()
            greedy_path = self.greedy_nearest_neighbor()
            
            # Choose the method with shorter path length
            if len(mst_path) > 1:
                mst_length = sum(np.linalg.norm(self.points[mst_path[i]] - self.points[mst_path[i+1]]) 
                               for i in range(len(mst_path)-1))
            else:
                mst_length = float('inf')
                
            if len(greedy_path) > 1:
                greedy_length = sum(np.linalg.norm(self.points[greedy_path[i]] - self.points[greedy_path[i+1]]) 
                                  for i in range(len(greedy_path)-1))
            else:
                greedy_length = float('inf')
            
            path = mst_path if mst_length < greedy_length else greedy_path
        else:
            raise ValueError(f"Unknown method: {method}")
        
        if optimize:
            path = self.optimize_path_2opt(path)
        
        reconstructed_points = self.points[path]
        original_path = self.get_original_path(path)
        
        return path, reconstructed_points, original_path
    
    def print_overlap_summary(self):
        """Print detailed information about overlapping points."""
        if not self.overlap_info:
            print("No overlapping points found")
            return
        
        print("\n=== Overlapping Points Summary ===")
        for merged_idx, info in self.overlap_info.items():
            print(f"Merged point {merged_idx}:")
            print(f"  Contains original points: {info['original_indices']}")
            print(f"  Number of overlapping points: {info['count']}")
            print(f"  Merged coordinates: ({info['merged_point'][0]:.6f}, {info['merged_point'][1]:.6f})")
            print(f"  Original coordinates:")
            for i, (orig_idx, point) in enumerate(zip(info['original_indices'], info['original_points'])):
                print(f"    Point {orig_idx}: ({point[0]:.6f}, {point[1]:.6f})")
            print()