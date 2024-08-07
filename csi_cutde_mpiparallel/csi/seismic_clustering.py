'''
Composite Seismic Clustering with Linear Probing Optimization (CSS-LPO)
Abbr.: CSS-LPO
Written by Kefeng He, April 2024.

This module implements the CSS-LPO method for seismic data analysis. 
The CSS-LPO method first applies a clustering algorithm to group similar seismic events together. 
Then, it uses a linear probing technique to detect and optimize the linear structures within each cluster. 
This method allows for the selection of different detection methods and the combination of multiple linear detection methods for further optimization.

The CSS-LPO method is designed to effectively identify linear features in seismic data, which can be crucial for understanding seismic activity and predicting future events. 
It provides a flexible and powerful tool for seismologists and researchers in related fields.
'''

# Externals
import numpy as np
from sklearn.cluster import DBSCAN, OPTICS
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering
from scipy.spatial import KDTree
from scipy.linalg import sqrtm
from skimage.transform import probabilistic_hough_line
from sklearn.linear_model import RANSACRegressor
from scipy.stats import linregress
from hdbscan import HDBSCAN
from abc import ABC, abstractmethod

# --------------------------Factory Class--------------------------#
class MethodFactory:
    def __init__(self):
        self._creators = {}

    def register_method(self, method_name, creator):
        if method_name in self._creators:
            raise ValueError(f"Method already registered: {method_name}")
        self._creators[method_name] = creator

    def unregister_method(self, method_name):
        if method_name not in self._creators:
            raise ValueError(f"Method not registered: {method_name}")
        del self._creators[method_name]

    def get_method(self, method_name):
        creator = self._creators.get(method_name)
        if not creator:
            raise ValueError(f"Method not registered: {method_name}")
        return creator()

    def is_registered(self, method_name):
        return method_name in self._creators

    def get_registered_methods(self):
        return list(self._creators.keys())


class ClusterMethodFactory(MethodFactory):
    pass

class LinearDetectionMethodFactory(MethodFactory):
    pass

def register_cluster_method(method_name):
    def decorator(creator):
        cluster_method_factory.register_method(method_name, creator)
        return creator
    return decorator

def register_linear_detection_method(method_name):
    def decorator(creator):
        linear_detection_method_factory.register_method(method_name, creator)
        return creator
    return decorator

# Create a factory for linear detection methods
cluster_method_factory = ClusterMethodFactory()

# Create a factory for linear detection methods
linear_detection_method_factory = LinearDetectionMethodFactory()

def is_linear_cluster(cluster_points, detection_methods, threshold=10):
    '''
    linear_detection_methods = {
        'hough': {'threshold': 5},
        'r2': {'threshold': 0.5},
        'pca': {'threshold': 0.85},
        'ransac': {'residual_threshold': 1.0}
    }

    is_linear = is_linear_cluster(cluster_points, linear_detection_methods)
    '''
    # Get the detectors
    detectors = {}
    for method, config in detection_methods.items():
        detector = linear_detection_method_factory.get_method(method)
        # Set the attributes of the detector
        for attr, value in config.items():
            setattr(detector, attr, value)
        detectors[method] = detector

    if len(cluster_points) < threshold:
        return False

    for method in detection_methods:
        if not detectors[method].detect(cluster_points):
            return False
    return True
# --------------------------Factory Class--------------------------#

#--------------------------Clustering earthquakes--------------------------#
class ClusterMethod(ABC):
    @abstractmethod
    def cluster(self, X_scaled, name, **kwargs):
        pass

@register_cluster_method('dbscan')
class DBSCANMethod(ClusterMethod):
    def cluster(self, X_scaled, eps=1, min_samples=5, **kwargs):
        '''
        Cluster the earthquakes on a profile using DBSCAN.
        DBSCAN: Density-Based Spatial Clustering of Applications with Noise
        Args:
            * eps                  : The maximum distance between two samples for one to be considered as in the neighborhood of the other. Default is 1.
            * min_samples          : The number of samples in a neighborhood for a point to be considered as a core point. Default is 5.
        '''
        db = DBSCAN(eps=eps, min_samples=min_samples).fit(X_scaled)
        return db.labels_

@register_cluster_method('optics')
class OPTICSMethod(ClusterMethod):
    def cluster(self, X_scaled, min_samples=5, xi=0.05, min_cluster_size=None, **kwargs):
        '''
        Cluster the earthquakes on a profile using OPTICS.
        OPTICS: Ordering Points To Identify the Clustering Structure
        Args:
            * min_samples          : The number of samples in a neighborhood for a point to be considered as a core point. Default is 5.
            * xi                   : The parameter for the OPTICS algorithm. Default is 0.05.
            * min_cluster_size     : The minimum number of samples in a cluster. Default is None.
        '''
        optics = OPTICS(min_samples=min_samples, xi=xi, min_cluster_size=min_cluster_size).fit(X_scaled)
        return optics.labels_

@register_cluster_method('hdbscan')
class HDBSCANMethod(ClusterMethod):
    def cluster(self, X_scaled, min_cluster_size=5, allow_single_cluster=False, **kwargs):
        '''
        Cluster the earthquakes on a profile using HDBSCAN.
        HDBSCAN: Hierarchical Density-Based Spatial Clustering of Applications with Noise
        Args:
            * min_cluster_size     : The minimum number of samples in a cluster. Default is 5.
            * allow_single_cluster : Whether to allow a single cluster. Default is False.
        '''
        hdbscan = HDBSCAN(min_cluster_size=min_cluster_size, allow_single_cluster=allow_single_cluster).fit(X_scaled)
        return hdbscan.labels_

@register_cluster_method('kmeans')
class KMeansMethod(ClusterMethod):
    def cluster(self, X_scaled, n_clusters=2, **kwargs):
        '''
        Cluster the earthquakes on a profile using KMeans.
        KMeans: A popular centroid-based clustering algorithm.
        Args:
            * n_clusters : The number of clusters to form. Default is 2.
        '''
        kmeans = KMeans(n_clusters=n_clusters).fit(X_scaled)
        return kmeans.labels_

@register_cluster_method('agglomerative')
class AgglomerativeMethod(ClusterMethod):
    def cluster(self, X_scaled, n_clusters=2, **kwargs):
        '''
        Cluster the earthquakes on a profile using Agglomerative Clustering.
        Agglomerative Clustering: A hierarchical clustering method using a bottom-up approach.
        Args:
            * n_clusters : The number of clusters to form. Default is 2.
        '''
        agglomerative = AgglomerativeClustering(n_clusters=n_clusters).fit(X_scaled)
        return agglomerative.labels_

@register_cluster_method('spectral')
class SpectralMethod(ClusterMethod):
    def cluster(self, X_scaled, n_clusters=2, **kwargs):
        '''
        Cluster the earthquakes on a profile using Spectral Clustering.
        Spectral Clustering: A technique with roots in graph theory, where the approach is used to identify communities of nodes in a graph.
        Args:
            * n_clusters : The number of clusters to form. Default is 2.
        '''
        spectral = SpectralClustering(n_clusters=n_clusters).fit(X_scaled)
        return spectral.labels_

# class SpectralMethod(ClusterMethod):
#     def cluster(self, X_scaled, n_clusters=2, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity='rbf', n_neighbors=10, eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1, kernel_params=None, **kwargs):
#         '''
#         Cluster the earthquakes on a profile using Spectral Clustering.
#         Args:
#             * n_clusters           : The number of clusters to form. Default is 2.
#             * eigen_solver         : The eigenvalue decomposition strategy. Default is None.
#             * random_state         : Determines random number generation for eigen_solver and cluster initialization. Use an int to make the randomness deterministic. Default is None.
#             * n_init               : Number of time the k-means algorithm will be run with different centroid seeds. The final results will be the best output of n_init consecutive runs in terms of inertia. Default is 10.
#             * gamma                : Kernel coefficient for rbf, poly, sigmoid, laplacian and chi2 kernels. Ignored for affinity='nearest_neighbors'. Default is 1.0.
#             * affinity             : How to construct the affinity matrix. Default is 'rbf'.
#             * n_neighbors          : Number of neighbors to use when constructing the affinity matrix using the nearest neighbors method. Ignored for affinity='rbf'. Default is 10.
#             * eigen_tol            : Stopping criterion for eigendecomposition of the Laplacian matrix when eigen_solver='arpack'. Ignored for affinity='nearest_neighbors'. Default is 0.0.
#             * assign_labels        : The strategy to use to assign labels in the embedding space. There are two ways to assign labels after the laplacian embedding. k-means and discretize. Default is 'kmeans'.
#             * degree               : Degree of the polynomial kernel. Ignored by other kernels. Default is 3.
#             * coef0                : Zero coefficient for polynomial and sigmoid kernels. Ignored by other kernels. Default is 1.
#             * kernel_params        : Parameters (keyword arguments) and values for kernel passed as callable object. Ignored by other kernels. Default is None.
#         '''
#         spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver=eigen_solver, random_state=random_state, n_init=n_init, gamma=gamma, affinity=affinity, n_neighbors=n_neighbors, eigen_tol=eigen_tol, assign_labels=assign_labels, degree=degree, coef0=coef0, kernel_params=kernel_params).fit(X_scaled)
#         return spectral.labels_

@register_cluster_method('klemb')
class KLEmbedScanMethod(ClusterMethod):
    def __init__(self, eps, min_pts, ecc_pts, xi=.05):
        self.eps = eps
        self.min_pts = min_pts
        self.ecc_pts = ecc_pts
        self.xi = xi

    def kl_dist(self, x, y):
        diff = np.array([x[0] - y[0], x[1] - y[1]])

        cov1 = np.array([[x[2], x[3]], [x[3], x[4]]])
        inv1 = np.array([[x[5], x[6]], [x[6], x[7]]])
        inv_sqrt1 = np.array([[x[8], x[9]], [x[9], x[10]]])

        cov2 = np.array([[y[2], y[3]], [y[3], y[4]]])
        inv2 = np.array([[y[5], y[6]], [y[6], y[7]]])
        inv_sqrt2 = np.array([[y[8], y[9]], [y[9], y[10]]])

        I = np.eye(2)

        A = .5 * np.linalg.norm(np.dot(np.dot(inv_sqrt2, cov1), inv_sqrt2) - I, 'fro')
        B = .5 * np.linalg.norm(np.dot(np.dot(inv_sqrt1, cov2), inv_sqrt1) - I, 'fro')

        C_value = np.dot(np.dot(diff.T, inv1), diff)
        C_value = max(C_value, 0)  # ensure non-negative
        C = 1 / np.sqrt(2) * np.sqrt(C_value)

        D_value = np.dot(np.dot(diff.T, inv2), diff)
        D_value = max(D_value, 0)  # ensure non-negative
        D = 1 / np.sqrt(2) * np.sqrt(D_value)

        return A + B + C + D

    def cluster(self, dataset, **kwargs):
        kd = KDTree(dataset)

        embeddings = []
        for p in range(len(dataset)):
            cluster = kd.query(x=dataset[p], k=self.ecc_pts)[1].tolist()
            cov = np.cov(np.array([dataset[k] for k in cluster]), rowvar=False)
            cov /= max(np.linalg.eig(cov)[0])
            mean = np.mean(np.array([dataset[k] for k in cluster]), axis=0)
            inv = 1 / (cov[0, 0] * cov[1, 1] - cov[0, 1] * cov[0, 1]) * \
            np.array([[cov[1, 1], -cov[0, 1]], [-cov[0, 1], cov[0, 0]]])
            inv_sqrt = sqrtm(inv)

            embeddings.append(np.concatenate([mean, cov.flatten(), inv.flatten(), inv_sqrt.flatten()]))
        embeddings = np.array(embeddings)

        dist_func = lambda x, y: self.kl_dist(x, y)

        return OPTICS(min_samples=self.min_pts, metric=dist_func, cluster_method="xi", xi=self.xi).fit(embeddings).labels_
#--------------------------Clustering earthquakes--------------------------#

#--------------------------Extracting linear clusters--------------------------#
class LinearDetectionMethod(ABC):
    @abstractmethod
    def detect(self, cluster_points):
        pass

@register_linear_detection_method('r2')
class R2Method(LinearDetectionMethod):
    def __init__(self, threshold=0.5):
        """
        Initialize the R2Method with a threshold.

        Args:
            threshold (float, optional): The minimum R-squared value to detect a line. 
                                         The higher the threshold, the fewer lines will be detected. 
                                         Recommended values are in the range [0.5, 1.0]. Default is 0.5.
        """
        self.threshold = threshold

    def detect(self, cluster_points):
        """
        Detect if the given points form a linear cluster based on the R-squared value.

        Args:
            cluster_points (numpy.ndarray): The points in the cluster.

        Returns:
            bool: True if the R-squared value is greater than or equal to the threshold, False otherwise.
        """
        slope, intercept, r_value, p_value, std_err = linregress(cluster_points[:, 0], cluster_points[:, 1])
        return r_value**2 >= self.threshold


@register_linear_detection_method('pca')
class PCAMethod(LinearDetectionMethod):
    def __init__(self, threshold=0.90):
        """
        Initialize the PCAMethod with a threshold.

        Args:
            threshold (float, optional): The minimum explained variance ratio to detect a line. 
                                         The higher the threshold, the fewer lines will be detected. 
                                         Recommended values are in the range [0.9, 1.0]. Default is 0.95.
        """
        self.threshold = threshold

    def detect(self, cluster_points):
        """
        Detect if the given points form a linear cluster based on the explained variance ratio.

        Args:
            cluster_points (numpy.ndarray): The points in the cluster.

        Returns:
            bool: True if the explained variance ratio is greater than or equal to the threshold, False otherwise.
        """
        pca = PCA(n_components=1)
        pca.fit(cluster_points)
        explained_variance_ratio = pca.explained_variance_ratio_[0]
        return explained_variance_ratio >= self.threshold


@register_linear_detection_method('hough')
class HoughTransformMethod(LinearDetectionMethod):
    def __init__(self, threshold_ratio=0.2, line_length=10, line_gap=3):
        """
        Initialize the HoughTransformMethod with a threshold ratio, line length and line gap.

        Args:
            threshold_ratio (float, optional): The ratio of the number of points in the cluster 
                                               to set as the threshold for the Hough Transform. 
                                               Default is 0.2.
            line_length (int, optional): The minimum line length. Default is 10.
            line_gap (int, optional): The maximum gap between line segments 
                                      lying on the same line to consider them as a single line. 
                                      Default is 3.
        """
        self.threshold_ratio = threshold_ratio
        self.line_length = line_length
        self.line_gap = line_gap

    def detect(self, cluster_points):
        """
        Detect if the given points form a linear cluster based on the Hough Transform.

        Args:
            cluster_points (numpy.ndarray): The points in the cluster.

        Returns:
            bool: True if lines are detected, False otherwise.
        """
        # Calculate the threshold based on the number of points in the cluster
        threshold = int(len(cluster_points) * self.threshold_ratio)
        # Apply Hough Transform
        lines = probabilistic_hough_line(cluster_points, threshold=threshold, 
                                         line_length=self.line_length, line_gap=self.line_gap)
        # If lines are detected, consider it as linear
        return len(lines) > 0

@register_linear_detection_method('ransac')
class RANSACMethod(LinearDetectionMethod):
    def __init__(self, residual_threshold=1.0):
        """
        Initialize the RANSACMethod with a residual threshold.

        Args:
            residual_threshold (float, optional): Maximum distance for a data point to be classified as an inlier. 
                                                  The higher the threshold, the more points will be classified as inliers. 
                                                  Recommended values are in the range [0.1, 10.0]. Default is 1.0.
        """
        self.residual_threshold = residual_threshold

    def detect(self, cluster_points):
        # Apply RANSAC
        ransac = RANSACRegressor(residual_threshold=self.residual_threshold)
        ransac.fit(cluster_points[:, 0].reshape(-1, 1), cluster_points[:, 1])
        inlier_mask = ransac.inlier_mask_
        # If the majority of points are inliers, consider it as linear
        return np.sum(inlier_mask) / len(cluster_points) > 0.5
#--------------------------Extracting linear clusters--------------------------#