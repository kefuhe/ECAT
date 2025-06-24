import matplotlib.pyplot as plt
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter
from matplotlib.ticker import AutoLocator
from ..plottools import sci_plot_style

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import block_diag
import scipy
# SMC and MPI 
from numba import njit
import time
import yaml
import os 
import glob

from csi import gps, insar
from .BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches as relocfault
from .multifaultsolve_boundLSE import multifaultsolve_boundLSE as multifaultsolve
from .SMC_MPI import SMC_samples_parallel_mpi
from .bayesian_config import BayesianMultiFaultsInversionConfig
import h5py
from mpi4py import MPI
import logging
import time
# 
from collections import namedtuple
import os
from typing import List
from numpy import ndarray

from .bayesian_utils import det_of_laplace_smooth_lu, logpdf_multivariate_normal
from . import lsqlin
# using the C++ backend
os.environ['CUTDE_USE_BACKEND'] = 'cpp' # cuda, cpp, or opencl

def log_time(start_time, end_time, message, log_enabled):
    if log_enabled:
        print(f"{message}: {end_time - start_time} seconds")

@njit
def compute_log_prior(samples: ndarray, lb: ndarray, ub: ndarray) -> float:
    if np.any((samples < lb) | (samples > ub)):
        return -np.inf
    else:
        return 0.0

@njit
def compute_magnitude_log_prior(slip_components, moment_magnitude_threshold, 
                                patch_areas, shear_modulus, magnitude_tolerance):
    num_patches = len(patch_areas)
    
    if len(slip_components) < 2 * num_patches:  # If only one component of slip (dip or strikeslip)
        slip = slip_components[:num_patches]
        np.abs(slip, out=slip)
    else:  # If both components of slip (dip or strikeslip)
        slip = slip_components[:2 * num_patches].reshape(2, num_patches)
        slip = np.sqrt(np.sum(slip**2, axis=0))
    
    moment = np.sum(shear_modulus * patch_areas * slip)
    moment_magnitude = 2.0 / 3.0 * (np.log(moment) - 9.1)
    
    magnitude_difference = np.abs(moment_magnitude_threshold - moment_magnitude)
    
    if magnitude_difference > magnitude_tolerance:
        return -np.inf
    else:
        return 0

@njit
def compute_data_log_likelihood(G: ndarray, samples: ndarray, observations: ndarray, 
                                inv_cov: ndarray, log_cov_det: float) -> float:
    simulations = np.dot(G, samples)
    data_log_likelihood = logpdf_multivariate_normal(observations, simulations, 
                                                     inv_cov, log_cov_det)
    return data_log_likelihood

# @njit
# def compute_smooth_log_likelihood(GL: ndarray, samples: ndarray, alpha: float) -> float:
#     size = GL.shape[0]
#     LS = np.dot(GL, samples)
#     LS_t_LS = np.sum(LS ** 2)
#     alpha_2 = alpha ** 2
#     # Calculate the log determinant of the Laplacian matrix
#     smooth_log_likelihood = -0.5 * size * np.log(alpha_2) - LS_t_LS / (2 * alpha_2)
#     return smooth_log_likelihood

@njit
def compute_smooth_log_likelihood(GL: ndarray, samples: ndarray, alpha: ndarray) -> float:
    """
    Compute the smooth log-likelihood.

    Original likelihood formula:
    L = (2π)^(-n/2) * |Σ|^(-1/2) * exp(-0.5 * x^T * Σ^(-1) * x)

    Log-likelihood formula:
    log(L) = -0.5 * log(|Σ|) - 0.5 * x^T * Σ^(-1) * x - 0.5 * n * log(2π)

    Parameters:
    GL (ndarray): Laplacian matrix.
    samples (ndarray): Sample data.
    alpha (ndarray): Regularization parameter vector.

    Returns:
    float: Smooth log-likelihood.
    """
    LS = np.dot(GL, samples)
    alpha_2 = alpha ** 2
    # Calculate the log determinant of the Laplacian matrix for each alpha
    log_det_cov = np.sum(np.log(alpha_2))
    inv_cov = 1 / alpha_2
    # LS^T * inv_cov * LS
    LS_t_inv_cov_LS = np.sum(LS ** 2 * inv_cov)
    smooth_log_likelihood = -0.5 * log_det_cov - 0.5 * LS_t_inv_cov_LS
    return smooth_log_likelihood

# @njit
# def compute_logpdf_slip_F_J_invCov(G: ndarray, GL: np.ndarray, inv_cov: ndarray, log_cov_det: float, sigmas_2: np.ndarray, alpha_2: float) -> float:
#     inv_cov = np.zeros((len(self.observations), len(self.observations)))
#     log_det = 0.0
#     st = 0
#     ed = 0
#     for ind, idataname in enumerate(self.datanames):
#         ed += len(self.obs_dict[idataname])
#         isigmas_2 = sigmas[ind]**2
#         inv_cov[st:ed, st:ed] = np.divide(self.inv_covs[idataname], isigmas_2)
#         log_det += len(self.obs_dict[idataname]) * np.log(isigmas_2) # self.logdets[idataname] + 
#         st = ed

def compute_smooth_log_likelihood_csr(GL: csr_matrix, samples: ndarray, alpha: float) -> float:
    size = GL.shape[0]
    GL_dense = GL.toarray()  # Transform to dense matrix
    LTL = GL_dense.transpose().dot(GL_dense)  # Calculate L^T * L
    # LTL += 1e-5 * np.eye(LTL.shape[0])  # Add a small value to the diagonal
    LTL = csr_matrix(LTL)  # Transform to csr_matrix
    LS = GL.dot(samples)
    LS_t_LS = np.sum(LS ** 2)
    log_det_LTL = np.log(det_of_laplace_smooth_lu(LTL))
    alpha_2 = alpha ** 2
    smooth_log_likelihood = -0.5 * size * np.log(alpha_2) - LS_t_LS / (2 * alpha_2) + 1/2 * log_det_LTL
    return smooth_log_likelihood

@njit
def compute_log_posterior(samples: ndarray, G: ndarray, observations: ndarray, lb, ub, 
                          inv_cov: ndarray, log_cov_det: float, alpha: float, GL: csr_matrix) -> float:
    log_prior = compute_log_prior(samples, lb, ub)
    if log_prior == -np.inf:
        return -np.inf
    else:
        data_log_likelihood = compute_data_log_likelihood(G, samples, observations, inv_cov, log_cov_det)
        smooth_log_likelihood = compute_smooth_log_likelihood(GL, samples, alpha)
        return log_prior + data_log_likelihood + smooth_log_likelihood
    
@njit
def compute_magnitude_log_posterior(samples: ndarray, G: ndarray, observations: ndarray, lb, ub, 
                                    inv_cov: ndarray, log_cov_det: float, moment_magnitude_threshold, 
                                    patch_areas, shear_modulus, magnitude_tolerance, sigma: float, alpha: float, GL: csr_matrix) -> float:
    log_prior = compute_log_prior(samples, lb, ub)
    if log_prior == -np.inf:
        return -np.inf
    else:
        log_magnitude_prior = compute_magnitude_log_prior(samples, moment_magnitude_threshold, 
                                                          patch_areas, shear_modulus, magnitude_tolerance)
        if log_magnitude_prior == -np.inf:
            return -np.inf
        else:
            data_log_likelihood = compute_data_log_likelihood(G, samples, observations, inv_cov, log_cov_det)
            smooth_log_likelihood = compute_smooth_log_likelihood(GL, samples, alpha)
            return log_prior + log_magnitude_prior + data_log_likelihood + smooth_log_likelihood

def make_target_for_sampler(Gs: List[ndarray], observations: List[ndarray], lb, ub, 
                            inv_covs: List[ndarray], log_dets: List[float], sigmas: List[float], alpha: float, GL: csr_matrix):
    @njit
    def target(samples):
        return compute_log_posterior(samples, Gs, observations, lb, ub, inv_covs, log_dets, sigmas, alpha, GL)
    return target

def make_magnitude_target_for_sampler(Gs: List[ndarray], observations: List[ndarray], lb, ub, moment_magnitude_threshold, 
                                      patch_areas, shear_modulus, magnitude_tolerance, 
                                      inv_covs: List[ndarray], log_dets: List[float], sigmas: List[float], alpha: float, GL: csr_matrix):
    def target(samples):
        return compute_magnitude_log_posterior(samples, Gs, observations, lb, ub, moment_magnitude_threshold, 
                                               patch_areas, shear_modulus, magnitude_tolerance, 
                                               inv_covs, log_dets, sigmas, alpha, GL)
    return target


NT1 = namedtuple('NT1', 'N Neff target LB UB')
# tuple object for the samples
NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl')

    
class BoundsManager:
    """
    Class to manage the bounds for all parameters in the inversion.
    """
    def __init__(self, inversion_instance):
        """
        Initialize the BoundsManager with the inversion instance.
        """
        self.config = inversion_instance.config
        self.mcmc_samples = inversion_instance.mcmc_samples
        self.multifaults = inversion_instance.multifaults
        self.inversion_instance = inversion_instance
        self.lb = np.ones(self.mcmc_samples) * np.nan
        self.ub = np.ones(self.mcmc_samples) * np.nan
        self.geometry_positions = inversion_instance.geometry_positions
        self.sigmas_position = inversion_instance.sigmas_position
        self.alpha_position = inversion_instance.alpha_position
        self.slip_positions = inversion_instance.slip_positions.copy()
        self.poly_positions = inversion_instance.poly_positions.copy() 
        # Transfer slip_positions and poly_positions to the correct MCMC sampling position.
        if self.config.slip_sampling_mode == 'rake_fixed':
            total_half = 0
            for ifault in self.inversion_instance.multifaults.faults:
                lb_slip, ub_slip = self.slip_positions[ifault.name]
                lb_poly, ub_poly = self.poly_positions[ifault.name]
                self.slip_positions[ifault.name] = [lb_slip-total_half, ub_slip-total_half -len(ifault.patch)]
                total_half += len(ifault.patch)
                self.poly_positions[ifault.name] = [lb_poly-total_half, ub_poly-total_half]
        self.bounds = {
            'geometry': None,
            'poly': None,
            'strikeslip': None,
            'dipslip': None,
            'rake_angle': None,
            'slip_magnitude': None,
            'alpha': None,
            'sigmas': None
        }

    def set_default_bounds(self, lb=None, ub=None):
        """
        Set the default bounds for all parameters.
        """
        lb_to_use = lb
        ub_to_use = ub

        if lb_to_use is not None:
            self.lb[np.isnan(self.lb)] = lb_to_use
        if ub_to_use is not None:
            self.ub[np.isnan(self.ub)] = ub_to_use
    
    def update_bounds_for_all_mcmc_parameters(self, geometry=None, slip_magnitude=None, rake_angle=None, strikeslip=None, dipslip=None, poly=None, sigmas=None, alpha=None):
        """
        Update the bounds for all MCMC parameters.

        Parameters:
        geometry (dict): Dictionary containing the bounds for geometry parameters.
        slip_magnitude (dict): Dictionary containing the bounds for slip magnitude.
        rake_angle (dict): Dictionary containing the bounds for rake angle.
        strikeslip (dict): Dictionary containing the bounds for strike-slip component.
        dipslip (dict): Dictionary containing the bounds for dip-slip component.
        poly (dict): Dictionary containing the bounds for polynomial parameters.
        sigmas (list): List containing the bounds for sigmas.
        alpha (list): List containing the bounds for alpha.
        """
        self.update_bounds_for_all_faults(geometry, slip_magnitude, rake_angle, strikeslip, dipslip, poly)
        self.update_bounds_for_sigmas(sigmas)
        self.update_bounds_for_alpha(alpha)
    
    def update_bounds_from_config(self, config_file=None, encoding='utf-8'):
        """
        Update parameter bounds from a configuration file.
        """
        if config_file is not None:
            try:
                with open(config_file, 'r', encoding=encoding) as file:
                    self.bounds_config = yaml.safe_load(file)
            except FileNotFoundError:
                print(f"Configuration file {config_file} not found.")
                return
            except yaml.YAMLError as e:
                print(f"Error parsing configuration file: {e}")
                return
    
        lb = self.bounds_config.get('lb', None)
        ub = self.bounds_config.get('ub', None)
        if lb is not None and ub is not None and lb > ub:
            print("Global lower bound should be less than upper bound.")
            return
    
        self.set_default_bounds(lb, ub)
        geometry = self.bounds_config.get('geometry', None)
        slip_magnitude = self.bounds_config.get('slip_magnitude', None)
        rake_angle = self.bounds_config.get('rake_angle', None)
        strikeslip = self.bounds_config.get('strikeslip', None)
        dipslip = self.bounds_config.get('dipslip', None)
        poly = self.bounds_config.get('poly', None)
        sigmas = self.bounds_config.get('sigmas', None)
        alpha = self.bounds_config.get('alpha', None)
        self.update_bounds_for_all_mcmc_parameters(geometry, slip_magnitude, rake_angle, 
                                                   strikeslip, dipslip, poly, sigmas, alpha)
    
    def update_bounds_for_all_faults(self, geometry=None, slip_magnitude=None, rake_angle=None, strikeslip=None, dipslip=None, poly=None):
        """
        Update the bounds for all faults based on the slip_sampling_mode.
        Parameters slip_magnitude, rake_angle, strikeslip, dipslip are set to None by default and are required based on the mode.
        """
        for fault_name in self.config.faults:
            if fault_name == 'defaults':
                continue
            if self.config.faults[fault_name]['geometry']['update'] and geometry.get(fault_name, None) is not None:
                self.update_bounds_for_geometry(fault_name, geometry[fault_name])
            
            # Assert checks to ensure the necessary parameters are provided based on the slip_sampling_mode
            if self.config.slip_sampling_mode == "rake_fixed":
                if slip_magnitude is not None:
                    self.update_slip_bounds_based_on_mode(fault_name, slip_magnitude.get(fault_name, None))
            elif self.config.slip_sampling_mode == "ss_ds":
                if strikeslip is not None or dipslip is not None:
                    istrikeslip = strikeslip.get(fault_name, None) if strikeslip is not None else None
                    idipslip = dipslip.get(fault_name, None) if dipslip is not None else None
                    if istrikeslip is not None or idipslip is not None:
                        self.update_slip_bounds_based_on_mode(fault_name, None, None, istrikeslip, idipslip)
            elif self.config.slip_sampling_mode == "magnitude_rake":
                if slip_magnitude is not None or rake_angle is not None:
                    islip_magnitude = slip_magnitude.get(fault_name, None) if slip_magnitude is not None else None
                    irake_angle = rake_angle.get(fault_name, None) if rake_angle is not None else None
                    if islip_magnitude is not None or irake_angle is not None:
                        self.update_slip_bounds_based_on_mode(fault_name, islip_magnitude, irake_angle)
            
            if poly is not None and poly.get(fault_name, None) is not None:
                self.update_bounds_for_poly(fault_name, poly[fault_name])
    
    def update_slip_bounds_based_on_mode(self, fault_name, slip_magnitude=None, rake_angle=None, strikeslip=None, dipslip=None):
        """
        根据slip_sampling_mode更新滑动参数的边界。
        """
        slip_start, slip_end = self.slip_positions[fault_name]
        slip_half = (slip_end + slip_start) // 2

        if self.config.slip_sampling_mode == 'rake_fixed':
            if slip_magnitude is not None:
                # update self.bounds
                if self.bounds['slip_magnitude'] is None:
                    self.bounds['slip_magnitude'] = {}
                self.bounds['slip_magnitude'][fault_name] = (slip_magnitude[0], slip_magnitude[1])
                # update slip bounds
                self.lb[slip_start:slip_end] = self.bounds['slip_magnitude'][fault_name][0]
                self.ub[slip_start:slip_end] = self.bounds['slip_magnitude'][fault_name][1]
        elif self.config.slip_sampling_mode == 'magnitude_rake':
            assert slip_magnitude is not None or rake_angle is not None, "Either slip_magnitude or rake_angle must not be None in magnitude_rake mode"
            if slip_magnitude is not None:
                # update self.bounds
                if self.bounds['slip_magnitude'] is None:
                    self.bounds['slip_magnitude'] = {}
                self.bounds['slip_magnitude'][fault_name] = (slip_magnitude[0], slip_magnitude[1])
                # update slip bounds
                self.lb[slip_start:slip_half] = self.bounds['slip_magnitude'][fault_name][0]
                self.ub[slip_start:slip_half] = self.bounds['slip_magnitude'][fault_name][1]
            if rake_angle is not None:
                # update self.bounds
                if self.bounds['rake_angle'] is None:
                    self.bounds['rake_angle'] = {}
                self.bounds['rake_angle'][fault_name] = (rake_angle[0], rake_angle[1])
                # update slip bounds
                self.lb[slip_half:slip_end] = self.bounds['rake_angle'][fault_name][0]
                self.ub[slip_half:slip_end] = self.bounds['rake_angle'][fault_name][1]
        else:  # 'ss_ds' mode
            assert strikeslip is not None or dipslip is not None, "Either strikeslip or dipslip must not be None in ss_ds mode"
            if strikeslip is not None:
                # update self.bounds
                if self.bounds['strikeslip'] is None:
                    self.bounds['strikeslip'] = {}
                self.bounds['strikeslip'][fault_name] = (strikeslip[0], strikeslip[1])
                # update slip bounds
                self.lb[slip_start:slip_half] = self.bounds['strikeslip'][fault_name][0]
                self.ub[slip_start:slip_half] = self.bounds['strikeslip'][fault_name][1]
            if dipslip is not None:
                # update self.bounds
                if self.bounds['dipslip'] is None:
                    self.bounds['dipslip'] = {}
                self.bounds['dipslip'][fault_name] = (dipslip[0], dipslip[1])
                # update slip bounds
                self.lb[slip_half:slip_end] = self.bounds['dipslip'][fault_name][0]
                self.ub[slip_half:slip_end] = self.bounds['dipslip'][fault_name][1]
    
    def get_bounds_for_F_J(self):
        '''
        Get the bounds for the F_J samples mode.
        '''
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        return self.lb[linear_sample_start:], self.ub[linear_sample_start:]

    def get_bounds_for_geometry_hyperparameters(self):
        '''
        Get the bounds for the geometry hyperparameters.
        '''
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        return self.lb[:linear_sample_start], self.ub[:linear_sample_start]

    def get_inequality_constraints_for_rake_angle(self, rake_angle=None):
        '''
        Generate linear constraints (A, b) for the rake angle range based on strike-slip and dip-slip components.
        Satify the following constraints:
         A * x <= b, where b = 0 vector
         rake_angle: (dict) {fault_name: (lower_bound, upper_bound)}
         rake: rake: (-180, 180) or (0, 360), anti-clockwise is positive
         rake_ub - rake_lb <= 180
        '''
        if rake_angle is None:
            rake_angle = self.bounds_config['rake_angle']
        else:
            self.bounds_config['rake_angle'] = rake_angle
        
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        nlinear = self.mcmc_samples - linear_sample_start
        npatch_list = [len(fault.patch) for fault in self.multifaults.faults]
        npatch = int(np.sum(npatch_list))
        A = np.zeros((2 * npatch, nlinear))
        b = np.zeros(2 * npatch)

        patch_count = 0
        for ifault in self.multifaults.faults:
            start, end = self.slip_positions[ifault.name]
            start -= linear_sample_start
            end -= linear_sample_start
            half = (start + end) // 2
            # Get the rake angle bounds
            rake_start, rake_end = rake_angle[ifault.name]
            # Generate the linear constraints
            inpatch = len(ifault.patch)
            for i in range(inpatch):
                # cross product of slip and rake,
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) < 0
                A[patch_count + i, start+i] = np.sin(np.deg2rad(rake_start))
                A[patch_count + i, half+i] = -np.cos(np.deg2rad(rake_start))
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) > 0
                A[patch_count + inpatch+i, start+i] = -np.sin(np.deg2rad(rake_end))
                A[patch_count + inpatch+i, half+i] =   np.cos(np.deg2rad(rake_end))
            patch_count += inpatch
        
        return A, b

    def get_equality_constraints_for_fixed_rake(self, fixed_rake):
        '''
        Generate linear equality constraints (A_eq, b_eq) for a fixed rake angle based on strike-slip and dip-slip components.
        Satisfy the following constraints:
         A_eq * x = b_eq, where b_eq = 0 vector
         fixed_rake: (dict) {fault_name: fixed_rake_angle}
         rake: fixed value
        '''
        linear_sample_start = self.inversion_instance.linear_sample_start_position
        nlinear = self.mcmc_samples - linear_sample_start
        npatch_list = [len(fault.patch) for fault in self.multifaults.faults]
        npatch = int(np.sum(npatch_list))
        A_eq = np.zeros((npatch, nlinear))
        b_eq = np.zeros(npatch)

        patch_count = 0
        for ifault in self.multifaults.faults:
            start, end = self.slip_positions[ifault.name]
            start -= linear_sample_start
            end -= linear_sample_start
            half = (start + end) // 2
            # Get the fixed rake angle
            rake = fixed_rake[ifault.name]
            # Generate the linear constraints
            inpatch = len(ifault.patch)
            for i in range(inpatch):
                # cross product of slip and rake,
                # x: (ss, ds), y: (cos, sin) of rake with x.cross(y) = z, z = (ss*sin(rake) - ds*cos(rake)) = 0
                A_eq[patch_count + i, start+i] = np.sin(np.deg2rad(rake))
                A_eq[patch_count + i, half+i] = -np.cos(np.deg2rad(rake))
            patch_count += inpatch
        
        return A_eq, b_eq

    def update_bounds_for_sigmas(self, bounds):
        if self.config.sigmas['update'] and bounds is not None:
            start, end = self.sigmas_position
            self.lb[start:end] = bounds[0]
            self.ub[start:end] = bounds[1]
            # update self.bounds
            self.bounds['sigmas'] = bounds
    
    def update_bounds_for_alpha(self, bounds):
        if self.config.alpha['update'] and bounds is not None:
            start, end = self.alpha_position
            self.lb[start:end] = bounds[0]
            self.ub[start:end] = bounds[1]
            # update self.bounds
            self.bounds['alpha'] = bounds
    
    def update_bounds_for_geometry(self, fault_name, bounds):
        if self.config.faults[fault_name]['geometry']['update']:
            start, end = self.config.faults[fault_name]['geometry']['sample_positions']
            self.lb[start:end] = bounds[0]
            self.ub[start:end] = bounds[1]
            # update self.bounds
            if self.bounds['geometry'] is None:
                self.bounds['geometry'] = {}
            self.bounds['geometry'][fault_name] = bounds

    def update_bounds_for_slip(self, fault_name, bounds):
        start, end = self.slip_positions[fault_name]
        self.lb[start:end] = bounds[0]
        self.ub[start:end] = bounds[1]
        half = (start + end) // 2
        # update self.bounds
        if self.config.slip_sampling_mode == 'rake_fixed':
            if self.bounds['slip_magnitude'] is None:
                self.bounds['slip_magnitude'] = {}
            self.bounds['slip_magnitude'][fault_name] = bounds
        elif self.config.slip_sampling_mode == 'magnitude_rake':
            if self.bounds['slip_magnitude'] is None:
                self.bounds['slip_magnitude'] = {}
            if self.bounds['rake_angle'] is None:
                self.bounds['rake_angle'] = {}
            self.bounds['slip_magnitude'][fault_name] = [self.lb[start:half], self.ub[start:half]]
            self.bounds['rake_angle'][fault_name] = [self.lb[half:end], self.ub[half:end]]
        else:  # 'ss_ds' mode
            if self.bounds['strikeslip'] is None:
                self.bounds['strikeslip'] = {}
            if self.bounds['dipslip'] is None:
                self.bounds['dipslip'] = {}
            self.bounds['strikeslip'][fault_name] = [self.lb[start:half], self.ub[start:half]]
            self.bounds['dipslip'][fault_name] = [self.lb[half:end], self.ub[half:end]]
    
    def update_bounds_for_poly(self, fault_name, bounds):
        start, end = self.poly_positions[fault_name]
        self.lb[start:end] = bounds[0]
        self.ub[start:end] = bounds[1]
        # update self.bounds
        if self.bounds['poly'] is None:
            self.bounds['poly'] = {}
        self.bounds['poly'][fault_name] = bounds


class BayesianMultiFaultsInversion:
    def __init__(self, config="default_config.yml", multifaults=None, geodata=None, faults_list=None, gfmethods=None, 
                 bounds_config='bounds_config.yml', verbose=True):
        if isinstance(config, str):
            assert geodata is not None, "geodata must be provided when config is a file"
            self.config = BayesianMultiFaultsInversionConfig(config, multifaults=multifaults, geodata=geodata, faults_list=faults_list, gfmethods=gfmethods, verbose=verbose)
        else:
            self.config = config

        self.update_config(self.config)
        self._initialize_bounds(bounds_config)

    def update_config(self, config):
        self.config = config
        self._update_faults()
        self._calculate_parameters()

    def _update_faults(self):
        # Update the faults based on the configuration parameters and method parameters for each fault 
        for fault_name, fault_config in self.config.faults.items():
            if fault_name != 'default':
                # Update Green's functions
                self.multifaults.update_GFs(fault_names=[fault_name], **fault_config['method_parameters']['update_GFs'])
                # Update Laplacian
                self.multifaults.update_Laplacian(fault_names=[fault_name], **fault_config['method_parameters']['update_Laplacian'])

    def _calculate_parameters(self):
        self.Gs = {fault.name: fault.Gassembled for fault in self.multifaults.faults}
        self.inv_covs, self.chol_decomps, self.logdets = self.multifaults.compute_data_inv_covs_and_logdets(self.geodata)
        self.patch_areas = self.multifaults.compute_fault_areas()
        self.GLs = self.multifaults.GLs
        self.GL_combined = block_diag([fault.GL for fault in self.multifaults.faults]).toarray()
        self.calculate_sigmas_alpha_positions()
        self.calculate_geometry_positions()
        self.calculate_slip_and_poly_positions()
        self.calculate_linear_sample_start_position()
        self.calculate_sample_slip_only_positions()
        self.combine_GL_poly()

    def _initialize_bounds(self, bounds_config='bounds_config.yml'):
        self.bounds_manger = BoundsManager(self)
        try:
            # Set the bounds for the parameters from bounds_config.yml
            self.set_parameter_bounds_from_config(config_file=bounds_config, encoding='utf-8')
        except FileNotFoundError:
            print(f"Bounds configuration file '{bounds_config}' not found.")
        except Exception as e:
            print(f"Error setting bounds from config file: {e}")
        
        linear_sample_start = self.linear_sample_start_position
        self._slip_poly_lb = self.bounds_manger.lb[linear_sample_start:]
        self._slip_poly_ub = self.bounds_manger.ub[linear_sample_start:]
        self._hyper_lb = self.bounds_manger.lb[:linear_sample_start]
        self._hyper_ub = self.bounds_manger.ub[:linear_sample_start]

    @property
    def slip_poly_lb(self):
        return self._slip_poly_lb
    
    @property
    def slip_poly_ub(self):
        return self._slip_poly_ub
    
    @property
    def hyper_lb(self):
        return self._hyper_lb
    
    @property
    def hyper_ub(self):
        return self._hyper_ub

    @classmethod
    def from_config(cls, config: BayesianMultiFaultsInversionConfig):
        return cls(config)

    @classmethod
    def from_file(cls, config_file: str):
        config = BayesianMultiFaultsInversionConfig.from_file(config_file)
        return cls(config)

    @classmethod
    def from_parameters(cls, **kwargs):
        config = BayesianMultiFaultsInversionConfig(**kwargs)
        return cls(config)

    def walk(self, nchains=None, chain_length=None, samples=None, magprior=True, comm=None, filename='samples_smc.h5',
             save_every=1, save_at_interval=False, save_at_final=True, covariance_epsilon=1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0,
             sliplb=None, slipub=None, rake_angle=None, rake_sigma=None, rake_range=None, magposteriors=False,
             log_enabled=False, decay_rate=0.1, run_bayesian=True, **kwargs):
        """
        General entry point for SMC sampling, dispatching to the appropriate method based on the bayesian_sampling_mode.
    
        Parameters:
        nchains (int): Number of chains for the SMC sampling. Default is 100.
        chain_length (int): Length of each chain. Default is 50.
        samples (array): Initial samples for the SMC sampling. If None, samples are generated uniformly between the lower and upper bounds.
        magprior (bool): If True, use magnitude prior for generating samples. Default is True.
        comm (MPI.Comm): MPI communicator. If None, MPI.COMM_WORLD is used.
        filename (str): Name of the file where the final samples are saved. Default is 'samples_smc.h5'.
        save_every (int): Frequency at which the samples are saved. Default is 1.
        save_at_interval (bool): If True, save samples at regular intervals. Default is False.
        save_at_final (bool): If True, save samples at the end of the walk. Default is True.
        covariance_epsilon (float): Epsilon value for the covariance matrix. Default is 1e-6.
        amh_a (float): Parameter 'a' for the Adaptive Metropolis-Hastings algorithm. Default is 1.0/9.0.
        amh_b (float): Parameter 'b' for the Adaptive Metropolis-Hastings algorithm. Default is 8.0/9.0.
        sliplb (dict): Lower bounds for each fault. If None, use lb in self.bounds_manger.
        slipub (dict): Upper bounds for each fault. If None, use ub in self.bounds_manger.
        rake_angle (float): Rake angle in degrees. Required if mode is 'ss_ds'.
        rake_sigma (float): Standard deviation of rake angle. Required if mode is 'ss_ds'.
        rake_range (tuple): Lower and upper bounds of rake angle. Required if mode is 'ss_ds'.
        magposteriors (bool): If True, use magnitude posteriors. Default is False.
        log_enabled (bool): If True, enable logging. Default is False.
        decay_rate (float): Decay rate for magnitude posteriors. Default is 0.1.
        run_bayesian (bool): If True, run the Bayesian process. Default is True.
        **kwargs: Additional keyword arguments for specific methods.
    
        Returns:
        final (NT2): A named tuple containing the final samples, their posterior values, beta, stage, and None for acceptance and swap.
        """
        mode = self.config.bayesian_sampling_mode
    
        if mode == 'SMC_F_J':
            return self.walk_F_J(nchains=nchains, chain_length=chain_length, samples=samples, comm=comm, filename=filename,
                                 save_every=save_every, save_at_interval=save_at_interval, save_at_final=save_at_final,
                                 covariance_epsilon=covariance_epsilon, amh_a=amh_a, amh_b=amh_b, log_enabled=log_enabled,
                                 decay_rate=decay_rate, run_bayesian=run_bayesian, **kwargs)
        elif mode == 'FULLSMC':
            return self.walk_smc(nchains=nchains, chain_length=chain_length, samples=samples, magprior=magprior, comm=comm,
                                 filename=filename, save_every=save_every, save_at_interval=save_at_interval,
                                 save_at_final=save_at_final, covariance_epsilon=covariance_epsilon, amh_a=amh_a, amh_b=amh_b,
                                 sliplb=sliplb, slipub=slipub, rake_angle=rake_angle, rake_sigma=rake_sigma, rake_range=rake_range,
                                 magposteriors=magposteriors, log_enabled=log_enabled, decay_rate=decay_rate, run_bayesian=run_bayesian, **kwargs)
        else:
            raise ValueError(f"Unknown bayesian_sampling_mode: {mode}")

    def walk_smc(self, nchains=None, chain_length=None, samples=None, magprior=True, comm=None, filename='samples_smc.h5',
                 save_every=1, save_at_interval=False, save_at_final=True, covariance_epsilon=1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0,
                 sliplb=None, slipub=None, rake_angle=None, rake_sigma=None, rake_range=None, magposteriors=False,
                 log_enabled=False, decay_rate=0.1, run_bayesian=True):
        """
        Perform a Sequential Monte Carlo (SMC) sampling walk.
    
        Parameters:
        nchains (int): Number of chains for the SMC sampling. Default is 100.
        chain_length (int): Length of each chain. Default is 50.
        samples (array): Initial samples for the SMC sampling. If None, samples are generated uniformly between the lower and upper bounds.
        magprior (bool): If True, use magnitude prior for generating samples. Default is True.
        comm (MPI.Comm): MPI communicator. If None, MPI.COMM_WORLD is used.
        filename (str): Name of the file where the final samples are saved. Default is 'samples_smc.h5'.
        save_every (int): Frequency at which the samples are saved. Default is 1.
        save_at_interval (bool): If True, save samples at regular intervals. Default is False.
        save_at_final (bool): If True, save samples at the end of the walk. Default is True.
        covariance_epsilon (float): Epsilon value for the covariance matrix. Default is 1e-6.
        amh_a (float): Parameter 'a' for the Adaptive Metropolis-Hastings algorithm. Default is 1.0/9.0.
        amh_b (float): Parameter 'b' for the Adaptive Metropolis-Hastings algorithm. Default is 8.0/9.0.
        sliplb (dict): Lower bounds for each fault. If None, use lb in self.bounds_manger.
        slipub (dict): Upper bounds for each fault. If None, use ub in self.bounds_manger.
        rake_angle (float): Rake angle in degrees. Required if mode is 'ss_ds'.
        rake_sigma (float): Standard deviation of rake angle. Required if mode is 'ss_ds'.
        rake_range (tuple): Lower and upper bounds of rake angle. Required if mode is 'ss_ds'.
        magposteriors (bool): If True, use magnitude posteriors. Default is False.
        log_enabled (bool): If True, enable logging. Default is False.
        decay_rate (float): Decay rate for magnitude posteriors. Default is 0.1.
        run_bayesian (bool): If True, run the Bayesian process. Default is True.
    
        Returns:
        final (NT2): A named tuple containing the final samples, their posterior values, beta, stage, and None for acceptance and swap.
        """
        # Get the MPI rank
        if comm is None:
            comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    
        nchains = nchains if nchains is not None else self.config.nchains
        chain_length = chain_length if chain_length is not None else self.config.chain_length
    
        assert nchains is not None, "Number of chains must be provided in the configuration or as an argument."
        assert chain_length is not None, "Chain length must be provided in the configuration or as an argument."
    
        self.target = self.make_target_for_parallel(log_enabled=log_enabled) if not magposteriors else self.make_magnitude_target_for_parallel(decay_rate=decay_rate, log_enabled=log_enabled)
    
        if not run_bayesian:
            return None
    
        if rank == 0:
            self.print_parameter_discribution()
            # print('Total samples:', self.total_samples)
            # self.print_parameter_positions()
            print('Number of MCMC samples:', self.mcmc_samples)
            self.print_mcmc_parameter_positions()
    
        opt = NT1(nchains, chain_length, self.target, self.lb, self.ub)
    
        if samples is None:
            samples = NT2(None, None, None, None, None, None)
        if samples is None and magprior:
            samples = self.prior_samples_vectorize(self.target, nchains, sliplb=sliplb, slipub=slipub, 
                                                   rake_angle=rake_angle, rake_sigma=rake_sigma, rake_range=rake_range)
    
        if rank == 0:
            print('Starting the loop...', flush=True)
    
        # run the SMC sampling
        final = SMC_samples_parallel_mpi(opt, samples, NT1, NT2, comm, save_at_final, 
                                         save_every, save_at_interval, covariance_epsilon, amh_a, amh_b)
        self.sampler = final
        if rank == 0:
            self.save2h5(final, filename)
            print('Finished the loop.')
        
        return final

    def walk_F_J(self, nchains=None, chain_length=None, samples=None, comm=None, filename='samples_smc.h5',
                 save_every=1, save_at_interval=False, save_at_final=True, covariance_epsilon=1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0,
                 log_enabled=False, A=None, b=None, Aeq=None, beq=None, lb=None, ub=None, x0=None, opts=None, smooth_prior_weight=1.0,
                 magnitude_log_prior=False, decay_rate=0.1, run_bayesian=True):
        """
        Perform a Sequential Monte Carlo (SMC) sampling walk.
    
        Parameters:
        nchains (int): Number of chains for the SMC sampling. Default is 100.
        chain_length (int): Length of each chain. Default is 50.
        samples (array): Initial samples for the SMC sampling. If None, samples are generated uniformly between the lower and upper bounds.
        comm (MPI.Comm): MPI communicator. If None, MPI.COMM_WORLD is used.
        filename (str): Name of the file where the final samples are saved. Default is 'samples_smc.h5'.
        save_every (int): Frequency at which the samples are saved. Default is 1.
        save_at_interval (bool): If True, save samples at regular intervals. Default is False.
        save_at_final (bool): If True, save samples at the end of the walk. Default is True.
        covariance_epsilon (float): Epsilon value for the covariance matrix. Default is 1e-6.
        amh_a (float): Parameter 'a' for the Adaptive Metropolis-Hastings algorithm. Default is 1.0/9.0.
        amh_b (float): Parameter 'b' for the Adaptive Metropolis-Hastings algorithm. Default is 8.0/9.0.
        log_enabled (bool): If True, enable logging. Default is False.
        A (array): Matrix A for the linear constraints.
        b (array): Vector b for the linear constraints.
        Aeq (array): Matrix Aeq for the equality constraints.
        beq (array): Vector beq for the equality constraints.
        lb (array): Lower bounds for the parameters.
        ub (array): Upper bounds for the parameters.
        x0 (array): Initial guess for the parameters.
        opts (dict): Options for the optimization algorithm.
        smooth_prior_weight (float): Weight for the smoothness prior. Default is 1.0.
        magnitude_log_prior (bool): If True, use magnitude log prior. Default is False.
        decay_rate (float): Decay rate for magnitude log prior. Default is 0.1.
        run_bayesian (bool): If True, run the Bayesian process. Default is True.
    
        Returns:
        final (NT2): A named tuple containing the final samples, their posterior values, beta, stage, and None for acceptance and swap.
        """
        # Get the MPI rank
        if comm is None:
            comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
    
        nchains = nchains if nchains is not None else self.config.nchains
        chain_length = chain_length if chain_length is not None else self.config.chain_length
    
        assert nchains is not None, "Number of chains must be provided in the configuration or as an argument."
        assert chain_length is not None, "Chain length must be provided in the configuration or as an argument."
    
        self.target = self.make_F_J_target_for_parallel(log_enabled=log_enabled, A=A, b=b, Aeq=Aeq, beq=beq, 
                                                        lb=lb, ub=ub, x0=x0, opts=opts, smooth_prior_weight=smooth_prior_weight,
                                                        magnitude_log_prior=magnitude_log_prior, decay_rate=decay_rate)
    
        if not run_bayesian:
            return None
    
        if rank == 0:
            self.print_parameter_discribution()
            # print('Total samples:', self.total_samples)
            # self.print_parameter_positions()
            print('Number of MCMC samples:', self.mcmc_samples)
            self.print_mcmc_parameter_positions()
    
        hyper_lb, hyper_ub = self.bounds_manger.get_bounds_for_geometry_hyperparameters()
        opt = NT1(nchains, chain_length, self.target, hyper_lb, hyper_ub) # Use the bounds for the hyperparameters
    
        if samples is None:
            samples = NT2(None, None, None, None, None, None)
    
        if rank == 0:
            print('Starting the loop...', flush=True)
    
        # run the SMC sampling
        final = SMC_samples_parallel_mpi(opt, samples, NT1, NT2, comm, save_at_final, 
                                        save_every, save_at_interval, covariance_epsilon, amh_a, amh_b)
        self.sampler = final
        if rank == 0:
            self.save2h5(final, filename)
            print('Finished the loop.')
        
        return final
    
    def returnModel(self, model='mean', lb=None, ub=None, A=None, b=None, recal_target=False):
        from scipy.stats import gaussian_kde
        if recal_target or not hasattr(self, 'target'):
            if self.config.bayesian_sampling_mode == 'SMC_F_J':
                self.target = self.make_F_J_target_for_parallel(log_enabled=False, A=A, b=b, lb=lb, ub=ub)
            else:
                self.target = self.make_target_for_parallel()
        
        if isinstance(model, str):
            if model == 'mean':
                specs = self.sampler.allsamples.mean(axis=0)
            elif model == 'median':
                specs = np.median(self.sampler.allsamples, axis=0)
            elif model == 'std':
                specs = self.sampler.allsamples.std(axis=0)
            elif model == 'MAP':
                # Assuming 'logposterior' is the key for log posterior values
                max_posterior_index = np.argmax(self.sampler.postval)
                specs = self.sampler.allsamples[max_posterior_index, :]
            elif model == 'max_prob':
                # Find the mode of the distribution for each dimension
                specs = np.zeros(self.sampler.allsamples.shape[1])
                for i in range(self.sampler.allsamples.shape[1]):
                    kde = gaussian_kde(self.sampler.allsamples[:, i])
                    grid = np.linspace(self.sampler.allsamples[:, i].min(), self.sampler.allsamples[:, i].max(), 1000)
                    densities = kde(grid)
                    max_prob_index = np.argmax(densities)
                    specs[i] = grid[max_prob_index]
            else:
                raise ValueError("Invalid model type. Use 'mean', 'median', 'std', 'MAP', or 'max_prob'.")
        elif isinstance(model, (np.ndarray, list)):
            specs = np.array(model)
        else:
            raise ValueError("Model must be a string, a numpy array, or a list.")
        
        # Save the desired model 
        self.model = specs

        # Update the model geometry
        for fault in self.multifaults.faults:
            # print(f"Fault {fault.name}:")
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
            #     print(f"  Geometry positions: {self.config.faults[fault.name]['geometry']['sample_positions']}")
                fault_config = self.config.faults[fault.name]
                # print('specs:', specs)
                self._update_fault_geometry_and_mesh(fault.name, fault_config, specs)
                self._update_fault_GFs_and_Laplacian(fault.name, fault_config)
        
        if self.bayesian_sampling_mode == 'SMC_F_J':
            if isinstance(model, str) and model == 'std':
                mpost = []
                for isample in self.sampler.allsamples:
                    self.target(isample)
                    mpost.append(self.mpost)
                specs_slip_poly = np.std(mpost, axis=0)
                specs_full = np.hstack((specs[:self.linear_sample_start_position], specs_slip_poly))
                self.target(specs[:self.linear_sample_start_position])
            else:
                self.target(specs)
                specs_slip_poly = self.mpost
                specs_full = np.hstack((specs[:self.linear_sample_start_position], specs_slip_poly))
            specs = specs_full
            self.model = specs_full
        else:
            self.G_combined = np.hstack([fault.Gassembled for fault in self.multifaults.faults])
            self.mpost = specs[self.linear_sample_start_position:]
        
        print('Number of data: {}'.format(self.multifaults.Nd))
        print('Number of MCMC parameters: {}'.format(self.mcmc_samples)) # self.multifaults.Np
        print('Parameter Description ----------------------------------')
        # update model slip and poly
        total_half = 0
        for fault in self.multifaults.faults:
            # print('-----------------')
            print(f"Fault {fault.name}:")
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                print(f"  Geometry positions: {self.config.faults[fault.name]['geometry']['sample_positions']}")
            
            slip_start, slip_end = self.slip_positions[fault.name]
            slip_start -= total_half
            slip_end -= total_half
            if self.config.slip_sampling_mode == 'rake_fixed':
                print(f"  Slip positions: [{slip_start}, {slip_start + half}]")
                half = (slip_end - slip_start) // 2
                ss = specs[slip_start:slip_start + half]*np.cos(np.radians(self.config.rake_angle))
                ds = specs[slip_start + half:slip_end]*np.sin(np.radians(self.config.rake_angle))
                fault.slip[:, :2] = np.vstack([ss, ds]).T

                total_half += half
            elif self.config.slip_sampling_mode == 'magnitude_rake':
                half = (slip_end - slip_start) // 2
                print(f"  Slip magnitude positions: [{slip_start}, {slip_start + half}]")
                print(f"  Rake positions: [{slip_start + half}, {slip_end}]")
                slip_mag = specs[slip_start:slip_start + half]
                rake = specs[slip_start + half:slip_end]
                ss = slip_mag*np.cos(np.radians(rake))
                ds = slip_mag*np.sin(np.radians(rake))
                fault.slip[:, :2] = np.vstack([ss, ds]).T
            else:
                print(f"  Slip positions: [{slip_start}, {slip_end}]")
                fault.slip[:, :2] = specs[slip_start:slip_end].reshape(2, -1).T

            poly_start, poly_end = self.poly_positions[fault.name]
            poly_start -= total_half
            poly_end -= total_half
            if poly_start != poly_end:
                print(f"  Poly positions: [{poly_start}, {poly_end}]")
            for i, (key, value) in enumerate(fault.poly.items()):
                if value is not None:
                    fault.polysol[key] = specs[poly_start: poly_start + value]
                    poly_start += value

        if self.config.sigmas['update']:
            sigmas_start, sigmas_end = self.sigmas_position
            print(f"Sigmas position: [{sigmas_start}, {sigmas_end}]")
            self.sigmas = specs[sigmas_start: sigmas_end].tolist()
        if self.config.alpha['update']:
            alpha_start, alpha_end = self.alpha_position
            print(f"Alpha position: [{alpha_start}, {alpha_end}]")
            self.alpha = specs[alpha_start: alpha_end] # .item()
        
        if (not isinstance(model, str)) or (model not in ('std', 'STD', 'Std')):
            # Predict the data and print the RMS and VR
            # Caluculate RMS and VR for the solution and print the results
            rms = np.sqrt(np.mean((np.dot(self.G_combined, self.mpost) - self.observations)**2))
            vr = (1 - np.sum((np.dot(self.G_combined, self.mpost) - self.observations)**2) / np.sum(self.observations**2)) * 100
            # self.combine_GL_poly()
            roughness = np.dot(self.GL_combined_poly, self.mpost)
            roughness = np.sqrt(np.mean(roughness**2))
            print(f'Roughness: {roughness:.4f}, RMS: {rms:.4f}, VR: {vr:.2f}%')

            for idata, ivert in zip(self.config.geodata['data'], self.config.geodata['verticals']):
                idata.buildsynth(self.multifaults.faults, direction='sd', poly='include', vertical=ivert)
                if idata.dtype in ('gps', 'insar'):
                    if idata.dtype == 'insar':
                        id = idata.vel
                        isynth = idata.synth
                    else:
                        if ivert:
                            id = np.vstack((idata.vel_enu[:, 0], idata.vel_enu[:, 1], idata.vel_enu[:, 2]))
                            isynth = np.vstack((idata.synth[:, 0], idata.synth[:, 1], idata.synth[:, 2]))
                        else:
                            id = np.vstack((idata.vel_enu[:, 0], idata.vel_enu[:, 1]))
                            isynth = np.vstack((idata.synth[:, 0], idata.synth[:, 1]))
                elif idata.dtype in ('opticorr', 'optical'):
                    id = np.hstack((idata.east, idata.north))
                    isynth = np.hstack((idata.east_synth, idata.north_synth))
                irms = np.sqrt(np.mean((isynth - id)**2))
                ivr = (1 - np.sum((isynth - id)**2) / np.sum(id**2)) * 100
                print(f'{idata.name} RMS: {irms:.4f}, VR: {ivr:.2f}%')

        return specs

    def plot_faults_geometry_correction(self, figsize=None, style=['science'],  # notebook
                                        show=True, save=False, filename='faults_perturb.png',
                                        xlabelpad=None, ylabelpad=None, zlabelpad=None,
                                        xtickpad=None, ytickpad=None, ztickpad=None,
                                        elevation=None, azimuth=None, shape=(1.0, 1.0, 0.4), show_title=True,
                                        zratio=None, zaxis_position='bottom-left', show_grid=True, grid_color='#bebebe',
                                        background_color='white', axis_color=None, output_dir='faults_output', output_gmt=True):
        """
        Plot the geometry correction of faults in a 3D plot and optionally output the original and corrected fault edges.
    
        Parameters:
        - figsize (tuple): Size of the figure (default is None).
        - style (list): Style for the plot (default is ['science']).
        - show (bool): Whether to show the plot (default is True).
        - save (bool): Whether to save the plot (default is False).
        - filename (str): Filename to save the plot (default is 'faults_perturb.png').
        - xlabelpad (float): Padding for the x-axis label (default is None).
        - ylabelpad (float): Padding for the y-axis label (default is None).
        - zlabelpad (float): Padding for the z-axis label (default is None).
        - xtickpad (float): Padding for the x-axis ticks (default is None).
        - ytickpad (float): Padding for the y-axis ticks (default is None).
        - ztickpad (float): Padding for the z-axis ticks (default is None).
        - elevation (float): Elevation angle for the 3D plot (default is None).
        - azimuth (float): Azimuth angle for the 3D plot (default is None).
        - shape (tuple): Shape of the 3D plot (default is (1.0, 1.0, 0.4)).
        - show_title (bool): Whether to show the title (default is True).
        - zratio (float): Ratio for the z-axis (default is None).
        - zaxis_position (str): Position of the z-axis ('bottom-left', 'top-right', default is 'bottom-left').
        - show_grid (bool): Whether to show grid lines (default is True).
        - grid_color (str): Color of the grid lines (default is '#bebebe').
        - background_color (str): Background color of the plot (default is 'white').
        - axis_color (str): Color of the axes (default is None).
        - output_dir (str): Directory to save the output files (default is 'faults_output').
        - output_gmt (bool): Whether to output the original and corrected fault edges in GMT format (default is True).
    
        Returns:
        - None
        """
        from ..plottools import optimize_3d_plot
        from matplotlib.ticker import FuncFormatter
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import os
    
        # Extract faults data
        trifaults = self.multifaults.faults
    
        # Create output directory if it doesn't exist
        if output_gmt and not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
        # Create a 3D plot
        with sci_plot_style(style=style):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
    
            # Define colors for different parts of the faults
            colors = {'top_coords': 'r', 'top_coords_ref': 'k',
                      'bottom_coords_ref': 'k', 'bottom_coords': 'b'}
    
            # Plot each fault and output to GMT format if required
            for fault_data in trifaults:
                fault_name = fault_data.name
                if self.config.faults[fault_name]['geometry']['update']:
                    for part, color in colors.items():
                        coords = getattr(fault_data, part)
                        x, y, z = coords[:, 0], coords[:, 1], -coords[:, 2]
                        ax.plot(x, y, z, color)
    
                        if output_gmt:
                            # Save original and corrected coordinates to files
                            part_name = part.replace('_coords', '')
                            xy_filename = os.path.join(output_dir, f"{fault_name}_{part_name}_xy.txt")
                            lonlat_filename = os.path.join(output_dir, f"{fault_name}_{part_name}_lonlat.txt")
                            np.savetxt(xy_filename, np.column_stack((x, y, -z)), fmt='%.6f')
                            lon, lat = fault_data.xy2ll(x, y)
                            np.savetxt(lonlat_filename, np.column_stack((lon, lat, -z)), fmt='%.6f')
                else:
                    print(f"Fault {fault_name} geometry is not updated.")
    
            # Set labels and title with optional labelpad
            ax.set_xlabel('X (km)', labelpad=xlabelpad)
            ax.set_ylabel('Y (km)', labelpad=ylabelpad)
            ax.set_zlabel('Depth (km)', labelpad=zlabelpad)
            if show_title:
                ax.set_title('Geometry Correction')
    
            # Adjust tick parameters with optional pad
            if xtickpad is not None:
                ax.tick_params(axis='x', pad=xtickpad)
            if ytickpad is not None:
                ax.tick_params(axis='y', pad=ytickpad)
            if ztickpad is not None:
                ax.tick_params(axis='z', pad=ztickpad)
    
            # Set z-axis tick labels to their absolute values
            ax.zaxis.set_major_formatter(FuncFormatter(lambda val, pos: f'{abs(val)}'))

            # Set View, reference to csi.geodeticplot.set_view
            if elevation is not None and azimuth is not None:
                ax.view_init(elev=elevation, azim=azimuth)

            # Optimize 3D plot
            optimize_3d_plot(ax, zratio=zratio, shape=shape, zaxis_position=zaxis_position,
                             show_grid=show_grid, grid_color=grid_color,
                             background_color=background_color, axis_color=axis_color)
    
            # Save or show plot
            if save:
                plt.savefig(filename, dpi=600)
            if show:
                plt.show()
    
    def plot_multifaults_slip(self, figsize=(None, None), slip='total', cmap='precip3_16lev_change.cpt', norm=None,
                              show=True, savefig=False, ftype='pdf', dpi=600, bbox_inches=None, 
                              method='cdict', N=None, drawCoastlines=False, plot_on_2d=True,
                              style=['notebook'], cbaxis=[0.1, 0.2, 0.1, 0.02], cblabel='',
                              xlabelpad=None, ylabelpad=None, zlabelpad=None,
                              xtickpad=None, ytickpad=None, ztickpad=None,
                              elevation=None, azimuth=None, shape=(1.0, 1.0, 1.0), zratio=None, plotTrace=True,
                              depth=None, zticks=None, map_expand=0.2, fault_expand=0.1,
                              plot_faultEdges=False, faultEdges_color='k', faultEdges_linewidth=1.0, suffix='',
                              remove_direction_labels=False, zaxis_position='bottom-left', show_grid=True, grid_color='#bebebe',
                              background_color='white', axis_color=None, cbticks=None, cblinewidth=None, cbfontsize=None, cb_label_side='opposite',
                              map_cbaxis=None):
        """
        Plot the slip distribution of multiple faults.
    
        Parameters:
        - figsize (tuple): Size of the figure and map (default is (None, None)).
        - slip (str): Type of slip to plot (default is 'total').
        - cmap (str): Colormap to use (default is 'precip3_16lev_change.cpt').
        - norm: Normalization for the colormap (default is None).
        - show (bool): Whether to show the plot (default is True).
        - savefig (bool): Whether to save the figure (default is False).
        - ftype (str): File type for saving the figure (default is 'pdf').
        - dpi (int): Dots per inch for the saved figure (default is 600).
        - bbox_inches: Bounding box in inches for saving the figure (default is None).
        - method (str): Method for getting the colormap (default is 'cdict').
        - N (int): Number of colors in the colormap (default is None).
        - drawCoastlines (bool): Whether to draw coastlines (default is False).
        - plot_on_2d (bool): Whether to plot on a 2D map (default is True).
        - style (list): Style for the plot (default is ['notebook']).
        - cbaxis (list): Colorbar axis position (default is [0.1, 0.2, 0.1, 0.02]).
        - cblabel (str): Label for the colorbar (default is '').
        - xlabelpad, ylabelpad, zlabelpad (float): Padding for the axis labels (default is None).
        - xtickpad, ytickpad, ztickpad (float): Padding for the axis ticks (default is None).
        - elevation, azimuth (float): Elevation and azimuth angles for the 3D plot (default is None).
        - shape (tuple): Shape of the 3D plot (default is (1.0, 1.0, 1.0)).
        - zratio (float): Ratio for the z-axis (default is None).
        - plotTrace (bool): Whether to plot the fault trace (default is True).
        - depth (float): Depth for the z-axis (default is None).
        - zticks (list): Ticks for the z-axis (default is None).
        - map_expand (float): Expansion factor for the map (default is 0.2).
        - fault_expand (float): Expansion factor for the fault (default is 0.1).
        - plot_faultEdges (bool): Whether to plot the fault edges (default is False).
        - faultEdges_color (str): Color for the fault edges (default is 'k').
        - faultEdges_linewidth (float): Line width for the fault edges (default is 1.0).
        - suffix (str): Suffix for the saved figure filename (default is '').
        - remove_direction_labels : If True, remove E, N, S, W from axis labels (default is False)
        - zaxis_position (str): Position of the z-axis (bottom-left, top-right) (default is 'bottom-left').
        - show_grid (bool): Whether to show grid lines (default is True).
        - grid_color (str): Color of the grid lines (default is 'gray').
        - background_color (str): Background color of the plot (default is 'white').
        - axis_color (str): Color of the axes (default is None).
        - cbticks (list): List of ticks to set on the colorbar (default is None).
        - cblinewidth (int): Width of the colorbar label border and tick lines (default is 1).
        - cbfontsize (int): Font size of the colorbar label (default is 10).
        - cb_label_side (str): Position of the label relative to the ticks ('opposite' or 'same', default is 'opposite').
        - map_cbaxis    : Axis for the colorbar on the map plot, default is None
    
        Returns:
        - None
        """
        from ..plottools import plot_slip_distribution
    
        # Print Base Information
        for ifault in self.multifaults.faults:
            print(f'Fault {ifault.name}: Mean Strike: {np.mean(ifault.getStrikes()*180/np.pi):.2f}°, Mean Dip: {np.mean(ifault.getDips()*180/np.pi):.2f}°')
    
        if len(self.multifaults.faults) > 1:
            # Combine faults if there are more than one
            combined_fault = self.multifaults.faults[0].duplicateFault()
            combined_fault.name = 'Combined Fault'
            for fault in self.multifaults.faults[1:]:
                for ipatch, islip in zip(fault.patch, fault.slip):
                    combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                    combined_fault.addpatch(ipatch, islip)
            combined_fault.setTrace(0.1)
            mfault = combined_fault
            add_faults = self.multifaults.faults
        else:
            # Directly plot if there is only one fault
            mfault = self.multifaults.faults[0]
            add_faults = [mfault]
    
        # Plot the slip distribution
        plot_slip_distribution(mfault, add_faults=add_faults, slip=slip, cmap=cmap, norm=norm, figsize=figsize, method=method, N=N,
                               drawCoastlines=drawCoastlines, plot_on_2d=plot_on_2d, cbaxis=cbaxis, cblabel=cblabel,
                               show=show, savefig=savefig, ftype=ftype, dpi=dpi, bbox_inches=bbox_inches,
                               remove_direction_labels=remove_direction_labels, cbticks=cbticks, cblinewidth=cblinewidth,
                               cbfontsize=cbfontsize, cb_label_side=cb_label_side, map_cbaxis=map_cbaxis, style=style,
                               xlabelpad=xlabelpad, ylabelpad=ylabelpad, zlabelpad=zlabelpad, xtickpad=xtickpad,
                               ytickpad=ytickpad, ztickpad=ztickpad, elevation=elevation, azimuth=azimuth, shape=shape,
                               zratio=zratio, plotTrace=plotTrace, depth=depth, zticks=zticks, map_expand=map_expand,
                               fault_expand=fault_expand, plot_faultEdges=plot_faultEdges, faultEdges_color=faultEdges_color,
                               faultEdges_linewidth=faultEdges_linewidth, suffix=suffix, show_grid=show_grid,
                               grid_color=grid_color, background_color=background_color, axis_color=axis_color,
                               zaxis_position=zaxis_position)
    
        # All Done
        return

    def plot_kde_matrix(self, figsize=None, save=False, filename='kde_matrix.png', show=True, 
                        style='white', fill=True, scatter=False, scatter_size=15, 
                        plot_sigmas=False, plot_alpha=False, plot_faults=False, faults=None, 
                        plot_geometry=False, axis_labels=None,
                        hspace=None, wspace=None, xtick_rotation=None, ytick_rotation=None,
                        plot_posterior_sigmas=False):
        """
        Plot a Kernel Density Estimation (KDE) matrix for the given parameters.
    
        Parameters:
        - figsize (tuple): Size of the figure (default is (7.5, 6.5)).
        - save (bool): Whether to save the figure (default is False).
        - filename (str): Filename to save the figure (default is 'kde_matrix.png').
        - show (bool): Whether to show the figure (default is True).
        - style (str): Seaborn style to use for the plot (default is 'white').
        - fill (bool): Whether to fill the KDE plots (default is True).
        - scatter (bool): Whether to include scatter plots in the upper triangle (default is False).
        - scatter_size (int): Size of the scatter plot points (default is 15).
        - plot_sigmas (bool): Whether to include sigma parameters in the plot (default is False).
        - plot_alpha (bool): Whether to include alpha parameters in the plot (default is False).
        - plot_faults (bool): Whether to include fault parameters in the plot (default is False).
        - faults (list or str): Specific faults to include in the plot (default is None).
        - plot_geometry (bool): Whether to include geometry parameters in the plot (default is False).
        - axis_labels (list): List of axis labels for the plot (default is None).
        - hspace (float): Horizontal space between subplots (default is None).
        - wspace (float): Vertical space between subplots (default is None).
        - xtick_rotation (float): Rotation angle for x-axis ticks (default is None).
        - ytick_rotation (float): Rotation angle for y-axis ticks (default is None).
        - plot_posterior_sigmas (bool): Whether to include posterior sigmas in the plot (default is False).
    
        Returns:
        - None
        """
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
    
        # Get the SMC chains
        trace = self.sampler.allsamples
        keys = []
        index = []
        if plot_faults:
            if faults is None:
                for fault_name in self.faultnames:
                    keys += [f"{fault_name}_{key}" for key in self.param_keys[fault_name]]
                    index += self.param_index[fault_name]
            elif type(faults) in (list, ):
                for fault_name in faults:
                    keys += [f"{fault_name}_{key}" for key in self.param_keys[fault_name]]
                    index += self.param_index[fault_name]
            elif type(faults) in (str, ):
                assert faults in self.faultnames, f"Fault {faults} not found."
                keys += self.param_keys[faults]
                index += self.param_index[faults]
        
        if plot_geometry:
            for fault_name in self.faultnames:
                if self.config.nonlinear_inversion and self.config.faults[fault_name]['geometry']['update']:
                    keys += [f"{fault_name}_{i}" for i in range(self.config.faults[fault_name]['geometry']['sample_positions'][1] - self.config.faults[fault_name]['geometry']['sample_positions'][0])]
                    index += list(range(self.config.faults[fault_name]['geometry']['sample_positions'][0], self.config.faults[fault_name]['geometry']['sample_positions'][1]))
        
        if plot_sigmas:
            keys += [f"sigmas_{i}" for i in range(self.sigmas_position[1]-self.sigmas_position[0])]
            index += list(range(self.sigmas_position[0], self.sigmas_position[1]))
            if plot_posterior_sigmas:
                index_sigmas = list(range(self.sigmas_position[0], self.sigmas_position[1]))
                aprior_sigmas = []
                for idata in self.config.geodata['data']:
                    isigma = np.mean(np.sqrt(np.diag(idata.Cd)))
                    aprior_sigmas.append(isigma)
                aprior_sigmas = np.array(aprior_sigmas)
        
        if plot_alpha:
            keys += [f'alpha_{i}' for i in range(self.alpha_position[1]-self.alpha_position[0])]
            index += list(range(self.alpha_position[0], self.alpha_position[1]))
        
        # Convert the SMC chains to a DataFrame
        df = pd.DataFrame(trace[:, index], columns=keys)
        
        if plot_posterior_sigmas:
            df.iloc[:, index_sigmas] = 10**df.iloc[:, index_sigmas] * aprior_sigmas[None, :]
        # Remove columns with zero variance
        df = df.loc[:, df.var() != 0]
        
        # Set the style
        sns.set_style(style)
        
        # Create a pair grid with separate y-axis for diagonal plots
        g = sns.PairGrid(df, diag_sharey=False)

        if figsize is not None:
            g.figure.set_size_inches(*figsize)
        
        # Remove the upper half of plots if scatter is not required
        if not scatter:
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i, j].set_visible(False)
        
        # Plot a filled KDE on the diagonal
        g.map_diag(sns.kdeplot, fill=fill)
        
        # Plot a filled KDE with scatter points on the off-diagonal
        g.map_lower(sns.kdeplot, fill=fill)
        
        # Plot scatter points on the upper half if required
        if scatter:
            g.map_upper(sns.scatterplot, s=scatter_size)
        
        # Set x-tick rotation if provided
        if xtick_rotation is not None:
            for ax in g.axes[-1, :]:
                for label in ax.get_xticklabels():
                    label.set_rotation(xtick_rotation)
                    label.set_ha('right')

        # Set y-tick rotation if provided
        if ytick_rotation is not None:
            for ax in g.axes[:, 0]:
                for label in ax.get_yticklabels():
                    label.set_rotation(ytick_rotation)
                    label.set_ha('right')
        # Set axis labels if provided
        if axis_labels:
            for i, label in enumerate(axis_labels):
                g.axes[-1, i].set_xlabel(label, fontsize=12)
                g.axes[i, 0].set_ylabel(label, fontsize=12)
        
        plt.tight_layout()
        if wspace is not None or hspace is not None:
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
        # Save the figure if required
        if save:
            plt.savefig(filename, dpi=600)
        
        # Show the figure if required
        if show:
            plt.show()

    def extract_and_plot_bayesian_results(self, rank=0, filename='samples_100_50.h5', 
                                          plot_faults=True, plot_std=False, plot_sigmas=True, plot_data=True,
                                          antisymmetric=True, res_use_data_norm=True, cmap='jet', azimuth=None, elevation=None,
                                          slip_cmap='precip3_16lev_change.cpt', depth_range=None, z_ticks=None, 
                                          axis_shape=(1.0, 1.0, 0.6), zratio=None, best_model='median', 
                                          gps_title=True, sar_title=True, sar_cbaxis=[0.15, 0.25, 0.25, 0.02],
                                          gps_figsize=None, sar_figsize=(3.5, 2.7), gps_scale=0.05, gps_legendscale=0.2,
                                          file_type='png', fault_cbaxis=[0.15, 0.22, 0.15, 0.02], fault_style=['notebook'],
                                          remove_direction_labels=False, cbticks=None, cblinewidth=None, cbfontsize=None, cb_label_side='opposite',
                                          map_cbaxis=None, data_poly=None):
        """
        Extract and plot the Bayesian results.
    
        args:
        rank: process rank (default is 0)
        filename: name of the HDF5 file to save the samples (default is 'samples_mag_rake_multifaults.h5')
        plot_faults: whether to plot faults (default is True)
        plot_std: whether to plot standard deviation (default is False)
        plot_sigmas: whether to plot sigmas (default is True)
        plot_data: whether to plot data (default is True)
        antisymmetric: whether to set the colormap to be antisymmetric (default is True)
        res_use_data_norm: whether to make the norm of 'res' consistent with 'data' and 'synth' (default is True)
        cmap: colormap to use (default is 'jet')
        slip_cmap: colormap for slip (default is 'precip3_16lev_change.cpt')
        depth_range: depth range for the plot (default is None)
        z_ticks: z-axis ticks for the plot (default is None)
        best_model: the best model to use for plotting (default is 'median')
        gps_title: whether to show title for GPS data plots (default is True)
        sar_title: whether to show title for SAR data plots (default is True)
        sar_cbaxis: colorbar axis position for SAR data plots (default is [0.15, 0.25, 0.25, 0.02])
        gps_figsize: figure size for GPS data plots (default is None)
        sar_figsize: figure size for SAR data plots (default is (3.5, 2.7))
        gps_scale: scale for GPS data plots (default is 0.05)
        gps_legendscale: legend scale for GPS data plots (default is 0.2)
        file_type: file type to save the figures (default is 'png')
        remove_direction_labels : If True, remove E, N, S, W from axis labels (default is False)
        cbticks (list): List of ticks to set on the colorbar (default is None).
        cblinewidth (int): Width of the colorbar label border and tick lines (default is 1).
        cbfontsize (int): Font size of the colorbar label (default is 10).
        cb_label_side (str): Position of the label relative to the ticks ('opposite' or 'same', default is 'opposite').
        map_cbaxis    : Axis for the colorbar on the map plot, default is None
        data_poly: None or 'include' (default is None)
        """
        if rank == 0:
            import cmcrameri
            from ..getcpt import get_cpt 
    
            if slip_cmap is not None and slip_cmap.endswith('.cpt'):
                cmap_slip = get_cpt.get_cmap(slip_cmap, method='list', N=15)
            else:
                cmap_slip = slip_cmap
            if slip_cmap is None:
                cmap_slip = get_cpt.get_cmap('precip3_16lev_change.cpt', method='list', N=15)
            self.load_from_h5(filename)
    
            if plot_std:
                self.returnModel(model='std')  # std mean
                self.plot_multifaults_slip(slip='total', cmap=cmap_slip,
                                                drawCoastlines=False, cblabel='Slip (m)',
                                                savefig=True, style=fault_style, cbaxis=fault_cbaxis,
                                                xtickpad=5, ytickpad=5, ztickpad=5,
                                                xlabelpad=15, ylabelpad=15, zlabelpad=15,
                                                shape=axis_shape, zratio=zratio, elevation=elevation, azimuth=azimuth,
                                                depth=depth_range, zticks=z_ticks, fault_expand=0.0,
                                                plot_faultEdges=False, suffix='std', remove_direction_labels=remove_direction_labels,
                                                cbticks=cbticks, cbfontsize=cbfontsize, cblinewidth=cblinewidth, cb_label_side=cb_label_side,
                                                map_cbaxis=map_cbaxis)
            self.returnModel(model=best_model)  # best model
            if plot_sigmas:
                self.plot_kde_matrix(plot_sigmas=True, plot_alpha=True, fill=True, save=True,
                                        scatter=False, filename='kde_matrix_sigmas.png')
            print('Hyper-parameters: [', ', '.join(f'{x:.7g}' for x in self.model[:self.sample_slip_only_positions[0]]), ']', sep='')
            print('STD Hyper-parameters: [', ', '.join(f'{x:.7g}' for x in self.sampler.allsamples.std(axis=0)[:self.sample_slip_only_positions[0]]), ']', sep='')
    
            if plot_faults:
                self.plot_multifaults_slip(slip='total', cmap=cmap_slip,
                                                drawCoastlines=False, cblabel='Slip (m)',
                                                savefig=True, style=fault_style, cbaxis=fault_cbaxis,
                                                xtickpad=5, ytickpad=5, ztickpad=5,
                                                xlabelpad=15, ylabelpad=15, zlabelpad=15,
                                                shape=axis_shape, zratio=zratio, elevation=elevation, azimuth=azimuth,
                                                depth=depth_range, zticks=z_ticks, fault_expand=0.0,
                                                plot_faultEdges=False, suffix=best_model if isinstance(best_model, str) else 'custom',
                                                ftype='pdf',
                                                remove_direction_labels=remove_direction_labels,
                                                cbticks=cbticks, cbfontsize=cbfontsize, cblinewidth=cblinewidth, cb_label_side=cb_label_side,
                                                map_cbaxis=map_cbaxis)

            # Build synthetic data and plot
            faults = self.multifaults.faults
            cogps_vertical_list = []
            cosar_list = []
            coopt_list = []
            datas = self.config.geodata.get('data', [])
            verticals = self.config.geodata.get('verticals', [])
            for data, vertical in zip(datas, verticals):
                if data.dtype == 'gps':
                    cogps_vertical_list.append([data, vertical])
                elif data.dtype == 'insar':
                    cosar_list.append(data)
                elif data.dtype == 'opticorr':
                    coopt_list.append(data)
            # Plot GPS data
            for fault in faults:
                if fault.lon is None or fault.lat is None:
                    fault.setTrace(0.1)
                fault.color = 'b' # Set the color to blue
            for cogps, vertical in cogps_vertical_list:
                cogps.buildsynth(faults, vertical=vertical, poly=data_poly)
                if plot_data:
                    box = [cogps.lon.min(), cogps.lon.max(), cogps.lat.min(), cogps.lat.max()]
                    cogps.plot(faults=faults, drawCoastlines=True, data=['data', 'synth'], scale=gps_scale, 
                                legendscale=gps_legendscale, color=['k', 'r'],
                                seacolor='lightblue', box=box, titleyoffset=1.02, title=gps_title, figsize=gps_figsize,
                                remove_direction_labels=remove_direction_labels)
                    cogps.fig.savefig(f'gps_{cogps.name}', ftype=file_type, dpi=600, 
                                    bbox_inches='tight', mapaxis=None, saveFig=['map'])
            # Plot SAR data
            for fault in faults:
                fault.color = 'k'
            for cosar in cosar_list:
                cosar.buildsynth(faults, vertical=True, poly=data_poly)
                if plot_data:
                    datamin, datamax = cosar.vel.min(), cosar.vel.max()
                    absmax = max(abs(datamin), abs(datamax))
                    data_norm = [-absmax, absmax] if antisymmetric else [datamin, datamax]
                    for data in ['data', 'synth', 'res']:
                        if data == 'res':
                            cosar.res = cosar.vel - cosar.synth
                            absmax = max(abs(cosar.res.min()), abs(cosar.res.max()))
                            res_norm = [-absmax, absmax] if antisymmetric else [cosar.res.min(), cosar.res.max()]
                            res_norm = data_norm if res_use_data_norm else res_norm
                            cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=sar_figsize, norm=res_norm, cmap=cmap,
                                cbaxis=sar_cbaxis, drawCoastlines=True, titleyoffset=1.02, title=sar_title,
                                remove_direction_labels=remove_direction_labels)
                        else:
                            cosar.plot(faults=faults, data=data, seacolor='lightblue', figsize=sar_figsize, norm=data_norm, cmap=cmap,
                                    cbaxis=sar_cbaxis, drawCoastlines=True, titleyoffset=1.02, title=sar_title,
                                    remove_direction_labels=remove_direction_labels)
                        cosar.fig.savefig(f'sar_{cosar.name}_{data}', ftype=file_type, dpi=600, saveFig=['map'], 
                                        bbox_inches='tight', mapaxis=None)
            
            # Plot Opticorr data
            for fault in faults:
                fault.color = 'k'
            for coopt in coopt_list:
                coopt.buildsynth(faults, vertical=False, poly=data_poly)

    def save2h5(self, samples, filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('allsamples', data=samples.allsamples)
            f.create_dataset('postval', data=samples.postval)
            f.create_dataset('beta', data=samples.beta)
            f.create_dataset('stage', data=samples.stage)
            f.create_dataset('covsmpl', data=samples.covsmpl)
            f.create_dataset('resmpl', data=samples.resmpl)

    def load_from_h5(self, filename):
        with h5py.File(filename, 'r') as f:
            
            # Create a namedtuple to store the data
            data = NT2(
                allsamples=f['allsamples'][:],
                postval=f['postval'][:],
                beta=f['beta'][:],
                stage=f['stage'][:],
                covsmpl=f['covsmpl'][:],
                resmpl=f['resmpl'][:]
            )

        self.sampler = data
            
        return data

    def resample_prior_from_samples_file(self, filename, nchains=1000):
        """
        Load and optionally resample initial samples from a given file.
        
        This method loads previously obtained sampling results from a specified file
        and resamples them according to their posterior values if the number of chains
        in the file does not match the requested number of chains. This is useful for
        initializing the sampling process with a set of samples that are representative
        of the posterior distribution.
        
        Parameters:
        - filename: str, the path to the file containing the previous sampling results.
        - nchains: int, the desired number of chains to resample to. Defaults to 1000.
        
        Returns:
        - samples: NT2, a namedtuple containing the resampled allsamples and postval,
                   along with placeholders for future use.
        """
        final = self.load_from_h5(filename)
        nchains_file, Nparams = final.allsamples.shape
        rng = np.random.default_rng()  # Create a new Generator instance

        if nchains_file != nchains:
            # Resample based on the proportion of posterior values
            weights = final.postval.flatten()
            weights /= np.sum(weights)  # Normalize weights
            indices = rng.choice(nchains_file, size=nchains, replace=True, p=weights)
            allsamples_resampled = final.allsamples[indices]
            postval_resampled = final.postval[indices]
        else:
            allsamples_resampled = final.allsamples
            postval_resampled = final.postval

        samples = NT2(allsamples_resampled, postval_resampled, np.array([0]), np.array([1]), None, None)
        return samples

    @property
    def total_samples(self):
        """Calculate the total number of samples required for the inversion."""
        return self._calculate_samples(rake_fixed=False)

    @property
    def mcmc_samples(self):
        """Calculate the total number of MCMC samples required for the inversion."""
        rake_fixed = self.config.slip_sampling_mode == 'rake_fixed'
        return self._calculate_samples(rake_fixed=rake_fixed)

    def _calculate_samples(self, rake_fixed):
        """Calculate the total number of samples required for the inversion based on the configuration whether rake is fixed or not."""
        total_samples = 0
        for fault in self.multifaults.faults:
            npatches = len(fault.patch) # Number of patches
            num_slip_samples = len(fault.slipdir)*npatches
            if rake_fixed:
                num_slip_samples //= 2
            num_poly_samples = np.sum([npoly for npoly in fault.poly.values() if npoly is not None], dtype=int)
            total_samples += num_slip_samples + num_poly_samples

            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                num_geometry_samples = self.config.faults[fault.name]['geometry']['sample_positions'][1] - self.config.faults[fault.name]['geometry']['sample_positions'][0]
                total_samples += num_geometry_samples

        if self.sigmas_position is not None:
            total_samples += self.sigmas_position[1] - self.sigmas_position[0]
        if self.alpha_position is not None:
            total_samples += self.alpha_position[1] - self.alpha_position[0]

        return total_samples

    def print_parameter_discribution(self, redo=True):
        '''
        Print the parameter description.

        Returns:
            * None
        '''

        # Create the parameter description
        if redo:
            self.multifaults.makeParamDescription()

        print('Number of data: {}'.format(self.multifaults.Nd))
        print('Number of parameters: {}'.format(self.multifaults.Np))
        print('Parameter Description ----------------------------------')

        # Loop over the param description
        for fault in self.multifaults.paramDescription:

            description = self.multifaults.paramDescription[fault]

            if ('Strike Slip' in description) or ('Dip Slip' in description) or ('Tensile' in description) or ('Coupling' in description) or ('Extra Parameters' in description):

                #Prepare the table
                print('-----------------')
                print('{:30s}||{:12s}||{:12s}||{:12s}||{:12s}||{:12s}'.format('Fault Name', 'Strike Slip', 'Dip Slip', 'Tensile', 'Coupling', 'Extra Parms'))

                # Get info
                if 'Strike Slip' in description:
                    ss = description['Strike Slip']
                else:
                    ss = 'None'
                if 'Dip Slip' in description:
                    ds = description['Dip Slip']
                else:
                    ds = 'None'
                if 'Tensile Slip' in description:
                    ts = description['Tensile Slip']
                else:
                    ts = 'None'
                if 'Coupling' in description:
                    cp = description['Coupling']
                else:
                    cp = 'None'
                if 'Extra Parameters' in description:
                    op = description['Extra Parameters']
                else:
                    op = 'None'

                # print things
                print('{:30s}||{:12s}||{:12s}||{:12s}||{:12s}||{:12s}'.format(fault, ss, ds, ts, cp, op))

            elif 'Pressure' in description:

                #Prepare the table
                print('-----------------')
                print('{:30s}||{:12s}||{:12s}'.format('Fault Name', 'Pressure', 'Extra Parms'))

                # Get info
                if 'Pressure' in description:
                    dp = description['Pressure']
                else:
                    dp = 'None'
                if 'Extra Parameters' in description:
                    op = description['Extra Parameters']
                else:
                    op = 'None'

                # print things
                print('{:30s}||{:12s}||{:12s}'.format(fault, dp, op))
    

        if 'Equalized' in self.multifaults.paramDescription:
            for case in self.multifaults.paramDescription['Equalized']:
                new,old = case
                print('-----------------') 
                print('Equalized parameter indexes: {} --> {}'.format(old,new))

    def print_parameter_positions(self):
        """Print the parameter positions."""
        print("Parameter positions:")
        for fault in self.multifaults.faults:
            print(f"Fault {fault.name}:")
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                print(f"  Geometry positions: {self.config.faults[fault.name]['geometry']['sample_positions']}")
            print(f"  Slip positions: {self.slip_positions[fault.name]}")
            print(f"  Poly positions: {self.poly_positions[fault.name]}")
        if self.config.sigmas['update']:
            print(f"Sigmas position: {self.sigmas_position}")
        if self.config.alpha['update']:
            print(f"Alpha position: {self.alpha_position}")
    
    def print_mcmc_parameter_positions(self):
        """Print the MCMC parameter positions."""
        print("MCMC Parameter Description ----------------------------------")
        total_half = 0
        for fault in self.multifaults.faults:
            # print('-----------------')
            print(f"Fault {fault.name}:")
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                print(f"  Geometry positions: {self.config.faults[fault.name]['geometry']['sample_positions']}")

            slip_start, slip_end = self.slip_positions[fault.name]
            slip_start -= total_half
            slip_end -= total_half
            if self.config.slip_sampling_mode == 'rake_fixed':
                half = (slip_end - slip_start) // 2
                print(f"  Slip positions: [{slip_start}, {slip_start + half}]")
                total_half += half
            elif self.config.slip_sampling_mode == 'magnitude_rake':
                half = (slip_end - slip_start) // 2
                print(f"  Slip magnitude positions: [{slip_start}, {slip_start + half}]")
                print(f"  Rake positions: [{slip_start + half}, {slip_end}]")
            else:
                print(f"  Slip positions: [{slip_start}, {slip_end}]")

            poly_start, poly_end = self.poly_positions[fault.name]
            poly_start -= total_half
            poly_end -= total_half
            if poly_start != poly_end:
                print(f"  Poly positions: [{poly_start}, {poly_end}]")

        if self.config.sigmas['update']:
            sigmas_start, sigmas_end = self.sigmas_position
            print(f"Sigmas position: [{sigmas_start}, {sigmas_end}]")
        if self.config.alpha['update']:
            alpha_start, alpha_end = self.alpha_position
            print(f"Alpha position: [{alpha_start}, {alpha_end}]")
    
    def print_moment_magnitude(self, mu=3.e10, slip_factor=1.0):
        import csi.faultpostproc as faultpp
        # Get the first fault
        first_fault = self.multifaults.faults[0]
        lon0, lat0, utmzone = first_fault.lon0, first_fault.lat0, first_fault.utmzone
    
        # Combine the first fault
        combined_fault = first_fault.duplicateFault()
    
        # Combine the remaining faults
        if len(self.multifaults.faults) > 1:
            for ifault in self.multifaults.faults[1:]:
                for patch, slip in zip(ifault.patch, ifault.slip):
                    combined_fault.N_slip = combined_fault.slip.shape[0] + 1
                    combined_fault.addpatch(patch, slip)
    
        # Combine the fault names
        fault_names = [fault.name for fault in self.multifaults.faults]
        combined_name = '_'.join(fault_names)

        # Scale the slip
        combined_fault.slip *= slip_factor
    
        # Patches 2 vertices
        combined_fault.setVerticesFromPatches()
        combined_fault.numpatch = len(combined_fault.patch)
        # Compute the triangle areas, moments, moment tensor, and magnitude
        combined_fault.compute_triangle_areas()
        fault_processor = faultpp(combined_name, combined_fault, mu, lon0=lon0, lat0=lat0, utmzone=utmzone)
        fault_processor.computeMoments()
        fault_processor.computeMomentTensor()
        fault_processor.computeMagnitude()
    
        # Print the moment magnitude
        self.tripproc = fault_processor
        print(f"Mo is: {fault_processor.Mo:.8e}")
        print(f"Mw is {fault_processor.Mw:.1f}")

    def set_parameter_bounds(self, lb=None, ub=None, geometry=None, poly=None, strikeslip=None, dipslip=None, rake_angle=None, slip_magnitude=None, alpha=None, sigmas=None):
        """
        Sets the parameter bounds for the Bayesian inversion process.

        This method configures the bounds for all parameters involved in the inversion, including the default bounds,
        geometry, polynomial coefficients, strike-slip, dip-slip, rake angle, slip magnitude, alpha, and sigmas. It
        utilizes the BoundsManager to handle the complexity of setting these bounds.

        Parameters:
        - lb (float, optional): The lower bound for all parameters if not specified individually.
        - ub (float, optional): The upper bound for all parameters if not specified individually.
        - geometry (dict, optional): Specific bounds for the geometry parameters.
        - poly (dict, optional): Specific bounds for the polynomial coefficients.
        - strikeslip (dict, optional): Specific bounds for the strike-slip parameters.
        - dipslip (dict, optional): Specific bounds for the dip-slip parameters.
        - rake_angle (dict, optional): Specific bounds for the rake angle parameters.
        - slip_magnitude (dict, optional): Specific bounds for the slip magnitude parameters.
        - alpha (list, optional): Specific bounds for the alpha parameter.
        - sigmas (list, optional): Specific bounds for the sigmas parameters.

        The method updates the bounds in the BoundsManager instance and stores the updated lower and upper bounds,
        as well as the detailed bounds configuration for each parameter type.
        """

        bounds_manger = self.bounds_manger
        bounds_manger.set_default_bounds(lb, ub)
        bounds_manger.update_bounds_for_all_mcmc_parameters(geometry, slip_magnitude, rake_angle, strikeslip, dipslip, poly, sigmas, alpha)
        self.bounds = bounds_manger.bounds
    
    def set_parameter_bounds_from_config(self, config_file='bounds_config.yml', encoding='utf-8'):
        '''
        Set the parameter bounds from the configuration file.
        '''
        self.bounds_manger.update_bounds_from_config(config_file, encoding)
        self.bounds = self.bounds_manger.bounds

    def get_bounds_for_F_J(self):
        """
        Retrieves the bounds for the F and J matrices used in the inversion process.

        This method is a convenience wrapper around the BoundsManager's method to get the bounds specifically formatted
        for use in constructing the F_J sampling mode, which are essential components of the Bayesian inversion algorithm.

        Returns:
        - A tuple containing two elements: the lower and upper bounds arrays that are used in the calculation of the F and J matrices.
        """
        return self.bounds_manger.get_bounds_for_F_J()

    def calculate_sigmas_alpha_positions(self):
        """
        Calculate the positions for sigmas and alpha parameters in the sampling vector.
        Ensures that the total geometry parameters are based on the maximum sampling position.
        """
        # Determine the maximum geometry parameter position
        max_geometry_position = 0
        for fault in self.multifaults.faults:
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                sample_positions = self.config.faults[fault.name]['geometry']['sample_positions']
                if sample_positions is None or len(sample_positions) != 2:
                    raise ValueError(f"Invalid sample_positions for fault {fault.name}. It should be a list with two elements [st, ed].")
                max_geometry_position = max(max_geometry_position, sample_positions[1])
    
        # Validate that the geometry sampling positions cover the range from 0 to max_geometry_position
        covered_positions = set()
        for fault in self.multifaults.faults:
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                sample_positions = self.config.faults[fault.name]['geometry']['sample_positions']
                covered_positions.update(range(sample_positions[0], sample_positions[1]))
    
        if set(range(max_geometry_position)) != covered_positions:
            raise ValueError(
                f"Geometry sampling positions do not fully cover the range from 0 to {max_geometry_position}. "
                f"Covered positions: {sorted(covered_positions)}"
            )
    
        # Set total_geometry_parameters to the maximum position
        self.total_geometry_parameters = max_geometry_position
    
        # Calculate the positions for sigmas
        n = len(self.multifaults.faults[0].d)  # Number of data points
        if self.config.sigmas['update']:
            self.sigmas_position = (self.total_geometry_parameters, self.total_geometry_parameters + n)
        else:
            self.sigmas_position = None
            n = 0
    
        # Calculate the positions for alpha
        if self.config.alpha['update']:
            n_alpha = len(self.config.alphaFaults)
            self.alpha_position = (self.total_geometry_parameters + n, self.total_geometry_parameters + n + n_alpha)
        else:
            self.alpha_position = None
    
    
    def calculate_geometry_positions(self):
        """
        Calculate the positions for geometry parameters in the sampling vector.
        Ensures that the positions are correctly aligned and do not overlap.
        """
        self.geometry_positions = {}
        max_geometry_position = 0
    
        for fault in self.multifaults.faults:
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                sample_positions = self.config.faults[fault.name]['geometry']['sample_positions']
                if sample_positions is None or len(sample_positions) != 2:
                    raise ValueError(f"Invalid sample_positions for fault {fault.name}. It should be a list with two elements [st, ed].")
                max_geometry_position = max(max_geometry_position, sample_positions[1])
                self.geometry_positions[fault.name] = sample_positions
            else:
                self.geometry_positions[fault.name] = [0, 0]
    
        # Validate that the geometry sampling positions cover the range from 0 to max_geometry_position
        covered_positions = set()
        for positions in self.geometry_positions.values():
            covered_positions.update(range(positions[0], positions[1]))
    
        if set(range(max_geometry_position)) != covered_positions:
            raise ValueError(
                f"Geometry sampling positions do not fully cover the range from 0 to {max_geometry_position}. "
                f"Covered positions: {sorted(covered_positions)}"
            )
    
        # Ensure total_geometry_parameters covers the full range
        self.total_geometry_parameters = max_geometry_position

    def calculate_slip_and_poly_positions(self):
        self.slip_positions = {}
        self.poly_positions = {}
        start_position = 0
        if self.sigmas_position is not None:
            start_position += self.sigmas_position[1] - self.sigmas_position[0]
        if self.alpha_position is not None:
            start_position += self.alpha_position[1] - self.alpha_position[0]
        for fault in self.multifaults.faults:
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                start_position += self.config.faults[fault.name]['geometry']['sample_positions'][1] - self.config.faults[fault.name]['geometry']['sample_positions'][0]
        for fault in self.multifaults.faults:
            npatches = len(fault.patch)
            num_slip_samples = len(fault.slipdir)*npatches
            num_poly_samples = np.sum([npoly for npoly in fault.poly.values() if npoly is not None], dtype=int)
            self.slip_positions[fault.name] = (start_position, start_position + num_slip_samples)
            self.poly_positions[fault.name] = (start_position + num_slip_samples, start_position + num_slip_samples + num_poly_samples)
            start_position += num_slip_samples + num_poly_samples

    def calculate_linear_sample_start_position(self):
        start_position = 0
        if self.config.sigmas['update']:
            start_position += self.sigmas_position[1] - self.sigmas_position[0]
        if self.config.alpha['update']:
            start_position += self.alpha_position[1] - self.alpha_position[0]
        for fault in self.multifaults.faults:
            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                start_position += self.config.faults[fault.name]['geometry']['sample_positions'][1] - self.config.faults[fault.name]['geometry']['sample_positions'][0]
        self.linear_sample_start_position = start_position
        return start_position

    def calculate_sample_slip_only_positions(self):
        slip_only_positions = []
        for fault_name in self.faultnames:
            slip_start, slip_end = self.slip_positions[fault_name]
            slip_only_positions.extend(list(range(slip_start, slip_end)))
        slip_only_positions = np.array(slip_only_positions)
        self.sample_slip_only_positions = slip_only_positions
        return slip_only_positions

    def compute_slip(self, samples, fault):
        slip_start, slip_end = self.slip_positions[fault.name]
        if self.config.slip_sampling_mode == 'magnitude_rake':
            slip_magnitude_and_rake = samples[slip_start:slip_end].copy()
            half = len(slip_magnitude_and_rake) // 2
            slip = slip_magnitude_and_rake[:half]
            return slip
        elif self.config.slip_sampling_mode == 'rake_fixed':
            slip = samples[slip_start:slip_start + (slip_end - slip_start) // 2].copy()
            return slip
        else:
            slip = samples[slip_start:slip_end].copy()  # Create a copy of slip to avoid modifying samples
            if len(fault.slipdir) == 2:  # If both components of slip (dip or strikeslip)
                slip = slip.reshape(2, -1)
                slip = np.sqrt(np.sum(slip**2, axis=0))
            else:  # If only one component of slip (dip or strikeslip)
                slip = np.abs(slip)
            return slip
    
    def transfer_magnitude_rake_to_ss_ds(self, slip_magnitude, rake):
        """
        Transfer the slip magnitude and rake to strike-slip and dip-slip components.

        Parameters:
        - slip_magnitude (np.ndarray): The slip magnitude samples.
        - rake (np.ndarray): The rake angle samples.

        Returns:
        - np.ndarray: The strike-slip and dip-slip components.
        """
        ss = slip_magnitude * np.cos(np.radians(rake))
        ds = slip_magnitude * np.sin(np.radians(rake))
        ss_ds = np.hstack([ss, ds])
        return ss_ds

    def transfer_samples(self, samples):
        if self.config.slip_sampling_mode == 'magnitude_rake':
            new_samples = samples.copy()
            for fault_name in self.faultnames:
                slip_start, slip_end = self.slip_positions[fault_name]
                slip_magnitude_and_rake = new_samples[slip_start:slip_end]
                half = len(slip_magnitude_and_rake) // 2
                slip_magnitude = slip_magnitude_and_rake[:half]
                rake = slip_magnitude_and_rake[half:]
                new_samples[slip_start:slip_end] = self.transfer_magnitude_rake_to_ss_ds(slip_magnitude, rake)
            return new_samples
        elif self.config.slip_sampling_mode == 'rake_fixed':
            new_samples = samples
            for fault_name in self.faultnames:
                slip_start, slip_end = self.slip_positions[fault_name]
                half = (slip_end - slip_start) // 2
                slip = new_samples[slip_start:slip_start + half]
                rake = np.full_like(slip, self.config.rake_angle)
                ss_ds = self.transfer_magnitude_rake_to_ss_ds(slip, rake)
                new_samples = np.concatenate((new_samples[:slip_start], ss_ds, new_samples[slip_start + half:]))
            return new_samples
        else:
            return samples
    
    def compute_magnitude_log_prior(self, samples, decay_rate=0.1):
        moment_magnitude_threshold = self.moment_magnitude_threshold
        magnitude_tolerance = self.magnitude_tolerance
    
        # Precompute the constant value
        constant_value = self.shear_modulus * np.array([self.patch_areas[fault.name] for fault in self.multifaults.faults])
    
        # Compute slip for all faults
        slips = np.array([self.compute_slip(samples, fault) for fault in self.multifaults.faults])
    
        # Compute moment for all faults
        moments = np.sum(constant_value * slips, axis=1)
    
        total_moment = np.sum(moments)*1e6 # km^2 to m^2
        moment_magnitude = 2.0 / 3.0 * (np.log10(total_moment) - 9.1)
    
        magnitude_difference = np.abs(moment_magnitude - moment_magnitude_threshold)
        # Compute the log prior using a piecewise function
        if magnitude_difference <= magnitude_tolerance:
            return 0.0  # log(1) for values within the range
        else:
            # Gaussian decay for values outside the range
            log_prior = -0.5 * (magnitude_difference / decay_rate) ** 2
            # log_prior = np.log(1e-20) # Set to a very small value for values outside the range
            return log_prior

    def generate_magnitude_single_slip_sample(self, faults=None, lb=None, ub=None):
        """
        Generate a single slip sample considering the moment magnitude constraint.

        Parameters:
        faults (list): List of fault names. If None, use all faults.
        lb (dict): Lower bounds for each fault. If None, use lb in self.bound_manager.bounds.
        ub (dict): Upper bounds for each fault. If None, use ub in self.bound_manager.bounds.

        Returns:
        dict: A dictionary where keys are fault names and values are slip samples.
        float: The moment magnitude of the generated sample.
        """

        # If faults is not provided, use all faults
        if faults is None:
            faults = [fault.name for fault in self.multifaults.faults]

        # If lb is not provided, use the first or second half of self.lb based on mode
        if lb is None or ub is None:
            lb = {}
            ub = {}
            if self.config.slip_sampling_mode == 'ss_ds':
                bound_ss = self.bounds_manger.bounds['strikeslip']
                bound_ds = self.bounds_manger.bounds['dipslip']
                for name in faults:
                    npatch = len(self.multifaults.faults_dict[name].patch)
                    lb_ss, ub_ss = np.full(npatch, bound_ss[name][0]), np.full(npatch, bound_ss[name][1])
                    lb_ds, ub_ds = np.full(npatch, bound_ds[name][0]), np.full(npatch, bound_ds[name][1])
                    lb[name] = np.min(np.abs(np.vstack((lb_ss, ub_ss, lb_ds, ub_ds)).T), axis=1)
                    ub[name] = np.max(np.abs(np.vstack((lb_ss, ub_ss, lb_ds, ub_ds)).T), axis=1)
            else:
                bound = self.bounds_manger.bounds['slip_magnitude']
                for name in faults:
                    npatch = len(self.multifaults.faults_dict[name].patch)
                    lb[name], ub[name] = np.full(npatch, bound[name][0]), np.full(npatch, bound[name][1])

        # Generate Mw from a normal distribution
        moment_magnitude = np.random.normal(self.moment_magnitude_threshold, self.magnitude_tolerance)

        # Convert Mw to total moment
        total_moment = np.power(10, 1.5*moment_magnitude + 9.1)

        # Calculate the total number of subfaults
        num_subfaults = sum(len(self.patch_areas[name]) for name in faults)

        # Generate moments for subfaults such that their sum is approximately equal to total moment
        moments = np.random.dirichlet(np.ones(num_subfaults)) * total_moment/self.shear_modulus

        # Convert moments to slips and clip them to be within bounds
        slips = []
        start = 0
        for name in faults:
            num_patches = len(self.patch_areas[name])
            slip = moments[start:start+num_patches] / (np.array(self.patch_areas[name])*1e6)  # km^2 to m^2
            slip = np.clip(slip, lb[name], ub[name])
            slips.append(slip)
            start += num_patches

        # Store slips in a dictionary
        slip_dict = {name: slip for name, slip in zip(faults, slips)}

        return slip_dict, moment_magnitude
    
    def generate_magnitude_multiple_slip_samples(self, nchains, faults=None, lb=None, ub=None):
        """
        Generate multiple slip samples considering the moment magnitude constraint.

        Parameters:
        nchains (int): Number of chains for which to generate samples.
        faults (list): List of fault names. If None, use all faults.
        lb (dict): Lower bounds for each fault. If None, use lb in self.bound_manager.bounds.
        ub (dict): Upper bounds for each fault. If None, use ub in self.bound_manager.bounds.

        Returns:
        dict: A dictionary where keys are fault names and values are arrays of slip samples.
        list: A list of moment magnitudes of the generated samples.
        """

        # Initialize a dictionary to store the samples
        samples = {name: [] for name in self.patch_areas.keys()}
        mws = []

        for _ in range(nchains):
            # Compute slip prior distribution for each chain
            slip_dict, mw = self.generate_magnitude_single_slip_sample(faults, lb, ub)

            # Append the slip values to the corresponding lists in the samples dictionary
            for name, slip in slip_dict.items():
                samples[name].append(slip)
            mws.append(mw)

        # Convert the lists in the samples dictionary to numpy arrays
        for name in samples.keys():
            samples[name] = np.vstack(samples[name])

        return samples, mws
    
    def prior_samples_vectorize(self, target, nchains, magprior=True, faults=None, sliplb=None, slipub=None, 
                                rake_angle=None, rake_sigma=None, rake_range=None):
        """
        Generate samples for a given number of chains.

        Parameters:
        nchains (int): Number of chains for which to generate samples.
        faults (list): List of fault names. If None, use all faults.
        sliplb (dict): Lower bounds for each fault. If None, use the first half of self.lb.
        slipub (dict): Upper bounds for each fault. If None, use the first half of self.ub.
        magprior (bool): If True, use magnitude prior for generating samples.
        rake_angle (float): Rake angle in degrees. Required if mode is 'ss_ds'.
        rake_sigma (float): Standard deviation of rake angle. Required if mode is 'ss_ds'.
        rake_range (tuple): Lower and upper bounds of rake angle. Required if mode is 'ss_ds'.

        Returns:
        NT2: A named tuple containing the generated samples, their posterior values, beta, stage, and None for acceptance and swap.
        """

        if self.config.slip_sampling_mode == 'ss_ds' and (rake_angle is None or rake_sigma is None or rake_range is None):
            raise ValueError("When mode is 'ss_ds', rake_angle, rake_sigma, and rake_range must be provided.")
        
        from scipy.stats import truncnorm
        numpars = self.lb.shape[0]
        diffbnd = self.ub - self.lb
        diffbndN = np.tile(diffbnd,(nchains,1))
        LBN = np.tile(self.lb,(nchains,1))
        
        sampzero = LBN +  np.random.rand(nchains,numpars) * diffbndN
        beta = np.array([0]) 
        stage = np.array([1]) 

        if magprior:
            samples, mws = self.generate_magnitude_multiple_slip_samples(nchains, faults=faults, lb=sliplb, ub=slipub)
            sample_mode = self.config.slip_sampling_mode
            for name in samples.keys():
                start, end = self.slip_positions[name]
                half = (end + start) // 2
                if sample_mode == 'magnitude_rake':
                    sampzero[:, start:half] = samples[name]
                elif sample_mode == 'rake_fixed':
                    rake_rad = np.radians(self.config.rake_angle)
                    ss = samples[name] * np.cos(rake_rad)
                    ds = samples[name] * np.sin(rake_rad)
                    sampzero[:, start:end] = np.hstack([ss, ds])
                elif sample_mode == 'ss_ds':
                    if rake_sigma == 0:
                        rake_rad = np.radians(rake_angle)
                    else:
                        rake_dist = truncnorm((rake_range[0] - rake_angle) / rake_sigma, (rake_range[1] - rake_angle) / rake_sigma, loc=rake_angle, scale=rake_sigma)
                        rake_rad = np.radians(rake_dist.rvs(samples[name].shape))
                    ss = samples[name] * np.cos(rake_rad)
                    ds = samples[name] * np.sin(rake_rad)
                    sampzero[:, start:end] = np.hstack([ss, ds])
        
        # Compute log prior
        logpost = np.apply_along_axis(target, 1, sampzero)
        postval = logpost.reshape(-1, 1)
            
        samples = NT2(sampzero, postval, beta, stage, None, None)
        return samples

    def compute_log_prior(self, samples):
        return compute_log_prior(samples, self.lb, self.ub)

    def make_target_for_parallel(self, log_enabled=False):
        self.bayesian_sampling_mode = 'FullSMC'
        if self.nonlinear_inversion:
            def target(samples):
                # Compute log prior
                log_prior = compute_log_prior(samples, self.lb, self.ub)
                if log_prior == -np.inf:
                    return -np.inf

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        # self._update_fault(fault_name, fault_config, samples)
                        self._update_fault_geometry_and_mesh(fault_name, fault_config, samples, log_enabled=log_enabled)
                        self._update_fault_GFs_and_Laplacian(fault_name, fault_config, log_enabled=log_enabled)

                new_samples = self.transfer_samples(samples)
                return log_prior + self._compute_likelihoods(new_samples)
        else:
            def target(samples):
                # Compute log prior
                # start_time = time.time()
                log_prior = compute_log_prior(samples, self.lb, self.ub)
                # end_time = time.time()
                # print(f"Execution time for computing log prior: {end_time - start_time} seconds")

                if log_prior == -np.inf:
                    return -np.inf
                
                new_samples = self.transfer_samples(samples)
                return log_prior + self._compute_likelihoods(new_samples, GL_combined=self.GL_combined)
        self.target = target
        return target

    def make_magnitude_target_for_parallel(self, decay_rate=0.1, log_enabled=False):
        self.bayesian_sampling_mode = 'FullSMC'
        if self.nonlinear_inversion:
            def target(samples):
                # Compute log prior
                log_prior = compute_log_prior(samples, self.lb, self.ub)
                if log_prior == -np.inf:
                    return -np.inf

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        self._update_fault_geometry_and_mesh(fault_name, fault_config, samples, update_areas=True, log_enabled=log_enabled)

                # Compute log magnitude prior
                # start_time_magnitude_log_prior = time.time()
                magnitude_log_prior = self.compute_magnitude_log_prior(samples, decay_rate=decay_rate)
                # end_time_magnitude_log_prior = time.time()
                # print(f"Execution time for computing magnitude log prior: {end_time_magnitude_log_prior - start_time_magnitude_log_prior} seconds")

                # if magnitude_log_prior != 0.0:
                #     return magnitude_log_prior

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        self._update_fault_GFs_and_Laplacian(fault_name, fault_config, log_enabled=log_enabled)

                new_samples = self.transfer_samples(samples)
                return log_prior + magnitude_log_prior + self._compute_likelihoods(new_samples)
        else:
            def target(samples):
                # Compute log prior
                log_prior = compute_log_prior(samples, self.lb, self.ub)

                if log_prior == -np.inf:
                    return -np.inf

                # Compute log magnitude prior
                magnitude_log_prior = self.compute_magnitude_log_prior(samples, decay_rate=decay_rate)
                # if magnitude_log_prior != 0.0:
                #     return magnitude_log_prior
                
                new_samples = self.transfer_samples(samples)
                return log_prior + magnitude_log_prior + self._compute_likelihoods(new_samples, GL_combined=self.GL_combined)
        self.target = target
        return target

    def make_F_J_target_for_parallel(self, log_enabled=False, A=None, b=None, Aeq=None, beq=None, \
                lb=None, ub=None, x0=None, opts=None, smooth_prior_weight=1.0, 
                magnitude_log_prior=False, decay_rate=0.1):
        self.bayesian_sampling_mode = 'SMC_F_J'
        self.config.slip_sampling_mode = 'ss_ds'

        # Get the constraints if not provided
        if (A is None or b is None) and self.config.use_rake_angle_constraints:
            A, b = self.bounds_manger.get_inequality_constraints_for_rake_angle()
        # Get the bounds if not provided    
        if (lb is None or ub is None) and self.config.use_bounds_constraints:
            lb, ub = self.get_bounds_for_F_J()
        
        hyper_lb, hyper_ub = self.bounds_manger.get_bounds_for_geometry_hyperparameters()

        if self.nonlinear_inversion:
            def target(samples):
                # Compute log prior
                log_prior = compute_log_prior(samples, hyper_lb, hyper_ub)
                if log_prior == -np.inf:
                    return -np.inf

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        self._update_fault_geometry_and_mesh(fault_name, fault_config, samples, log_enabled=log_enabled)
                        self._update_fault_GFs_and_Laplacian(fault_name, fault_config, log_enabled=log_enabled)

                return log_prior + self._compute_likelihoods_F_J(samples, A=A, b=b, Aeq=Aeq, beq=beq, \
                                                                 lb=lb, ub=ub, x0=x0, opts=opts, smooth_prior_weight=smooth_prior_weight,
                                                                 magnitude_log_prior=magnitude_log_prior, decay_rate=decay_rate)
        else:
            def target(samples):
                # Compute log prior
                log_prior = compute_log_prior(samples, hyper_lb, hyper_ub)

                if log_prior == -np.inf:
                    return -np.inf
                
                return log_prior + self._compute_likelihoods_F_J(samples, GL_combined=self.GL_combined, A=A, b=b, Aeq=Aeq, beq=beq, \
                                                                 lb=lb, ub=ub, x0=x0, opts=opts, smooth_prior_weight=smooth_prior_weight,
                                                                 magnitude_log_prior=magnitude_log_prior, decay_rate=decay_rate)
        self.target = target
        return target

    def _update_fault_geometry_and_mesh(self, fault_name, fault_config, samples, update_areas=False, log_enabled=False):
        start, end = fault_config['geometry']['sample_positions']
        # print(f"Updating fault {fault_name} geometry with samples from position {start} to {end}")
        sample_values = samples[start:end]
        # Update fault geometry
        start_time_geometry = time.time()
        if not self.multifaults.faults_dict[fault_name].geometry_updated:
            # print('sample_values', sample_values)
            self.multifaults.update_fault_geometry(fault_names=[fault_name], perturbations=sample_values, **fault_config['method_parameters']['update_fault_geometry'])
        end_time_geometry = time.time()
        log_time(start_time_geometry, end_time_geometry, "Execution time for updating fault geometry", log_enabled)
        # Update mesh
        start_time_mesh = time.time()
        if not self.multifaults.faults_dict[fault_name].mesh_updated:
            self.multifaults.update_mesh(fault_names=[fault_name], **fault_config['method_parameters']['update_mesh'])
        end_time_mesh = time.time()
        log_time(start_time_mesh, end_time_mesh, "Execution time for updating mesh", log_enabled)
        # Optionally update fault areas
        if update_areas:
            self.multifaults.compute_fault_areas(fault_names=[fault_name])

    def _update_fault_GFs_and_Laplacian(self, fault_name, fault_config, log_enabled=False):
        # Update GFs
        start_time_GFs = time.time()
        self.multifaults.update_GFs(fault_names=[fault_name], **fault_config['method_parameters']['update_GFs'])
        end_time_GFs = time.time()
        log_time(start_time_GFs, end_time_GFs, "Execution time for updating GFs", log_enabled)
        # Update Laplacian
        start_time_Laplacian = time.time()
        if not self.multifaults.faults_dict[fault_name].laplacian_updated:
            self.multifaults.update_Laplacian(fault_names=[fault_name], **fault_config['method_parameters']['update_Laplacian'])
        end_time_Laplacian = time.time()
        log_time(start_time_Laplacian, end_time_Laplacian, "Execution time for updating Laplacian", log_enabled)

    def _compute_likelihoods(self, samples, GL_combined=None):
        """
        Compute the total log-likelihood, including data and smoothness terms.
    
        Parameters:
        samples (ndarray): The parameter samples.
        GL_combined (ndarray, optional): The combined Laplacian matrix. Defaults to None.
    
        Returns:
        float: The total log-likelihood.
        """
        # Combine Green's functions for all faults
        G_combined = np.hstack([fault.Gassembled for fault in self.multifaults.faults])
        self.G_combined = G_combined
        if GL_combined is None:
            GL_combined = block_diag([fault.GL for fault in self.multifaults.faults]).toarray()
    
        # Extract sigmas and alpha values
        sigmas = self.config.sigmas['initial_value'] if self.sigmas_position is None else samples[self.sigmas_position[0]:self.sigmas_position[1]]
    
        # Convert log-scaled sigmas to non-log-scaled if necessary
        if self.config.sigmas.get('log_scaled', True):
            sigmas = np.power(10, sigmas)
    
        # Initialize inverse covariance matrix and log determinant
        inv_cov = np.zeros((len(self.observations), len(self.observations)))
        log_det = 0.0
        st = 0
        ed = 0
        for ind, idataname in enumerate(self.datanames):
            ed += len(self.obs_dict[idataname])
            isigmas_2 = sigmas[ind] ** 2
            inv_cov[st:ed, st:ed] = np.divide(self.inv_covs[idataname], isigmas_2)
            log_det += len(self.obs_dict[idataname]) * np.log(isigmas_2)
            st = ed
    
        # Compute data log-likelihood
        linear_sample = samples[self.linear_sample_start_position:]
        data_log_likelihood = compute_data_log_likelihood(G_combined, linear_sample, self.observations, inv_cov, log_det)
    
        # Check if alpha smoothing is enabled
        if not self.config.alpha.get('enabled', True):
            return data_log_likelihood
    
        # If alpha smoothing is enabled, proceed with smoothness log-likelihood
        alpha = self.config.alpha['initial_value'] if self.alpha_position is None else samples[self.alpha_position[0]:self.alpha_position[1]]
    
        # Convert log-scaled alpha to non-log-scaled if necessary
        if self.config.alpha.get('log_scaled', True):
            alpha = np.power(10, alpha)
    
        # Expand alpha values for each fault
        alpha_faults = alpha[self.config.alphaFaultsIndex]
        size_faults = [len(fault.patch) * len(fault.slipdir) for fault in self.multifaults.faults]
        alpha = np.hstack([[alpha_faults[ind]] * size_faults[ind] for ind in range(len(alpha_faults))])
    
        # Compute smoothness log-likelihood
        linear_sample_slip_only = samples[self.sample_slip_only_positions]
        smooth_log_likelihood = compute_smooth_log_likelihood(GL_combined, linear_sample_slip_only, alpha)
    
        # Return the total log-likelihood
        return data_log_likelihood + smooth_log_likelihood

    def _compute_likelihoods_F_J(self, samples, GL_combined=None, A=None, b=None, Aeq=None, beq=None, 
                                 lb=None, ub=None, x0=None, opts=None, 
                                 smooth_prior_weight=1.0, magnitude_log_prior=False, decay_rate=0.1):
        """
        Calculate the log likelihood of the data given the samples.
    
        Parameters:
        samples (array): The samples to be used in the calculation.
        GL_combined (array, optional): The combined Laplacian matrix for all faults. If None, it will be computed.
        A (array, optional): The A matrix for the linear constrained least squares problem.
        b (array, optional): The b vector for the linear constrained least squares problem.
        Aeq (array, optional): The Aeq matrix for the linear constrained least squares problem.
        beq (array, optional): The beq vector for the linear constrained least squares problem.
        lb (array, optional): The lower bounds for the linear constrained least squares problem.
        ub (array, optional): The upper bounds for the linear constrained least squares problem.
        x0 (array, optional): The initial guess for the linear constrained least squares problem.
        opts (dict, optional): The options for the linear constrained least squares problem.
        smooth_prior_weight (float, optional): The weight of the smoothness prior in the log likelihood calculation.
        magnitude_log_prior (bool, optional): If True, include the magnitude log prior in the log likelihood calculation.
        decay_rate (float, optional): The decay rate for the magnitude log prior.
    
        Returns:
        float: The log likelihood of the data given the samples.
        """
        G_combined = np.hstack([fault.Gassembled for fault in self.multifaults.faults])
        if GL_combined is None:
            GL_combined = block_diag([fault.GL for fault in self.multifaults.faults]).toarray()
    
        sigmas = self.config.sigmas['initial_value'] if self.sigmas_position is None else samples[self.sigmas_position[0]:self.sigmas_position[1]]
    
        # Convert log-scaled sigmas to non-log-scaled if necessary
        if self.config.sigmas.get('log_scaled', True):
            sigmas = np.power(10, sigmas)
    
        # Initialize inverse covariance matrix and log determinant
        inv_cov = np.zeros((len(self.observations), len(self.observations)))
        log_det = 0.0
        st = 0
        ed = 0
        chol_decomps = []
        for ind, idataname in enumerate(self.datanames):
            ed += len(self.obs_dict[idataname])
            isigmas_2 = sigmas[ind] ** 2
            inv_cov[st:ed, st:ed] = np.divide(self.inv_covs[idataname], isigmas_2)
            log_det += len(self.obs_dict[idataname]) * np.log(isigmas_2)
            chol_decomps.append(self.chol_decomps[idataname] / sigmas[ind])
            st = ed
    
        # Invert slip samples and poly samples
        G = G_combined
        d = self.observations
        W = scipy.linalg.block_diag(*chol_decomps)
    
        # Check if alpha smoothing is enabled
        if not self.config.alpha.get('enabled', True):
            # If alpha smoothing is disabled, skip smoothness-related calculations
            d2I = np.dot(W, d)
            G2I = np.dot(W, G)
            self.G_combined = G_combined
            self.G2I = G2I
            try:
                mpost = self.least_squares_inversion(G2I, d2I, reg=0, A=A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub, x0=x0, opts=opts)
            except:
                return -9999999
            self.mpost = mpost
            mpost = np.hstack((samples[:self.linear_sample_start_position], mpost))
    
            # Compute data log likelihood
            linear_sample = mpost[self.linear_sample_start_position:]
            data_log_likelihood = compute_data_log_likelihood(G_combined, linear_sample, self.observations, inv_cov, log_det)
    
            # Compute magnitude log prior if enabled
            magnitude_log_prior_value = 0.0
            if magnitude_log_prior:
                magnitude_log_prior_value = self.compute_magnitude_log_prior(mpost, decay_rate)
    
            return data_log_likelihood + magnitude_log_prior_value
    
        # If alpha smoothing is enabled, proceed with smoothness-related calculations
        alpha = self.config.alpha['initial_value'] if self.alpha_position is None else samples[self.alpha_position[0]:self.alpha_position[1]]
    
        # Convert log-scaled alpha to non-log-scaled if necessary
        if self.config.alpha.get('log_scaled', True):
            alpha = np.power(10, alpha)
    
        alpha_faults = alpha[self.config.alphaFaultsIndex]
        size_faults = [len(fault.patch) * len(fault.slipdir) for fault in self.multifaults.faults]
        alpha = np.hstack([[alpha_faults[ind]] * size_faults[ind] for ind in range(len(alpha_faults))])
    
        GL_combined_poly = self.combine_GL_poly(GL_combined)
        d2I = np.hstack((np.dot(W, d), np.zeros(GL_combined_poly.shape[0])))
        G2I = np.vstack((np.dot(W, G), GL_combined_poly / alpha[:, None]))
        self.G_combined = G_combined
        self.G2I = G2I
        try:
            mpost = self.least_squares_inversion(G2I, d2I, reg=0, A=A, b=b, Aeq=Aeq, beq=beq, lb=lb, ub=ub, x0=x0, opts=opts)
        except:
            return -9999999
        self.mpost = mpost
        mpost = np.hstack((samples[:self.linear_sample_start_position], mpost))
    
        # Compute magnitude log prior if enabled
        magnitude_log_prior_value = 0.0
        if magnitude_log_prior:
            magnitude_log_prior_value = self.compute_magnitude_log_prior(mpost, decay_rate)
    
        # Compute data log likelihood
        linear_sample = mpost[self.linear_sample_start_position:]
        data_log_likelihood = compute_data_log_likelihood(G_combined, linear_sample, self.observations, inv_cov, log_det)
    
        # Compute smooth log likelihood
        linear_sample_slip_only = mpost[self.sample_slip_only_positions]
        smooth_log_likelihood = compute_smooth_log_likelihood(GL_combined, linear_sample_slip_only, alpha)
    
        # Compute (ATA)^-1, where A = log[(GT*inv_cov*G + GL/alpha_2)^(-1/2)]
        ATA = np.dot(G2I.T, G2I)
        ATA_logdet = -1. / 2. * np.linalg.slogdet(ATA)[1]
    
        return data_log_likelihood + smooth_log_likelihood * smooth_prior_weight + magnitude_log_prior_value + ATA_logdet

    def least_squares_inversion(self, C, d, reg=0, A=None, b=None, Aeq=None, beq=None, \
        lb=None, ub=None, x0=None, opts=None):
        '''
            Solve linear constrained l2-regularized least squares. Can
            handle both dense and sparse matrices. Matlab's lsqlin
            equivalent. It is actually wrapper around CVXOPT QP solver.
                min_x ||C*x  - d||^2_2 + reg * ||x||^2_2
                s.t.    A * x <= b
                        Aeq * x = beq
                        lb <= x <= ub
            Input arguments:
                C   is m x n dense or sparse matrix
                d   is n x 1 dense matrix
                reg is regularization parameter
                A   is p x n dense or sparse matrix
                b   is p x 1 dense matrix
                Aeq is q x n dense or sparse matrix
                beq is q x 1 dense matrix
                lb  is n x 1 matrix or scalar
                ub  is n x 1 matrix or scalar
            '''
        # Compute using lsqlin equivalent to the lsqlin in matlab
        opts = {'show_progress': False}
        try:
            ret = lsqlin.lsqlin(C, d, reg, A, b, Aeq, beq, lb, ub, x0, opts)
        except:
            ret = lsqlin.lsqlin(C, d, reg, A, b, None, None, lb, ub, x0, opts)
        mpost = ret['x']
        # Store mpost
        self.mpost = lsqlin.cvxopt_to_numpy_matrix(mpost)

        return self.mpost
    
    def combine_GL_poly(self, GL_combined=None):
        if GL_combined is None:
            GL_combined_poly = []
            for fault in self.multifaults.faults:
                poly_positions = self.poly_positions.get(fault.name, (0, 0))
                # Create a zero matrix with the correct size
                combined = np.zeros((fault.GL.shape[0], fault.GL.shape[1] + poly_positions[1] - poly_positions[0]))
                # Copy the values from the original matrix to the combined matrix at the correct positions
                combined[:, :fault.GL.shape[1]] = fault.GL.toarray()
                GL_combined_poly.append(combined)
            self.GL_combined_poly = scipy.linalg.block_diag(*GL_combined_poly)
    
        return self.GL_combined_poly
    
    @property
    def observations(self):
        if not hasattr(self.multifaults, 'd') or self.multifaults.d is None:
            raise ValueError("multifaults.d is not set. Please call multifaults.assembleGFs first.")
        return self.multifaults.d
    
    @property
    def datanames(self):
        if not self.multifaults.faults:
            raise ValueError("No faults available.")
        return self.multifaults.faults[0].datanames

    @property
    def obs_dict(self):
        if not self.multifaults.faults:
            raise ValueError("No faults available.")
        return self.multifaults.faults[0].d

    @property
    def faultnames(self):
        return self.multifaults.faultnames

    @property
    def multifaults(self):
        return self.config.multifaults

    @multifaults.setter
    def multifaults(self, value):
        self.config.multifaults = value
    
    @property
    def bayesian_sampling_mode(self):
        return self.config.bayesian_sampling_mode
    
    @bayesian_sampling_mode.setter
    def bayesian_sampling_mode(self, value):
        self.config.bayesian_sampling_mode = value

    @property
    def geodata(self):
        return self.config.geodata['data']

    @geodata.setter
    def geodata(self, value):
        self.config.geodata['data'] = value

    @property
    def lb(self):
        return self.bounds_manger.lb

    @lb.setter
    def lb(self, value):
        self.bounds_manger.lb = value

    @property
    def ub(self):
        return self.bounds_manger.ub

    @ub.setter
    def ub(self, value):
        self.bounds_manger.ub = value

    @property
    def sigmas(self):
        return self.config.sigmas['initial_value']

    @sigmas.setter
    def sigmas(self, value):
        if not isinstance(value, list):
            raise ValueError("sigmas must be a list")
        self.config.sigmas['initial_value'] = value

    @property
    def alpha(self):
        return np.array(self.config.alpha['initial_value'])

    @alpha.setter
    def alpha(self, value):
        if not isinstance(value, (int, float, ndarray)):
            raise ValueError("alpha must be a number or a numpy array")
        self.config.alpha['initial_value'] = value

    @property
    def moment_magnitude_threshold(self):
        return self.config.moment_magnitude_threshold

    @moment_magnitude_threshold.setter
    def moment_magnitude_threshold(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("moment_magnitude_threshold must be a number")
        self.config.moment_magnitude_threshold = value

    @property
    def magnitude_tolerance(self):
        return self.config.magnitude_tolerance

    @magnitude_tolerance.setter
    def magnitude_tolerance(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("magnitude_tolerance must be a number")
        self.config.magnitude_tolerance = value

    @property
    def patch_areas(self):
        return self.config.patch_areas

    @patch_areas.setter
    def patch_areas(self, value):
        if not isinstance(value, dict):
            raise ValueError("patch_areas must be a dictionary")
        self.config.patch_areas = value

    @property
    def shear_modulus(self):
        return self.config.shear_modulus

    @shear_modulus.setter
    def shear_modulus(self, value):
        if not isinstance(value, (int, float)):
            raise ValueError("shear_modulus must be a number")
        self.config.shear_modulus = value

    @property
    def nonlinear_inversion(self):
        return self.config.nonlinear_inversion
    
    @nonlinear_inversion.setter
    def nonlinear_inversion(self, value):
        if not isinstance(value, bool):
            raise ValueError("nonlinear_inversion must be a boolean")
        self.config.nonlinear_inversion = value
    
    @property
    def GLs(self):
        return self.config.GLs

    @GLs.setter
    def GLs(self, value):
        if not isinstance(value, dict):
            raise ValueError("GLs must be a dictionary")
        self.config.GLs = value

    def get_geometry(self, fault_name):
        """
        Get the geometry configuration of the fault with the specified name.
        """
        if fault_name not in self.config.faults:
            raise ValueError(f"No such fault: {fault_name}")
        return self.config.faults[fault_name]['geometry']

    def set_geometry(self, fault_name, value):
        """
        Set the geometry configuration of the fault with the specified name.
        """
        if not isinstance(value, dict):
            raise ValueError(f"Geometry configuration must be a dictionary")
        self.config.faults[fault_name]['geometry'] = value

    def get_method_parameters(self, fault_name, method_name):
        """
        Get the parameters configuration of the specified method of the fault with the specified name.
        """
        if fault_name not in self.config.faults:
            raise ValueError(f"No such fault: {fault_name}")
        if method_name not in self.config.faults[fault_name]['method_parameters']:
            raise ValueError(f"No such method: {method_name}")
        return self.config.faults[fault_name]['method_parameters'][method_name]

    def set_method_parameters(self, fault_name, method_name, value):
        """
        Set the parameters configuration of the specified method of the fault with the specified name.
        """
        if not isinstance(value, dict):
            raise ValueError(f"Method parameters configuration must be a dictionary")
        self.config.faults[fault_name]['method_parameters'][method_name] = value

    def __getitem__(self, key):
        """
        Get the configuration of the specified key.
        """
        keys = key.split('/')
        current_config = self.config
        for k in keys:
            if k not in current_config:
                raise ValueError(f"No such key: {k}")
            current_config = current_config[k]
        return current_config

    def __setitem__(self, key, value):
        """
        Set the configuration of the specified key.
        """
        keys = key.split('/')
        current_config = self.config
        for k in keys[:-1]:
            if k not in current_config:
                raise ValueError(f"No such key: {k}")
            current_config = current_config[k]
        if not isinstance(value, type(current_config[keys[-1]])):
            raise ValueError(f"Value must be a {type(current_config[keys[-1]])}")
        current_config[keys[-1]] = value


class MyMultiFaultsInversion(multifaultsolve):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.faults_dict = {fault.name: fault for fault in self.faults}
        # To order the G matrix based on the order of the faults
        self.faultnames = [fault.name for fault in self.faults]

    def _apply_to_faults(self, func, fault_names=None):
        # Apply a function to the specified faults
        if fault_names is None:
            faults = self.faults_dict.values()  # Update all faults if fault_names is None
        else:
            faults = [self.faults_dict[name] for name in fault_names if name in self.faults_dict]  # Update specified faults
        for fault in faults:
            func(fault)

    def update_fault(self, method_name, fault_names=None, *args, **kwargs):
        # Update the specified faults
        def func(fault):
            try:
                method = getattr(fault, method_name)
                method(*args, **kwargs)  # Call the method with the specified arguments
            except AttributeError:
                raise ValueError(f"Fault object has no method named '{method_name}'")

        self._apply_to_faults(func, fault_names)

    def update_fault_geometry(self, method='perturb_bottom_coords', fault_names=None, perturbations=None, **kwargs):
        # Update the geometry of the specified faults using the specified method and perturbations (if any)
        method = kwargs.pop('method', method)
        # Update the geometry of the specified faults
        self.update_fault(method, fault_names=fault_names, perturbations=perturbations, **kwargs)

    def update_mesh(self, method='generate_mesh', fault_names=None, verbose=0, show=False, **kwargs):
        # Update the mesh of the specified faults using the specified method and arguments (if any)
        method = kwargs.pop('method', method)
        # Update the mesh of the specified faults
        self.update_fault(method, fault_names=fault_names, verbose=verbose, show=show, **kwargs)

    def update_GFs(self, geodata=None, verticals=None, fault_names=None, dataFaults=None, method=None):
        # Update the Green's functions of the specified faults
        def func(fault):
            # get the good indexes
            st_row = 0
            sliplist = [slip for slip, char in zip(['strikeslip', 'dipslip', 'tensile', 'coupling'], 'sdtc') if char in fault.slipdir]
            for obsdata, vertical, dataFault in zip(geodata, verticals, dataFaults or [None]*len(geodata)):
                # Determine method based on fault presence in dataFault
                gfmethod = 'empty' if dataFault is not None and fault.name not in dataFault else method
                # print(f"Method for {fault.name}: {gfmethod}")
                fault.buildGFs(obsdata, vertical=vertical, slipdir=fault.slipdir, method=gfmethod, verbose=False)
                st = 0
                for sp in sliplist:
                    Nclocal = fault.G[obsdata.name][sp].shape[1]
                    Nrowlocal = fault.G[obsdata.name][sp].shape[0]

                    fault.Gassembled[st_row:st_row+Nrowlocal, st:st+Nclocal] = fault.G[obsdata.name][sp]
                    st += Nclocal
                st_row += Nrowlocal

            # get the good indexes for self.G
            st, se = self.fault_indexes[fault.name]
            # Store the G matrix
            self.G[:, st:se] = fault.Gassembled

        self._apply_to_faults(func, fault_names)

    def update_Laplacian(self, method='Mudpy', bounds=('free', 'free', 'free', 'free'), 
                         topscale=0.25, bottomscale=0.03, fault_names=None):
        # Update the Laplacian matrix of the specified faults using the specified method and arguments (if any)
        if not hasattr(self, 'GLs') or self.GLs is None:
            self.GLs = {}

        def func(fault):
            lap = fault.buildLaplacian(method=method, bounds=bounds, 
                                       topscale=topscale, bottomscale=bottomscale)
            if len(fault.slipdir) == 1:
                fault.GL = csr_matrix(lap)
            else:
                fault.GL = block_diag([lap for _ in fault.slipdir]).tocsr()

            self.GLs[fault.name] = fault.GL

        self._apply_to_faults(func, fault_names)
    
    def compute_data_inv_covs_and_logdets(self, geodata):
        inv_covs = {}
        chol_decomps = {}  # Store the Cholesky decomposition of the inverse covariance matrix
        logdets = {}
        if not self.faults:
            raise ValueError("No faults available.")
        for idataname, idata in zip(self.faults[0].datanames, geodata):
            if idata.name != idataname:
                raise ValueError(f"Data name mismatch: expected {idataname}, got {idata.name}")
            inv_cov = np.linalg.inv(idata.Cd)
            chol_decomp = np.linalg.cholesky(inv_cov)  # Compute the Cholesky decomposition of the inverse covariance matrix
            det = np.linalg.slogdet(idata.Cd)[1]
            inv_covs[idataname] = inv_cov
            chol_decomps[idataname] = chol_decomp  # Store the Cholesky decomposition of the inverse covariance matrix
            logdets[idataname] = det
        return inv_covs, chol_decomps, logdets  # Return the inverse covariance matrices, Cholesky decompositions, and log determinants

    def compute_fault_areas(self, fault_names=None):
        # Compute the areas of the patches of the specified faults
        if not hasattr(self, 'patch_areas') or self.patch_areas is None:
            self.patch_areas = {}

        def func(fault):
            areas = fault.compute_patch_areas()
            self.patch_areas[fault.name] = areas

        self._apply_to_faults(func, fault_names)
        return self.patch_areas


if __name__ == "__main__":
    pass