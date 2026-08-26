"""
Bayesian Multi-Faults Inversion Module

This module provides a comprehensive framework for Bayesian inversion of slip 
distribution on multiple faults using Sequential Monte Carlo (SMC) sampling methods.
The implementation supports both linear and nonlinear inversions with various 
constraints and priors.

Key Features:
- Sequential Monte Carlo (SMC) sampling for Bayesian inference
- Support for multiple fault geometries with adaptive triangular patches
- Magnitude-constrained slip inversion
- MPI-parallelized sampling for computational efficiency
- Flexible parameter bounds management
- Multiple slip sampling modes (strike-slip/dip-slip, magnitude-rake, rake-fixed)
- Advanced plotting and visualization capabilities

Classes:
    BayesianMultiFaultsInversion: Main class for Bayesian multi-fault slip inversion

Example:
    >>> from eqtools.csiExtend import BayesianMultiFaultsInversion
    >>> # Initialize with configuration
    >>> inverter = BayesianMultiFaultsInversion(
    ...     config="config.yml",
    ...     bounds_config="bounds.yml",
    ...     geodata=geodata,
    ...     faults_list=['fault1', 'fault2'],
    ... )
    >>> # Run sampling
    >>> results = inverter.walk(nchains=1000, chain_length=50)
    >>> # Extract results
    >>> inverter.returnModel(model='median')

Authors:
    Kefeng He

Version:
    1.0.0

Last Updated:
    2025-08-01
"""

import copy

# Standard library imports
import os
import pathlib
import time
import glob
import logging
from collections import namedtuple
from dataclasses import dataclass
from typing import List

# Third-party scientific computing imports
import numpy as np
from numpy import ndarray
import scipy
import scipy.linalg
from scipy.sparse import csr_matrix, block_diag
from scipy.stats import gaussian_kde, truncnorm

# Numerical optimization and acceleration
from numba import njit

# Data format and I/O
import yaml
import h5py

# Parallel computing
from mpi4py import MPI

# Plotting and visualization
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import FuncFormatter, AutoLocator

# CSI library for geodetic data processing
from csi import gps, insar, leveling, crossfaultoffset

# Local imports - utilities and plotting
from ..viztools import normalize_image_format, sci_plot_style

# Local imports - core modules
from .BayesianAdaptiveTriangularPatches import BayesianAdaptiveTriangularPatches as relocfault
from .SMC_MPI import SMC_samples_parallel_mpi
from .config.bayesian_config import (
    BayesianMultiFaultsInversionConfig,
    normalize_bayesian_sampling_mode,
)
from .config.parameter_groups import attach_group_parameters, resolve_group_layout
from .fault_analysis_mixin import FaultAnalysisMixin
from .data_correction_constraints import DataCorrectionConstraintMixin
from .data_correction_report_mixin import DataCorrectionReportMixin
from .deep_slip_loading_mixin import DeepSlipLoadingMixin
from .interseismic_mixin import InterseismicKinematicsMixin
from .patch_indices import normalize_patch_indices
from .constraint_manager_smc import ConstraintManagerSMC
from .multifaults_base import MyMultiFaultsInversion
from .multifaultsolve_boundLSE import _validate_lsqlin_status
from .source_adapters import FaultAdapter
from .plot_product_mixin import FigureProductMixin
from .geom_ops import InvalidFaultGeometryError
from .covariance_utils import gaussian_log_likelihood
from .hyperparameter_reporting import (
    build_geometry_parameter_rows,
    build_scale_parameter_rows,
    format_geometry_parameter_report,
    format_scale_parameter_report,
)
from .quadratic_objective import (
    LeastSquaresBlock,
    assemble_quadratic_objective,
    gaussian_curvature_log_term,
    weighted_residual_quadratic,
)
import warnings
from .bayesian_utils import det_of_laplace_smooth_lu
from . import lsqlin

_INVALID_CONSTRAINED_SOLVE_LOGLIKE = -9999999.0

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
                                whitener: ndarray, sigma: float,
                                log_cov_det: float) -> float:
    """Evaluate a Gaussian data term from a left-whitener.

    ``whitener.T @ whitener`` is the precision represented by this score.
    Keeping the solve and score in whitened space prevents a second explicit
    precision representation from drifting out of alignment.
    """
    simulations = np.dot(G, samples)
    residual = np.subtract(simulations, observations)
    whitened_residual = np.dot(whitener, residual) / sigma
    return -0.5 * (
        np.dot(whitened_residual, whitened_residual) + log_cov_det
    )

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
def compute_log_posterior(
    samples, Gs, observations, lb, ub, whiteners, log_dets, sigmas, alpha, GL
):
    log_prior = compute_log_prior(samples, lb, ub)
    if log_prior == -np.inf:
        return -np.inf
    else:
        data_log_likelihood = 0.0
        for index in range(len(Gs)):
            data_log_likelihood += compute_data_log_likelihood(
                Gs[index],
                samples,
                observations[index],
                whiteners[index],
                sigmas[index],
                log_dets[index] + len(observations[index]) * np.log(sigmas[index] ** 2),
            )
        smooth_log_likelihood = compute_smooth_log_likelihood(GL, samples, alpha)
        return log_prior + data_log_likelihood + smooth_log_likelihood
    
@njit
def compute_magnitude_log_posterior(
    samples, Gs, observations, lb, ub, moment_magnitude_threshold,
    patch_areas, shear_modulus, magnitude_tolerance, whiteners, log_dets,
    sigmas, alpha, GL,
):
    log_prior = compute_log_prior(samples, lb, ub)
    if log_prior == -np.inf:
        return -np.inf
    else:
        log_magnitude_prior = compute_magnitude_log_prior(samples, moment_magnitude_threshold, 
                                                          patch_areas, shear_modulus, magnitude_tolerance)
        if log_magnitude_prior == -np.inf:
            return -np.inf
        else:
            data_log_likelihood = 0.0
            for index in range(len(Gs)):
                data_log_likelihood += compute_data_log_likelihood(
                    Gs[index],
                    samples,
                    observations[index],
                    whiteners[index],
                    sigmas[index],
                    log_dets[index]
                    + len(observations[index]) * np.log(sigmas[index] ** 2),
                )
            smooth_log_likelihood = compute_smooth_log_likelihood(GL, samples, alpha)
            return log_prior + log_magnitude_prior + data_log_likelihood + smooth_log_likelihood

def make_target_for_sampler(Gs: List[ndarray], observations: List[ndarray], lb, ub, 
                            whiteners: List[ndarray], log_dets: List[float], sigmas: List[float], alpha: float, GL: csr_matrix):
    @njit
    def target(samples):
        return compute_log_posterior(samples, Gs, observations, lb, ub, whiteners, log_dets, sigmas, alpha, GL)
    return target

def make_magnitude_target_for_sampler(Gs: List[ndarray], observations: List[ndarray], lb, ub, moment_magnitude_threshold, 
                                      patch_areas, shear_modulus, magnitude_tolerance, 
                                      whiteners: List[ndarray], log_dets: List[float], sigmas: List[float], alpha: float, GL: csr_matrix):
    def target(samples):
        return compute_magnitude_log_posterior(samples, Gs, observations, lb, ub, moment_magnitude_threshold, 
                                               patch_areas, shear_modulus, magnitude_tolerance, 
                                               whiteners, log_dets, sigmas, alpha, GL)
    return target


NT1 = namedtuple('NT1', 'N Neff target LB UB')
_SMCFJOptions = namedtuple(
    '_SMCFJOptions', 'N Neff target LB UB invalid_loglike'
)
# tuple object for the samples
NT2 = namedtuple('NT2', 'allsamples postval beta stage covsmpl resmpl')


@dataclass(frozen=True)
class _SMCFJQuadraticWorkspace:
    """Candidate-independent residual blocks for one fixed geometry state."""

    G_combined: np.ndarray
    data_blocks: tuple
    smoothing_blocks: tuple


class BayesianMultiFaultsInversion(
    DataCorrectionReportMixin,
    DataCorrectionConstraintMixin,
    DeepSlipLoadingMixin,
    InterseismicKinematicsMixin,
    FigureProductMixin,
    FaultAnalysisMixin,
):
    def __init__(self, config="default_config.yml", multifaults=None, geodata=None, faults_list=None, gfmethods=None, 
                 bounds_config='bounds_config.yml', interseismic_config=None, verbose=True, parallel_rank=None):
        if isinstance(config, str):
            assert geodata is not None, "geodata must be provided when config is a file"
            parallel_rank = parallel_rank if parallel_rank is not None else MPI.COMM_WORLD.Get_rank()
            self.config = BayesianMultiFaultsInversionConfig(config, multifaults=multifaults, geodata=geodata, faults_list=faults_list, 
                                                             gfmethods=gfmethods, verbose=verbose, parallel_rank=parallel_rank)
        else:
            self.config = config

        if interseismic_config is None:
            interseismic_config = getattr(self.config, 'interseismic_config_file', None)
        if interseismic_config is not None:
            self.config.load_interseismic_config(interseismic_config)

        self.update_config(self.config)
        self._initialize_bounds(bounds_config)

    def update_config(self, config):
        self.config = config
        # Expand normalized group-space sigma state to ordered data-set space.
        # Sampling still allocates one value per updatable group.
        sigma_layout = self.config.sigmas.get('group_layout')
        if sigma_layout is None:
            sigma_mode = self.config.sigmas.get('mode', 'individual')
            sigma_layout = attach_group_parameters(
                resolve_group_layout(
                    [data.name for data in self.config.geodata['data']],
                    sigma_mode,
                    self.config.sigmas.get('groups')
                    if sigma_mode == 'grouped' else None,
                    member_label='dataset',
                    single_group_name='all',
                    individual_prefix='group_',
                ),
                values=self.config.sigmas.get('initial_value', 0.0),
                update=self.config.sigmas.get('update', True),
                value_name='sigma initial_value',
                default_value=0.0,
            )
        sigma_member_indices = sigma_layout['member_param_indices']
        self._sigma_update_mask = sigma_layout['update_by_group'][sigma_member_indices]
        self._sigma_initial = sigma_layout['values_by_group'][sigma_member_indices]
        self._sigma_update_indices = np.where(self._sigma_update_mask)[0] # indices of sigma to be updated
        self._sigma_update_positions = sigma_layout['sample_index_by_group'][sigma_member_indices]
        self._sigma_update_positions = self._sigma_update_positions[self._sigma_update_indices]
        self._sigma_update_flag = np.any(self._sigma_update_mask) # whether any sigma is to be updated
        
        # Alpha's canonical member space contains only smoothing-capable
        # sources.  Materialize a full-source boundary array for downstream
        # indexing, leaving non-smoothing sources neutral and non-updatable.
        alpha_layout = self.config.alpha.get('group_layout')
        source_names = list(self.config.faultnames)
        if alpha_layout is None:
            configured_smoothing = getattr(
                self.config, '_smoothing_faultnames', None
            )
            smoothing_names = (
                list(configured_smoothing)
                if isinstance(configured_smoothing, (list, tuple))
                else source_names.copy()
            )
            alpha_mode = self.config.alpha.get('mode', 'single')
            alpha_groups = None
            if alpha_mode == 'grouped':
                alpha_groups = self.config.alpha.get('groups')
                if alpha_groups is None:
                    alpha_groups = {
                        f'Event_{index}': members
                        for index, members in enumerate(
                            self.config.alpha.get('faults', [])
                        )
                    }
            alpha_layout = attach_group_parameters(
                resolve_group_layout(
                    smoothing_names,
                    alpha_mode,
                    alpha_groups,
                    member_label='smoothing source',
                    single_group_name='all',
                    individual_prefix='smooth_',
                ),
                values=self.config.alpha.get('initial_value', 0.0),
                update=self.config.alpha.get('update', True),
                value_name='alpha initial_value',
                default_value=0.0,
            )
        self._alpha_update_mask = np.zeros(len(source_names), dtype=bool)
        self._alpha_initial = np.zeros(len(source_names), dtype=float)
        alpha_positions = np.full(len(source_names), -1, dtype=int)
        group_index = {
            name: index for index, name in enumerate(alpha_layout['group_names'])
        }
        for member_name in alpha_layout['member_names']:
            source_index = source_names.index(member_name)
            parameter_index = group_index[alpha_layout['member_to_group'][member_name]]
            self._alpha_update_mask[source_index] = alpha_layout['update_by_group'][parameter_index]
            self._alpha_initial[source_index] = alpha_layout['values_by_group'][parameter_index]
            alpha_positions[source_index] = alpha_layout['sample_index_by_group'][parameter_index]
        self._alpha_update_indices = np.where(self._alpha_update_mask)[0] # indices of alpha to be updated
        self._alpha_update_positions = alpha_positions
        self._alpha_update_positions = self._alpha_update_positions[self._alpha_update_indices]
        self._alpha_update_flag = np.any(self._alpha_update_mask) # whether any alpha is to be updated

        self._update_faults()
        self._calculate_parameters()

    def _dataset_sigmas_from_samples(self, samples):
        """Resolve one physical sigma scale per data set from a sample.

        This is the single layout adapter shared by likelihood evaluation and
        final-model reporting. Fixed groups retain their configured values;
        sampled groups overwrite only their mapped dataset positions.
        """
        sigmas = self._sigma_initial.astype(np.float64, copy=True)
        if self.sigmas_position is not None and len(self._sigma_update_indices) > 0:
            sampled = np.asarray(samples, dtype=float)[
                self.sigmas_position[0]:self.sigmas_position[1]
            ]
            sigmas[self._sigma_update_indices] = sampled[
                self._sigma_update_positions
            ]
        if self.config.sigmas.get('log_scaled', True):
            sigmas = np.power(10.0, sigmas)
        return np.asarray(sigmas, dtype=float)

    def _source_alphas_from_samples(self, samples):
        """Resolve one physical alpha scale per configured source.

        The full-source array is an internal indexing boundary.  Sources
        without a Laplacian retain a neutral placeholder here and are removed
        by :meth:`_smoothing_alphas_from_samples`; they never acquire a
        smoothing parameter.  Fixed groups retain their configured values and
        sampled groups overwrite only their mapped source positions.
        """
        alphas = self._alpha_initial.astype(np.float64, copy=True)
        if self.alpha_position is not None and len(self._alpha_update_indices) > 0:
            sampled = np.asarray(samples, dtype=float)[
                self.alpha_position[0]:self.alpha_position[1]
            ]
            alphas[self._alpha_update_indices] = sampled[
                self._alpha_update_positions
            ]
        if self.config.alpha.get('log_scaled', True):
            alphas = np.power(10.0, alphas)
        return np.asarray(alphas, dtype=float)

    def _smoothing_alphas_from_samples(self, samples):
        """Return physical alpha values in smoothing-source order only."""
        if not self.config.alpha_enabled:
            return np.array([], dtype=float)
        return self._source_alphas_from_samples(samples)[
            self._smoothing_alpha_faults_index
        ]

    def _publish_fit_sigma_context(self, samples):
        """Publish report-only sigma/group metadata for an active model."""
        sigmas = self._dataset_sigmas_from_samples(samples)
        mode = self.config.sigmas.get('mode', 'individual')
        groups = self.config.sigmas.get('groups')
        if mode == 'single':
            group_members = {'all': list(self.datanames)}
        elif mode == 'individual':
            group_members = {
                f'group_{name}': [name] for name in self.datanames
            }
        elif mode == 'grouped':
            group_members = {
                str(name): list(members) for name, members in groups.items()
            }
        else:
            raise ValueError(f"Unknown sigmas mode: {mode}")

        self.current_data_sigmas = dict(zip(self.datanames, sigmas))
        self.current_data_weights = {
            name: 1.0 / sigma
            for name, sigma in self.current_data_sigmas.items()
        }
        self.current_data_sigma_group_members = group_members
        self.current_data_sigma_groups = {
            member: group
            for group, members in group_members.items()
            for member in members
        }
        self.current_data_effective_dof = {}

    def _publish_fit_hyperparameter_context(self, samples):
        """Publish complete physical sigma/alpha values for one active model.

        ``self.model`` remains in sampler space.  These result attributes are
        deliberately name-keyed and physical so fixed groups, sampled groups,
        log scaling, and grouped membership cannot be confused by consumers.
        """
        self._publish_fit_sigma_context(samples)

        smoothing_names = [
            self.multifaults.faults[index].name
            for index in self._smoothing_alpha_faults_index
        ]
        smoothing_alphas = self._smoothing_alphas_from_samples(samples)
        self.current_smoothing_alphas = dict(
            zip(smoothing_names, smoothing_alphas)
        )
        self.current_smoothing_weights = {
            name: 1.0 / alpha
            for name, alpha in self.current_smoothing_alphas.items()
        }

        layout = self.config.alpha.get('group_layout', {})
        members_by_group = layout.get('members_by_group', {})
        self.current_alpha_group_members = {
            str(group): list(members)
            for group, members in members_by_group.items()
        }
        self.current_alpha_group_values = {
            group: self.current_smoothing_alphas[members[0]]
            for group, members in self.current_alpha_group_members.items()
            if members
        }

    def _clear_bayesian_hyperparameter_context(self):
        """Invalidate physical hyperparameters before activating a model."""
        self._clear_fit_weight_context()
        self.current_smoothing_alphas = {}
        self.current_smoothing_weights = {}
        self.current_alpha_group_members = {}
        self.current_alpha_group_values = {}

    def _update_faults(self):
        # Update the faults based on the configuration parameters and method parameters for each fault 
        datanames = [d.name for d in self.config.geodata.get('data', [])]
        Nd = len(datanames)
        faultnames = self.faultnames
        for fault_name, fault_config in self.config.faults.items():
            if fault_name != 'defaults':
                # Update Green's functions
                dataFaults = self.config.dataFaults
                # Check if dataFaults is a list of lists, each equal to faultnames
                if not (isinstance(dataFaults, list) and len(dataFaults) == Nd and all(fault_name in flist for flist in dataFaults)):
                    # Update Green's functions
                    self.multifaults.update_GFs(fault_names=[fault_name], **fault_config['method_parameters']['update_GFs'])
                # Update Laplacian
                self.multifaults.update_Laplacian(fault_names=[fault_name], **fault_config['method_parameters']['update_Laplacian'])

    def _calculate_parameters(self):
        self.Gs = {fault.name: fault.Gassembled for fault in self.multifaults.faults}
        self.data_covariance_metrics = (
            self.multifaults.compute_data_covariance_metrics(self.geodata)
        )
        self.patch_areas = self.multifaults.compute_fault_areas()
        self.GLs = self.multifaults.GLs
        # Filter out sources without GL (Pressure/Sbarbot) to avoid AttributeError
        gl_list = [fault.GL for fault in self.multifaults.faults
                   if hasattr(fault, 'GL') and fault.GL is not None]
        if gl_list:
            self.GL_combined = block_diag(gl_list).toarray()
        else:
            G_cols = sum(fault.Gassembled.shape[1] for fault in self.multifaults.faults)
            self.GL_combined = np.zeros((0, G_cols))
        self.calculate_sigmas_alpha_positions()
        self.calculate_geometry_positions()
        self.calculate_slip_and_poly_positions()
        self.calculate_linear_sample_start_position()
        self.calculate_sample_slip_only_positions()

        # Build smoothing-only source indices for alpha extraction.
        # Stores the SOURCE index (position in self.multifaults.faults) of each
        # smoothing-capable source, so that alpha[_smoothing_alpha_faults_index]
        # correctly picks per-source alpha values from the per-source alpha array
        # built in update_config (which has length = len(faultnames)).
        self._smoothing_alpha_faults_index = [
            i
            for i, fault in enumerate(self.multifaults.faults)
            if hasattr(fault, 'GL') and fault.GL is not None
        ]

        self.combine_GL_poly()

    def _initialize_bounds(self, bounds_config='bounds_config.yml'):
        """Initialize and transactionally compile configured constraints."""
        self.constraint_manager = ConstraintManagerSMC(
            self,
            verbose=self.config.verbose,
        )
        try:
            # Use new unified constraint application method
            self.constraint_manager._apply_constraint_config(
                bounds_config_file=bounds_config,
                encoding='utf-8'
            )
        except FileNotFoundError:
            if self.constraint_manager.verbose:
                print(f"Bounds configuration file '{bounds_config}' not found.")
        except Exception as e:
            if self.constraint_manager.verbose:
                print(f"Error setting bounds from config file: {e}")
            raise

    @property
    def slip_poly_lb(self):
        """Linear parameters lower bounds (always up-to-date)."""
        return self.constraint_manager.get_bounds_for_linear_parameters()[0]

    @property
    def slip_poly_ub(self):
        """Linear parameters upper bounds (always up-to-date)."""
        return self.constraint_manager.get_bounds_for_linear_parameters()[1]

    @property
    def hyper_lb(self):
        """Hyperparameters lower bounds (always up-to-date)."""
        return self.constraint_manager.get_bounds_for_hyperparameters()[0]

    @property
    def hyper_ub(self):
        """Hyperparameters upper bounds (always up-to-date)."""
        return self.constraint_manager.get_bounds_for_hyperparameters()[1]

    @property
    def lb(self):
        """Complete lower bounds array (always up-to-date)."""
        if self.config.bayesian_sampling_mode == 'SMC_FJ':
            return np.concatenate([self.hyper_lb, self.slip_poly_lb])
        else:
            return self.constraint_manager.get_bounds_for_fullsmc()[0]

    @property
    def ub(self):
        """Complete upper bounds array (always up-to-date)."""
        if self.config.bayesian_sampling_mode == 'SMC_FJ':
            return np.concatenate([self.hyper_ub, self.slip_poly_ub])
        else:
            return self.constraint_manager.get_bounds_for_fullsmc()[1]

    def update_bounds(
        self,
        *,
        lb=None,
        ub=None,
        geometry=None,
        sigmas=None,
        alpha=None,
        slip_magnitude_bounds=None,
        rake_angle_bounds=None,
        strikeslip_bounds=None,
        dipslip_bounds=None,
        poly_bounds=None,
    ):
        """Partially update the current coarse bounds declaration.

        ``rake_angle_bounds`` belongs to sampled nonlinear magnitude/rake
        parameterization. It is distinct from fault/patch rake-sector linear
        constraints.
        """
        with self.constraint_manager.atomic_bounds_update():
            if lb is not None or ub is not None:
                self.constraint_manager.set_global_bounds(
                    lb=lb,
                    ub=ub,
                    source="manual_update",
                )

            if geometry is not None or sigmas is not None or alpha is not None:
                self.constraint_manager.set_hyperparameter_bounds(
                    geometry=geometry,
                    sigmas=sigmas,
                    alpha=alpha,
                    source="manual_update",
                )

            if any(
                value is not None
                for value in (
                    slip_magnitude_bounds,
                    rake_angle_bounds,
                    strikeslip_bounds,
                    dipslip_bounds,
                    poly_bounds,
                )
            ):
                self.constraint_manager.set_linear_parameter_bounds(
                    slip_magnitude=slip_magnitude_bounds,
                    rake_angle=rake_angle_bounds,
                    strikeslip=strikeslip_bounds,
                    dipslip=dipslip_bounds,
                    poly=poly_bounds,
                    source="manual_update",
                )
        
        if self.constraint_manager.verbose:
            print("[OK] Bounds updated successfully - all parameters automatically synchronized")

    def update_fault_rake_limits(self, rake_limits, *, source="manual"):
        """Merge fault-level runtime rake sectors and rebuild constraints.

        Existing patch-level rake overrides are preserved and are resolved
        after these fault-level defaults.
        """
        result = self.constraint_manager.update_fault_rake_limits(
            rake_limits,
            replace=False,
            source=source,
            sync=False,
        )
        self.constraint_manager.get_combined_inequality_constraints()
        return result

    def replace_fault_rake_limits(self, rake_limits, *, source="manual"):
        """Replace all fault-level runtime rake sectors."""
        result = self.constraint_manager.update_fault_rake_limits(
            rake_limits,
            replace=True,
            source=source,
            sync=False,
        )
        self.constraint_manager.get_combined_inequality_constraints()
        return result

    def clear_fault_rake_limits(self):
        """Clear script-side fault rake sectors, preserving config/patch rake."""
        result = self.constraint_manager.clear_fault_rake_limits(sync=False)
        self.constraint_manager.get_combined_inequality_constraints()
        return result

    def set_fixed_rake_constraints(self, fixed_rake):
        """Replace fixed-rake equalities in the active SMC_FJ linear block."""
        with self.constraint_transaction():
            result = self.constraint_manager.set_fixed_rake_constraints(
                fixed_rake
            )
            self.constraint_manager.get_combined_equality_constraints()
            return result

    def clear_fixed_rake_constraints(self):
        """Remove fixed-rake equalities; repeated calls are safe."""
        result = self.constraint_manager.clear_fixed_rake_constraints()
        self.constraint_manager.get_combined_equality_constraints()
        return result

    def add_patch_constraints(self, patch_constraints, *, source="manual"):
        """Add patch-level bounds/rake override specs before sampling."""
        result = self.constraint_manager.add_patch_constraints(
            patch_constraints,
            source=source,
            sync=False,
        )
        self.constraint_manager.get_combined_inequality_constraints()
        self.constraint_manager.get_combined_equality_constraints()
        return result

    def replace_patch_constraints(self, patch_constraints, *, source="manual"):
        """Replace script-side patch overrides while keeping config rules."""
        result = self.constraint_manager.replace_patch_constraints(
            patch_constraints,
            source=source,
            sync=False,
        )
        self.constraint_manager.get_combined_inequality_constraints()
        self.constraint_manager.get_combined_equality_constraints()
        return result

    def clear_patch_constraints(self, *, source="manual"):
        """Remove script-side patch overrides while keeping config rules."""
        result = self.constraint_manager.clear_patch_constraints(
            source=source,
            sync=False,
        )
        self.constraint_manager.get_combined_inequality_constraints()
        self.constraint_manager.get_combined_equality_constraints()
        return result

    def get_constraint_snapshot(self, include_matrices=False, validate=False):
        """Return compact manager-owned bounds and constraint diagnostics."""
        return self.constraint_manager.get_constraint_snapshot(
            include_matrices=include_matrices,
            validate=validate,
        )
    
    def update_interseismic_config(self, interseismic_config, reapply=True):
        """Load a new interseismic config and optionally rebuild its constraints."""
        with self.constraint_transaction():
            parsed = self.config.load_interseismic_config(interseismic_config)
            if reapply:
                self.constraint_manager._remove_groups_by_owner(
                    'config',
                    families={
                        'interseismic_blocks',
                        'interseismic_cap',
                        'interseismic_backslip',
                    },
                )
                if parsed.get('blocks', {}).get('enabled', False):
                    self.constraint_manager.apply_interseismic_block_constraints()
                self.constraint_manager.add_euler_cap_constraints()
                self.constraint_manager.apply_interseismic_backslip_constraints()
                self.constraint_manager.get_combined_inequality_constraints()
                self.constraint_manager.get_combined_equality_constraints()
            return parsed

    def update_euler_cap_constraint(
        self,
        fault_name,
        *,
        selector=None,
        max_coupling=None,
        mode=None,
        min_loading_abs=None,
        enabled=None,
        reapply=True,
    ):
        """Update optional cap-constraint selector for one fault.

        Parameters
        ----------
        fault_name : str
            Fault whose cap constraint should be updated.
        selector : dict or iterable of int, optional
            Patch selector for cap rows only.  It does not affect tectonic
            loading-rate calculation.
        max_coupling : float, optional
            Upper multiplier ``k`` in ``|backslip| <= k * |loading|``.
            Defaults to the value already stored in config, or 1.0.
        mode : {"motion_sense", "loading_sign"}, optional
            Cap construction mode.  ``motion_sense`` is the default and works
            with estimated Euler loading.  ``loading_sign`` requires fixed
            loading and constrains ``0 <= -q / b <= k`` from the projected sign.
        min_loading_abs : float, optional
            Minimum absolute loading accepted by ``mode="loading_sign"``.
        enabled : bool, optional
            If provided, update ``cap_constraints.enabled``.
        reapply : bool, default True
            If True, rebuild the cap constraint matrix after updating config.

        Returns
        -------
        dict
            Current parsed ``config.interseismic_config`` dictionary.
        """
        faults_dict = getattr(getattr(self, 'multifaults', None), 'faults_dict', None)
        if faults_dict is None:
            faults_dict = {fault.name: fault for fault in getattr(self, 'faults', [])}
        if fault_name not in faults_dict:
            raise ValueError(f"Fault '{fault_name}' not found. Available: {list(faults_dict.keys())}")

        interseismic = copy.deepcopy(getattr(self.config, 'interseismic_config', {}))
        cap = interseismic.setdefault('cap_constraints', {})
        if enabled is not None:
            cap['enabled'] = bool(enabled)
        cap.setdefault('faults', {})
        cap['faults'].setdefault(fault_name, {})
        if selector is not None:
            if isinstance(selector, (list, tuple, np.ndarray)):
                indices = normalize_patch_indices(
                    faults_dict[fault_name],
                    selector,
                    allow_none_all=False,
                    unique=True,
                    name=f"cap selector for fault '{fault_name}'",
                )
                selector = {'patches': indices.tolist()}
            cap['faults'][fault_name]['selector'] = selector
        if max_coupling is not None:
            max_coupling = float(max_coupling)
            if max_coupling < 0.0:
                raise ValueError("max_coupling must be non-negative")
            cap['faults'][fault_name]['max_coupling'] = max_coupling
        if mode is not None:
            cap['faults'][fault_name]['mode'] = str(mode)
        if min_loading_abs is not None:
            min_loading_abs = float(min_loading_abs)
            if min_loading_abs < 0.0:
                raise ValueError("min_loading_abs must be non-negative")
            cap['faults'][fault_name]['min_loading_abs'] = min_loading_abs
        return self.update_interseismic_config(interseismic, reapply=reapply)
    
    def add_linear_inequality_constraint(
        self, A, b, *, name, source="user"
    ):
        """
        Add one user-owned inequality group ``A @ x <= b``.
        
        Parameters:
        -----------
        A : np.ndarray
            Constraint matrix (n_constraints × n_linear_params)
        b : np.ndarray
            Constraint vector (n_constraints,)
        name : str
            Constraint name
        source : str
            Constraint source description
        """
        if not self.constraint_manager._is_smc_fj_mode():
            raise RuntimeError(
                "Adding a linear inequality requires SMC_FJ with "
                "ss_ds sampling"
            )
        
        if self.constraint_manager.verbose:
            print(f"[+] Adding linear inequality constraint '{name}'...")
        
        try:
            with self.constraint_transaction():
                self.constraint_manager._register_inequality_group(
                    A,
                    b,
                    name,
                    source,
                    owner='user',
                )
                self.constraint_manager.get_combined_inequality_constraints()

                if self.constraint_manager.verbose:
                    matrix_shape = np.asarray(A).shape
                    vector_shape = np.asarray(b).shape
                    print(f"[OK] Added inequality constraint '{name}' ({matrix_shape[0]} constraints)")
                    print(f"   Matrix shape: {matrix_shape}, Vector shape: {vector_shape}")
                    print(f"   Source: {source}")
        except Exception as e:
            if self.constraint_manager.verbose:
                print(f"[X] Failed to add inequality constraint '{name}': {e}")
            raise

    def replace_linear_inequality_constraint(
        self, A, b, *, name, source="user"
    ):
        """Replace one existing user-owned inequality group."""
        if not self.constraint_manager._is_smc_fj_mode():
            raise RuntimeError(
                "Replacing a linear inequality requires SMC_FJ with "
                "ss_ds sampling"
            )
        with self.constraint_transaction():
            self.constraint_manager._replace_user_group(
                'inequality',
                A,
                b,
                name=name,
                source=source,
            )
            self.constraint_manager.get_combined_inequality_constraints()

    def add_linear_equality_constraint(
        self, Aeq, beq, *, name, source="user"
    ):
        """
        Add one user-owned equality group ``A @ x = b``.
        
        Parameters:
        -----------
        Aeq : np.ndarray
            Constraint matrix (n_constraints × n_linear_params)
        beq : np.ndarray
            Constraint vector (n_constraints,)
        name : str
            Constraint name
        source : str
            Constraint source description
        """
        if not self.constraint_manager._is_smc_fj_mode():
            raise RuntimeError(
                "Adding a linear equality requires SMC_FJ with ss_ds "
                "sampling"
            )
        
        if self.constraint_manager.verbose:
            print(f"[==] Adding linear equality constraint '{name}'...")
        
        try:
            with self.constraint_transaction():
                self.constraint_manager._register_equality_group(
                    Aeq,
                    beq,
                    name,
                    source,
                    owner='user',
                )
                self.constraint_manager.get_combined_equality_constraints()

                if self.constraint_manager.verbose:
                    matrix_shape = np.asarray(Aeq).shape
                    vector_shape = np.asarray(beq).shape
                    print(f"[OK] Added equality constraint '{name}' ({matrix_shape[0]} constraints)")
                    print(f"   Matrix shape: {matrix_shape}, Vector shape: {vector_shape}")
                    print(f"   Source: {source}")
        except Exception as e:
            if self.constraint_manager.verbose:
                print(f"[X] Failed to add equality constraint '{name}': {e}")
            raise

    def replace_linear_equality_constraint(
        self, Aeq, beq, *, name, source="user"
    ):
        """Replace one existing user-owned equality group."""
        if not self.constraint_manager._is_smc_fj_mode():
            raise RuntimeError(
                "Replacing a linear equality requires SMC_FJ with ss_ds "
                "sampling"
            )
        with self.constraint_transaction():
            self.constraint_manager._replace_user_group(
                'equality',
                Aeq,
                beq,
                name=name,
                source=source,
            )
            self.constraint_manager.get_combined_equality_constraints()

    def remove_linear_constraint(self, name):
        """Remove one user-owned raw linear group."""
        with self.constraint_transaction():
            self.constraint_manager._remove_group(name)
            self.constraint_manager.get_combined_inequality_constraints()
            self.constraint_manager.get_combined_equality_constraints()

    def get_linear_parameter_layout(self):
        """Return the validated SMC linear-suffix or inactive layout."""
        return self.constraint_manager.get_linear_parameter_layout()

    def constraint_transaction(self):
        """Return an advanced all-or-nothing constraint-update context."""
        return self.constraint_manager.constraint_transaction()

    def apply_constraints_from_config(
        self,
        bounds_config_file,
        *,
        encoding="utf-8",
    ):
        """Transactionally replace configuration-owned declarations."""
        return self.constraint_manager._apply_constraint_config(
            bounds_config_file=bounds_config_file,
            encoding=encoding,
        )

    def set_incompressibility_constraints(self, source_names=None):
        """Set incompressibility equality constraints for Sbarbot sources.

        For each volume element: eps11 + eps22 + eps33 = 0.

        Only effective in SMC_FJ mode with ss_ds slip sampling.

        Parameters
        ----------
        source_names : str or list of str, optional
            Sbarbot source name(s) to constrain. ``None`` applies to all
            Sbarbot sources.
        """
        if not self.constraint_manager._is_smc_fj_mode():
            raise RuntimeError(
                "Incompressibility constraints require SMC_FJ with ss_ds "
                "sampling"
            )

        if not hasattr(self.multifaults, 'adapters'):
            raise RuntimeError("Adapters not initialised on multifaults")

        if source_names is None:
            source_names = [f.name for f in self.multifaults.faults
                            if self.multifaults.adapters[f.name].source_type == 'Sbarbot']
        elif isinstance(source_names, str):
            source_names = [source_names]

        if not source_names:
            if self.constraint_manager.verbose:
                print("[!]  No Sbarbot sources found for incompressibility constraints")
            return

        layout = self.get_linear_parameter_layout()
        linear_start = layout['global_offset']
        n_linear = layout['width']

        with self.constraint_transaction():
            self.constraint_manager._remove_groups_by_owner(
                'managed',
                families={'incompressibility'},
            )
            for sname in source_names:
                adapter = self.multifaults.adapters[sname]
                if adapter.source_type != 'Sbarbot':
                    raise TypeError(f"'{sname}' is not a Sbarbot source")
                param_start = self.slip_positions[sname][0] - linear_start
                cfg = {
                    'incompressible': {
                        'type': 'equality',
                        'rule': 'incompressible',
                    }
                }
                for cname, A, b in adapter.generate_source_equality_constraints(
                    cfg,
                    param_start,
                    n_linear,
                ):
                    full_name = f"src_{sname}_{cname}"
                    self.constraint_manager._register_equality_group(
                        A,
                        b,
                        name=full_name,
                        source=f"incompressibility/{sname}",
                        replace=True,
                        owner='managed',
                        family='incompressibility',
                    )

    def clear_incompressibility_constraints(self):
        """Remove the complete managed incompressibility family."""
        removed = self.constraint_manager._remove_groups_by_owner(
            'managed',
            families={'incompressibility'},
        )
        return tuple(name for _, name in removed)

    @staticmethod
    def _normalize_fault_slip_component(component):
        comp = str(component).lower().replace(' ', '').replace('_', '')
        if comp in ('strikeslip', 'ss', 's', 'strike'):
            return 'strikeslip'
        if comp in ('dipslip', 'ds', 'd', 'dip'):
            return 'dipslip'
        raise ValueError(
            f"Unknown slip component '{component}'. Please use 'strikeslip' or 'dipslip'."
        )

    def add_zero_edge_slip_constraint(self, fault_names, edges, slip_modes):
        """
        Add zero-slip equality constraints for triangles on specified fault edges.

        Builds a constraint matrix per (fault, edge, slip_mode) combination and
        registers one equality group per combination instead of looping
        triangle by triangle.

        Parameters
        ----------
        fault_names : str or list of str
            Fault name(s) to constrain.
        edges : str or list of str
            Edge name(s), e.g. 'top', 'bottom', 'left', 'right'.
        slip_modes : str or list of str
            Slip mode(s) to zero out (case-insensitive, spaces/underscores ignored).
            Strike-slip aliases: 'strikeslip', 'strike_slip', 'strike slip', 'ss'.
            Dip-slip   aliases: 'dipslip',    'dip_slip',    'dip slip',    'ds'.

        Examples
        --------
        # Zero both slip components on the top edge of one fault
        inversion.add_zero_edge_slip_constraint(
            'Aheqi_2025', 'top', ['strikeslip', 'dipslip'])

        # Zero dip-slip on top and bottom edges of two faults
        inversion.add_zero_edge_slip_constraint(
            ['FaultA', 'FaultB'], ['top', 'bottom'], 'dip slip')
        """
        if not self.constraint_manager._is_smc_fj_mode():
            raise RuntimeError(
                "Zero-edge slip constraints require SMC_FJ with ss_ds "
                "sampling"
            )

        if isinstance(fault_names, str):
            fault_names = [fault_names]
        if isinstance(edges, str):
            edges = [edges]
        if isinstance(slip_modes, str):
            slip_modes = [slip_modes]

        slip_modes = list(dict.fromkeys(self._normalize_fault_slip_component(m) for m in slip_modes))

        groups = []
        for fault_name in fault_names:
            if fault_name not in self.multifaults.faults_dict:
                raise ValueError(
                    f"Fault '{fault_name}' not found. Available: {list(self.multifaults.faults_dict.keys())}"
                )
            # Explicit Fault-type guard: zero_edge_slip only makes sense for Fault sources
            if self.constraint_manager._get_source_type(fault_name) != 'Fault':
                raise ValueError(
                    f"zero_edge_slip constraint only applies to Fault sources, "
                    f"but '{fault_name}' is type '{self.constraint_manager._get_source_type(fault_name)}'"
                )
            fault = self.multifaults.faults_dict[fault_name]

            if not hasattr(fault, 'edge_triangles_indices'):
                raise AttributeError(
                    f"Fault '{fault_name}' has no 'edge_triangles_indices'. "
                    "Run edge detection first."
                )

            for edge in edges:
                if edge not in fault.edge_triangles_indices:
                    available = list(fault.edge_triangles_indices.keys())
                    raise KeyError(
                        f"Edge '{edge}' not found in fault '{fault_name}'. "
                        f"Available: {available}"
                    )
                tri_indices = np.asarray(fault.edge_triangles_indices[edge])

                for slip_mode in slip_modes:
                    global_indices = (
                        self.constraint_manager
                        ._get_component_columns_for_patches(
                            fault_name,
                            slip_mode,
                            tri_indices,
                            space='active_linear',
                        )
                    )
                    n_constrained = len(global_indices)

                    # number of linear parameters
                    # lsq_parameters = self.mcmc_samples - self.linear_sample_start_position
                    A = np.zeros((n_constrained, self.lsq_parameters))
                    A[np.arange(n_constrained), global_indices] = 1.0
                    b = np.zeros(n_constrained)

                    name = f"zero_edge_{fault_name}_{edge}_{slip_mode}"
                    groups.append((name, A, b))

        with self.constraint_transaction():
            for name, A, b in groups:
                self.constraint_manager._register_equality_group(
                    A,
                    b,
                    name=name,
                    source="zero_edge_slip",
                    owner="user",
                )
            self.constraint_manager.get_combined_equality_constraints()
        return [name for name, _, _ in groups]
    
    def add_patch_slip_constraint(self, fault_patches, slip_component, value=0.0, constraint_type='equality', operator='=='):
        """
        Set slip constraints for specific sub-fault patches.

        This method allows setting equality (e.g., slip = 0) or inequality 
        (e.g., slip >= 0) constraints for the strike-slip or dip-slip 
        components of a given set of patches.
        
        Only effective in SMC_FJ mode with ss_ds slip sampling.

        Parameters
        ----------
        fault_patches : dict
            Dictionary mapping fault names to lists of patch indices.
            Format: {'fault_name': [patch_idx1, patch_idx2, ...]}
        slip_component : str or list of str
            Slip component(s) to constrain. Can be 'strikeslip' or 'dipslip'.
            Aliases such as 'ss' and 'ds' are also accepted.
        value : float, optional
            The constraint value. Default is 0.0.
        constraint_type : str, optional
            Type of constraint: 'equality' or 'inequality'. Default is 'equality'.
        operator : str, optional
            Operator used for inequality constraints ('<=' or '>=').
            Ignored for equality constraints. Default is '=='.
        """
        if not self.constraint_manager._is_smc_fj_mode():
            raise RuntimeError(
                "Patch slip constraints require SMC_FJ with ss_ds sampling"
            )

        if isinstance(slip_component, str):
            slip_components = [slip_component]
        else:
            slip_components = list(slip_component)

        all_linear_indices = []

        for f_name, patch_indices in fault_patches.items():
            if f_name not in self.multifaults.faults_dict:
                raise ValueError(f"Fault '{f_name}' not found. Available faults: {list(self.multifaults.faults_dict.keys())}")

            for s_comp in slip_components:
                columns = (
                    self.constraint_manager
                    ._get_component_columns_for_patches(
                        f_name,
                        s_comp,
                        patch_indices,
                        space='active_linear',
                    )
                )
                all_linear_indices.extend(columns.tolist())

        n_constrained = len(all_linear_indices)
        if n_constrained == 0:
            return

        A = np.zeros((n_constrained, self.lsq_parameters))
        A[np.arange(n_constrained), all_linear_indices] = 1.0
        b = np.full(n_constrained, value)

        f_name_str = "_".join(fault_patches.keys())[:20]
        c_name_str = "_".join(slip_components)[:15]
        name = f"patch_slip_constraint_{f_name_str}_{c_name_str}"

        if constraint_type == 'equality':
            self.add_linear_equality_constraint(
                A,
                b,
                name=name,
                source='manual',
            )
        elif constraint_type == 'inequality':
            if operator in ('<=', '<'):
                # A*x <= b is the standard form
                pass
            elif operator in ('>=', '>'):
                # A*x >= b  => -A*x <= -b
                A = -A
                b = -b
            else:
                raise ValueError(f"Unsupported inequality operator '{operator}'. Please use '<=' or '>='.")
            self.add_linear_inequality_constraint(
                A,
                b,
                name=name,
                source='manual',
            )
        else:
            raise ValueError(f"Invalid constraint type '{constraint_type}'. Please use 'equality' or 'inequality'.")

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

    def walk(self, nchains=None, chain_length=None, samples=None, magprior=False, comm=None, filename='samples_smc.h5',
             save_every=1, save_at_interval=False, save_at_final=True, covariance_epsilon=1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0,
             sliplb=None, slipub=None, rake_angle=None, rake_sigma=None, rake_range=None, magposteriors=False,
             log_enabled=False, decay_rate=0.1, run_bayesian=True, **kwargs):
        """
        General entry point for SMC sampling, dispatching to the appropriate method based on the bayesian_sampling_mode.
    
        Parameters:
        nchains (int): Number of chains for the SMC sampling. Default is 100.
        chain_length (int): Length of each chain. Default is 50.
        samples (array): Initial samples for the SMC sampling. If None, samples are generated uniformly between the lower and upper bounds.
        magprior (bool): If True, use magnitude-aware initial samples. Default is False.
        comm (MPI.Comm): MPI communicator. If None, MPI.COMM_WORLD is used.
        filename (str): Name of the file where the final samples are saved. Default is 'samples_smc.h5'.
        save_every (int): Frequency at which the samples are saved. Default is 1.
        save_at_interval (bool): If True, save samples at regular intervals. Default is False.
        save_at_final (bool): If True, save samples at the end of the walk. Default is True.
        covariance_epsilon (float): Epsilon value for the covariance matrix. Default is 1e-6.
        amh_a (float): Parameter 'a' for the Adaptive Metropolis-Hastings algorithm. Default is 1.0/9.0.
        amh_b (float): Parameter 'b' for the Adaptive Metropolis-Hastings algorithm. Default is 8.0/9.0.
        sliplb (dict): Lower bounds for each fault. If None, use lb in self.constraint_manager.
        slipub (dict): Upper bounds for each fault. If None, use ub in self.constraint_manager.
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
    
        if mode == 'SMC_FJ':
            return self.walk_smc_fj(nchains=nchains, chain_length=chain_length, samples=samples, comm=comm, filename=filename,
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

    def walk_smc(self, nchains=None, chain_length=None, samples=None, magprior=False, comm=None, filename='samples_smc.h5',
                 save_every=1, save_at_interval=False, save_at_final=True, covariance_epsilon=1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0,
                 sliplb=None, slipub=None, rake_angle=None, rake_sigma=None, rake_range=None, magposteriors=False,
                 log_enabled=False, decay_rate=0.1, run_bayesian=True):
        """
        Perform a Sequential Monte Carlo (SMC) sampling walk.
    
        Parameters:
        nchains (int): Number of chains for the SMC sampling. Default is 100.
        chain_length (int): Length of each chain. Default is 50.
        samples (array): Initial samples for the SMC sampling. If None, samples are generated uniformly between the lower and upper bounds.
        magprior (bool): If True, use magnitude-aware initial samples. Default is False.
        comm (MPI.Comm): MPI communicator. If None, MPI.COMM_WORLD is used.
        filename (str): Name of the file where the final samples are saved. Default is 'samples_smc.h5'.
        save_every (int): Frequency at which the samples are saved. Default is 1.
        save_at_interval (bool): If True, save samples at regular intervals. Default is False.
        save_at_final (bool): If True, save samples at the end of the walk. Default is True.
        covariance_epsilon (float): Epsilon value for the covariance matrix. Default is 1e-6.
        amh_a (float): Parameter 'a' for the Adaptive Metropolis-Hastings algorithm. Default is 1.0/9.0.
        amh_b (float): Parameter 'b' for the Adaptive Metropolis-Hastings algorithm. Default is 8.0/9.0.
        sliplb (dict): Lower bounds for each fault. If None, use lb in self.constraint_manager.
        slipub (dict): Upper bounds for each fault. If None, use ub in self.constraint_manager.
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
        bounds_snapshot = self._fullsmc_bounds_snapshot
    
        if not run_bayesian:
            return None
    
        if rank == 0:
            self.print_parameter_discribution()
            # print('Total samples:', self.total_samples)
            # self.print_parameter_positions()
            print('Number of MCMC samples:', self.mcmc_samples)
            self.print_mcmc_parameter_positions()
    
        if (
            self.constraint_manager.state_revision
            != bounds_snapshot['state_revision']
        ):
            raise RuntimeError(
                "FULLSMC bounds changed after target construction. "
                "Rebuild the target or call walk_smc() again before sampling."
            )
        opt = NT1(
            nchains,
            chain_length,
            self.target,
            bounds_snapshot['lb'].copy(),
            bounds_snapshot['ub'].copy(),
        )
    
        if samples is None:
            if magprior:
                samples = self.prior_samples_vectorize(
                    self.target,
                    nchains,
                    sliplb=sliplb,
                    slipub=slipub,
                    rake_angle=rake_angle,
                    rake_sigma=rake_sigma,
                    rake_range=rake_range,
                )
            else:
                samples = NT2(None, None, None, None, None, None)
    
        # run the SMC sampling
        final = SMC_samples_parallel_mpi(opt, samples, NT1, NT2, comm, save_at_final, 
                                         save_every, save_at_interval, covariance_epsilon, amh_a, amh_b)
        self.sampler = final
        if rank == 0:
            self.save2h5(final, filename)
        
        return final

    def walk_smc_fj(self, nchains=None, chain_length=None, samples=None, comm=None, filename='samples_smc.h5',
                 save_every=1, save_at_interval=False, save_at_final=True, covariance_epsilon=1e-6, amh_a=1.0/9.0, amh_b=8.0/9.0,
                 log_enabled=False, x0=None, opts=None, smooth_prior_weight=1.0,
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
    
        self.target = self.make_smc_fj_target_for_parallel(log_enabled=log_enabled,
                                                        x0=x0, opts=opts, smooth_prior_weight=smooth_prior_weight,
                                                        magnitude_log_prior=magnitude_log_prior, decay_rate=decay_rate)
    
        if not run_bayesian:
            return None
    
        if rank == 0:
            self.print_parameter_discribution()
            # print('Total samples:', self.total_samples)
            # self.print_parameter_positions()
            print('Number of MCMC samples:', self.mcmc_samples)
            self.print_mcmc_parameter_positions()

        hyper_lb, hyper_ub = self.constraint_manager.get_bounds_for_hyperparameters()
        opt = _SMCFJOptions(
            nchains,
            chain_length,
            self.target,
            hyper_lb,
            hyper_ub,
            invalid_loglike=_INVALID_CONSTRAINED_SOLVE_LOGLIKE,
        )
    
        if samples is None:
            samples = NT2(None, None, None, None, None, None)
    
        # run the SMC sampling
        final = SMC_samples_parallel_mpi(
            opt, samples, _SMCFJOptions, NT2, comm, save_at_final,
            save_every, save_at_interval, covariance_epsilon, amh_a, amh_b,
        )
        self.sampler = final
        if rank == 0:
            self.save2h5(final, filename)
        
        return final
    
    def returnModel(
        self,
        model='mean',
        recal_target=False,
        print_stat=True,
        *,
        print_fit_statistics=None,
    ):
        """Activate one posterior representative and distribute its results.

        ``print_fit_statistics`` is the canonical result-API spelling;
        ``print_stat`` remains accepted for existing scripts.
        """
        from scipy.stats import gaussian_kde
        if print_fit_statistics is not None:
            print_stat = bool(print_fit_statistics)
        self._clear_bayesian_hyperparameter_context()
        if recal_target or not hasattr(self, 'target'):
            if self.config.bayesian_sampling_mode == 'SMC_FJ':
                self.target = self.make_smc_fj_target_for_parallel(log_enabled=False)
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
        
        if self.bayesian_sampling_mode == 'SMC_FJ':
            if isinstance(model, str) and model == 'std':
                mpost = []
                for isample in self.sampler.allsamples:
                    self.target(isample)
                    self._require_current_linear_solution('returnModel(std)')
                    mpost.append(self.mpost)
                specs_slip_poly = np.std(mpost, axis=0)
                specs_full = np.hstack((specs[:self.linear_sample_start_position], specs_slip_poly))
                self.target(specs[:self.linear_sample_start_position])
                self._require_current_linear_solution('returnModel(std)')
            else:
                self.target(specs)
                self._require_current_linear_solution(f'returnModel({model})')
                specs_slip_poly = self.mpost
                specs_full = np.hstack((specs[:self.linear_sample_start_position], specs_slip_poly))
            specs = specs_full
            self.model = specs_full
            mpost_tmp = self.mpost.copy()
        else:
            self.G_combined = np.hstack([fault.Gassembled for fault in self.multifaults.faults])
            if self.config.slip_sampling_mode == 'rake_fixed':
                expanded_specs = self.transfer_samples(specs)
                mpost_tmp = expanded_specs[self.linear_sample_start_position:].copy()
            else:
                mpost_tmp = specs[self.linear_sample_start_position:].copy()
            self.mpost = mpost_tmp
        
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
            
            full_slip_start, full_slip_end = self.full_slip_positions[fault.name]
            slip_start, slip_end = full_slip_start, full_slip_end
            slip_start -= total_half
            slip_end -= total_half

            # Get adapter for type-safe result distribution
            _adapter = None
            if hasattr(self.multifaults, 'adapters') and fault.name in self.multifaults.adapters:
                _adapter = self.multifaults.adapters[fault.name]

            if _adapter is not None and _adapter.source_type != 'Fault':
                # Non-Fault sources: distribute parameters directly via adapter
                print(f"  Slip positions: [{slip_start}, {slip_end}]")
                mpost_segment = specs[slip_start:slip_end]
                _adapter.distribute_results(mpost_segment)
            elif self.config.slip_sampling_mode == 'rake_fixed':
                compact_start, compact_end = self.constraint_manager.sample_slip_positions[
                    fault.name
                ]
                print(f"  Slip positions: [{compact_start}, {compact_end}]")
                ss_ds = expanded_specs[full_slip_start:full_slip_end]
                if _adapter is not None:
                    _adapter.distribute_results(ss_ds)
                else:
                    half = len(ss_ds) // 2
                    fault.slip[:, :2] = np.vstack([
                        ss_ds[:half], ss_ds[half:]
                    ]).T

                total_half += full_slip_end - full_slip_start - (
                    compact_end - compact_start
                )
            elif self.config.slip_sampling_mode == 'magnitude_rake':
                half = (slip_end - slip_start) // 2
                print(f"  Slip magnitude positions: [{slip_start}, {slip_start + half}]")
                print(f"  Rake positions: [{slip_start + half}, {slip_end}]")
                slip_mag = specs[slip_start:slip_start + half]
                rake = specs[slip_start + half:slip_end]
                ss = slip_mag*np.cos(np.radians(rake))
                ds = slip_mag*np.sin(np.radians(rake))
                if _adapter is not None:
                    _adapter.distribute_results(np.hstack([ss, ds]))
                else:
                    fault.slip[:, :2] = np.vstack([ss, ds]).T

                linear_start = self.linear_sample_start_position
                mpost_tmp[slip_start-linear_start:slip_end-linear_start] = np.hstack([ss, ds])
            else:
                print(f"  Slip positions: [{slip_start}, {slip_end}]")
                if _adapter is not None:
                    _adapter.distribute_results(specs[slip_start:slip_end])
                else:
                    fault.slip[:, :2] = specs[slip_start:slip_end].reshape(2, -1).T

            full_poly_start, full_poly_end = self.full_poly_positions[fault.name]
            if self.config.slip_sampling_mode == 'rake_fixed':
                poly_start, poly_end = self.constraint_manager.sample_poly_positions[
                    fault.name
                ]
                poly_values = expanded_specs[full_poly_start:full_poly_end]
            else:
                poly_start = full_poly_start - total_half
                poly_end = full_poly_end - total_half
                poly_values = specs[poly_start:poly_end]
            if poly_start != poly_end:
                print(f"  Poly positions: [{poly_start}, {poly_end}]")
            poly_offset = 0
            for i, (key, value) in enumerate(fault.poly.items()):
                if value is not None:
                    fault.polysol[key] = poly_values[
                        poly_offset:poly_offset + value
                    ]
                    poly_offset += value

        if self._sigma_update_flag:
            sigmas_start, sigmas_end = self.sigmas_position
            print(f"Sigmas position: [{sigmas_start}, {sigmas_end}]")
        if self._alpha_update_flag:
            alpha_start, alpha_end = self.alpha_position
            print(f"Alpha position: [{alpha_start}, {alpha_end}]")
        
        if (not isinstance(model, str)) or (model not in ('std', 'STD', 'Std')):
            self._publish_fit_hyperparameter_context(specs)
            # Predict the data and print the RMS and VR
            # Caluculate RMS and VR for the solution and print the results
            rms = np.sqrt(np.mean((np.dot(self.G_combined, mpost_tmp) - self.observations)**2))
            vr = (1 - np.sum((np.dot(self.G_combined, mpost_tmp) - self.observations)**2) / np.sum(self.observations**2)) * 100
            vr = max(vr, 0.0)  # Ensure VR is not negative
            # self.combine_GL_poly()
            roughness = np.dot(self.GL_combined_poly, mpost_tmp)
            roughness = np.sqrt(np.mean(roughness**2))

            # Calculate and print fit statistics
            if print_stat:
                self.calculate_and_print_fit_statistics(model=model)
                print(f'Roughness: {roughness:.4f}, RMS: {rms:.4f}, VR: {vr:.2f}%')

        return specs

    def _require_current_linear_solution(self, context):
        """Reject result extraction after an invalid constrained F_J solve."""
        if not getattr(self, '_last_linear_solve_valid', False) or self.mpost is None:
            raise RuntimeError(
                f"{context} could not obtain a feasible constrained linear "
                "solution; no previous mpost result was reused."
            )
    
    def calculate_and_print_fit_statistics(self, model='median'):
        """
        Calculate and print fit statistics for all datasets.
        
        Parameters:
        -----------
        model : str
            Model type to use ('median', 'mean', 'MAP', etc.)
        """
        super().calculate_and_print_fit_statistics(model=model)

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
        from ..viztools import optimize_3d_plot
        from matplotlib.ticker import FuncFormatter
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        import os
    
        # Extract faults data
        trifaults = self.multifaults.faults
    
        # Create output directory if it doesn't exist
        if output_gmt:
            output_path = pathlib.Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
    
        # Create a 3D plot
        with sci_plot_style(style=style):
            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111, projection='3d')
    
            # Current coordinates (red/blue) vs reference geometry (black)
            # Reference coords accessed via geometry_ref (frozen GeometryReference)
    
            # Plot each fault and output to GMT format if required
            for fault_data in trifaults:
                fault_name = fault_data.name
                if self.config.faults[fault_name]['geometry']['update'] and not self.config.faults[fault_name]['geometry'].get('follows'):
                    plot_items = [
                        (fault_data.top_coords,                'r', 'top'),
                        (fault_data.geometry_ref.top_coords,   'k', 'top_ref'),
                        (fault_data.geometry_ref.bottom_coords,'k', 'bottom_ref'),
                        (fault_data.bottom_coords,             'b', 'bottom'),
                    ]
                    for coords, color, part_name in plot_items:
                        if coords is None:
                            continue
                        x, y, z = coords[:, 0], coords[:, 1], -coords[:, 2]
                        ax.plot(x, y, z, color)

                        if output_gmt:
                            xy_filename = output_path / f"{fault_name}_{part_name}_xy.txt"
                            lonlat_filename = output_path / f"{fault_name}_{part_name}_lonlat.txt"
                            np.savetxt(xy_filename, np.column_stack((x, y, -z)), fmt='%.6f')
                            lon, lat = fault_data.xy2ll(x, y)
                            np.savetxt(lonlat_filename, np.column_stack((lon, lat, -z)), fmt='%.6f')
                else:
                    follows = self.config.faults[fault_name]['geometry'].get('follows')
                    if follows:
                        print(f"Fault {fault_name} shares geometry with master '{follows}', skipping plot.")
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

    def plot_kde_matrix(self, figsize=None, save=False, filename='kde_matrix.png', show=True, 
                        style='white', fill=True, scatter=False, scatter_size=15, 
                        plot_sigmas=False, plot_alpha=False, plot_faults=False, faults=None, 
                        plot_geometry=False, axis_labels=None,
                        hspace=None, wspace=None, xtick_rotation=None, ytick_rotation=None,
                        plot_posterior_sigmas=False, 
                        # Data Cleaning Options
                        remove_outliers=False, outlier_method='iqr', outlier_factor=1.5,
                        # KDE Optimization Options
                        adaptive_kde=False, kde_bw_method='scott',
                        # Main Mode Focusing Options
                        zoom_to_main_mode=False, percentile_range=(2.5, 97.5),
                        # Font size control - split into tick and label
                        tick_fontsize=None, label_fontsize=None,
                        # Tick marks control
                        show_minor_ticks=False, tick_direction='in',
                        major_tick_length=3, minor_tick_length=1.5,
                        tick_width=0.5,
                        ):
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
        - plot_sigmas (bool): Whether to include sampled sigma parameters in the plot
          (default is False). Fixed sigma groups have no KDE column and are skipped.
        - plot_alpha (bool): Whether to include sampled alpha parameters in the plot
          (default is False). Fixed alpha groups have no KDE column and are skipped.
        - plot_faults (bool): Whether to include fault parameters in the plot (default is False).
        - faults (list or str): Specific faults to include in the plot (default is None).
        - plot_geometry (bool): Whether to include geometry parameters in the plot (default is False).
        - axis_labels (list): List of axis labels for the plot (default is None).
          Labels may describe only sampled columns or every requested group; labels
          belonging to fixed sigma/alpha groups are ignored with those groups.
        - hspace (float): Horizontal space between subplots (default is None).
        - wspace (float): Vertical space between subplots (default is None).
        - xtick_rotation (float): Rotation angle for x-axis ticks (default is None).
        - ytick_rotation (float): Rotation angle for y-axis ticks (default is None).
        - plot_posterior_sigmas (bool): Whether to include posterior sigmas in the plot (default is False).
        
        Data Cleaning Options:
        - remove_outliers (bool): Whether to remove outliers before plotting (default is False).
        - outlier_method (str): Method to detect outliers ('iqr', 'zscore', 'percentile') (default is 'iqr').
        - outlier_factor (float): Factor for outlier detection (default is 1.5 for IQR, 3.0 for zscore).
        
        KDE Optimization Options:
        - adaptive_kde (bool): Whether to use adaptive KDE bandwidth (default is False).
        - kde_bw_method (str or float): Bandwidth method for KDE ('scott', 'silverman', or numeric value) (default is 'scott').
        
        Focus on Main Mode Options:
        - zoom_to_main_mode (bool): Whether to zoom to the main mode by removing extreme values (default is False).
        - percentile_range (tuple): Percentile range to keep for plotting (default is (2.5, 97.5)).
        
        Font Size Control:
        - tick_fontsize (float): Font size for tick labels (default is None).
        - label_fontsize (float): Font size for axis labels (default is None).
        
        Tick Marks Control:
        - show_minor_ticks (bool): Whether to show minor tick marks (default is False).
        - tick_direction (str): Direction of tick marks ('in', 'out', 'inout') (default is 'in').
        - major_tick_length (float): Length of major tick marks in points (default is 4).
        - minor_tick_length (float): Length of minor tick marks in points (default is 2.5).
        - tick_width (float): Width of tick marks in points (default is 1.0).
    
        Returns:
        - None
        """
        import seaborn as sns
        import pandas as pd
        import matplotlib.pyplot as plt
        import numpy as np
        from scipy import stats
    
        # Get the SMC chains
        trace = self.sampler.allsamples
        keys = []
        index = []
        # Keep a parallel mask for callers that supply labels for every
        # requested group, including fixed sigma/alpha groups.  Fixed groups
        # have no posterior coordinate and therefore no KDE column.
        axis_label_selection = []

        def group_update_mask(config_block):
            layout = config_block.get('group_layout')
            if layout is not None:
                updates = layout.get('update_by_group', [])
            else:
                updates = config_block.get('update', [])
            return np.asarray(updates, dtype=bool).reshape(-1)

        if plot_faults:
            if faults is None:
                for fault_name in self.faultnames:
                    fault_keys = [f"{fault_name}_{key}" for key in self.param_keys[fault_name]]
                    keys += fault_keys
                    index += self.param_index[fault_name]
                    axis_label_selection += [True] * len(fault_keys)
            elif type(faults) in (list, ):
                for fault_name in faults:
                    fault_keys = [f"{fault_name}_{key}" for key in self.param_keys[fault_name]]
                    keys += fault_keys
                    index += self.param_index[fault_name]
                    axis_label_selection += [True] * len(fault_keys)
            elif type(faults) in (str, ):
                assert faults in self.faultnames, f"Fault {faults} not found."
                fault_keys = list(self.param_keys[faults])
                keys += fault_keys
                index += self.param_index[faults]
                axis_label_selection += [True] * len(fault_keys)
        
        if plot_geometry:
            for fault_name in self.faultnames:
                if self.config.nonlinear_inversion and self.config.faults[fault_name]['geometry']['update']:
                    if self.config.faults[fault_name]['geometry'].get('follows'):
                        continue
                    geometry_keys = [f"{fault_name}_{i}" for i in range(self.config.faults[fault_name]['geometry']['sample_positions'][1] - self.config.faults[fault_name]['geometry']['sample_positions'][0])]
                    keys += geometry_keys
                    index += list(range(self.config.faults[fault_name]['geometry']['sample_positions'][0], self.config.faults[fault_name]['geometry']['sample_positions'][1]))
                    axis_label_selection += [True] * len(geometry_keys)
        
        if plot_sigmas:
            sigma_updates = group_update_mask(self.config.geodata['sigmas'])
            axis_label_selection += sigma_updates.tolist()
            if self.sigmas_position is not None:
                sigma_count = self.sigmas_position[1] - self.sigmas_position[0]
                if sigma_count != int(np.count_nonzero(sigma_updates)):
                    raise RuntimeError(
                        "Sigma KDE layout is inconsistent with sampled sigma groups."
                    )
                keys += [f"sigmas_{i}" for i in range(sigma_count)]
                index += list(range(self.sigmas_position[0], self.sigmas_position[1]))
                if plot_posterior_sigmas:
                    index_sigmas = list(range(self.sigmas_position[0], self.sigmas_position[1]))
                    aprior_sigmas = []
                    for idata in self.config.geodata['data']:
                        isigma = np.mean(np.sqrt(np.diag(idata.Cd)))
                        aprior_sigmas.append(isigma)
                    aprior_sigmas = np.array(aprior_sigmas)
            elif np.any(sigma_updates):
                raise RuntimeError(
                    "Sigma groups are marked for sampling but sigmas_position is not set."
                )
        
        if plot_alpha:
            alpha_updates = group_update_mask(self.config.alpha)
            axis_label_selection += alpha_updates.tolist()
            if self.alpha_position is not None:
                alpha_count = self.alpha_position[1] - self.alpha_position[0]
                if alpha_count != int(np.count_nonzero(alpha_updates)):
                    raise RuntimeError(
                        "Alpha KDE layout is inconsistent with sampled alpha groups."
                    )
                keys += [f'alpha_{i}' for i in range(alpha_count)]
                index += list(range(self.alpha_position[0], self.alpha_position[1]))
            elif np.any(alpha_updates):
                raise RuntimeError(
                    "Alpha groups are marked for sampling but alpha_position is not set."
                )

        if not keys:
            raise ValueError(
                "No sampled parameters were selected for the KDE matrix. "
                "Fixed sigma/alpha groups do not have posterior KDE columns."
            )

        resolved_axis_labels = None
        if axis_labels is not None:
            supplied_labels = list(axis_labels)
            if len(supplied_labels) == len(keys):
                resolved_axis_labels = supplied_labels
            elif len(supplied_labels) == len(axis_label_selection):
                resolved_axis_labels = [
                    label for label, selected
                    in zip(supplied_labels, axis_label_selection)
                    if selected
                ]
            else:
                raise ValueError(
                    "axis_labels must match either the sampled KDE columns "
                    "or all requested parameter groups (including fixed "
                    "sigma/alpha groups)."
                )
        
        # Convert the SMC chains to a DataFrame
        df = pd.DataFrame(trace[:, index], columns=keys)
        
        if plot_posterior_sigmas and self.sigmas_position is not None:
            df.iloc[:, index_sigmas] = 10**df.iloc[:, index_sigmas] * aprior_sigmas[None, :]
        # Remove columns with zero variance
        df = df.loc[:, df.var() != 0]
        if df.shape[1] == 0:
            raise ValueError(
                "The selected sampled parameters have zero variance; a KDE "
                "matrix cannot be constructed."
            )
        if resolved_axis_labels is not None:
            label_by_key = dict(zip(keys, resolved_axis_labels))
            resolved_axis_labels = [label_by_key[key] for key in df.columns]
        
        # Data cleaning: remove outliers
        if remove_outliers:
            original_len = len(df)
            df = self._remove_outliers_from_dataframe(df, method=outlier_method, factor=outlier_factor)
            removed_samples = original_len - len(df)
            print(f"Removed {removed_samples} outlier samples out of {original_len} total samples "
                  f"({removed_samples/original_len*100:.1f}%)")
        
        # Focus on main mode: remove extreme values
        if zoom_to_main_mode:
            original_len = len(df)
            for col in df.columns:
                lower_bound = df[col].quantile(percentile_range[0] / 100)
                upper_bound = df[col].quantile(percentile_range[1] / 100)
                df = df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]
            
            removed_samples = original_len - len(df)
            print(f"Zoomed to main mode: removed {removed_samples} samples "
                  f"({removed_samples/original_len*100:.1f}%) outside "
                  f"{percentile_range[0]}-{percentile_range[1]}% range")
        
        # Set the style
        sns.set_style(style)
    
        # Set PDF font type if saving as PDF
        if save and filename.endswith('.pdf'):
            pdf_fonttype = 42  # Use Type 42 (TrueType) for better compatibility
            plt.rcParams['pdf.fonttype'] = pdf_fonttype
        
        # Create a pair grid with separate y-axis for diagonal plots
        g = sns.PairGrid(df, diag_sharey=False)
    
        if figsize is not None:
            g.figure.set_size_inches(*figsize)
        
        # Remove the upper half of plots if scatter is not required
        if not scatter:
            for i, j in zip(*np.triu_indices_from(g.axes, 1)):
                g.axes[i, j].set_visible(False)
        
        # Define KDE plotting function with adaptive bandwidth
        def plot_kde_with_bandwidth(x, y=None, **kwargs):
            if y is None:  # Diagonal plot
                if adaptive_kde:
                    sns.kdeplot(x=x, fill=fill, bw_method=kde_bw_method, **kwargs)
                else:
                    sns.kdeplot(x=x, fill=fill, **kwargs)
            else:  # Off-diagonal plot
                if adaptive_kde:
                    sns.kdeplot(x=x, y=y, fill=fill, bw_method=kde_bw_method, **kwargs)
                else:
                    sns.kdeplot(x=x, y=y, fill=fill, **kwargs)
        
        # Plot KDE on the diagonal
        g.map_diag(plot_kde_with_bandwidth)
        
        # Plot KDE on the off-diagonal
        g.map_lower(plot_kde_with_bandwidth)
        
        # Plot scatter points on the upper half if required
        if scatter:
            g.map_upper(sns.scatterplot, s=scatter_size)
        
        # Configure tick marks for all subplots
        for i in range(len(g.axes)):
            for j in range(len(g.axes)):
                if g.axes[i, j].get_visible():
                    # Enable or disable minor ticks
                    if show_minor_ticks:
                        g.axes[i, j].minorticks_on()
                    else:
                        g.axes[i, j].minorticks_off()
                    
                    # Configure major tick marks
                    g.axes[i, j].tick_params(
                        axis='both',
                        which='major',
                        direction=tick_direction,
                        length=major_tick_length,
                        width=tick_width,
                        top=False,
                        right=False,
                        bottom=True,
                        left=True
                    )
                    
                    # Configure minor tick marks (only if enabled)
                    if show_minor_ticks:
                        g.axes[i, j].tick_params(
                            axis='both',
                            which='minor',
                            direction=tick_direction,
                            length=minor_tick_length,
                            width=tick_width,
                            top=False,
                            right=False,
                            bottom=True,
                            left=True
                        )
                    
                    # Ensure tick locators are set
                    g.axes[i, j].xaxis.set_major_locator(AutoLocator())
                    g.axes[i, j].yaxis.set_major_locator(AutoLocator())
        
        # Set tick rotation and font size if provided
        default_tick_fontsize = tick_fontsize if tick_fontsize is not None else 10
        if xtick_rotation is not None:
            for ax in g.axes[-1, :]:
                for label in ax.get_xticklabels():
                    label.set_rotation(xtick_rotation)
                    label.set_ha('right')
                    label.set_fontsize(default_tick_fontsize)

        if ytick_rotation is not None:
            for ax in g.axes[:, 0]:
                for label in ax.get_yticklabels():
                    label.set_rotation(ytick_rotation)
                    label.set_ha('right')
                    label.set_fontsize(default_tick_fontsize)
        
        # Set font sizes for all tick labels if tick_fontsize is provided and rotation is not specified
        if xtick_rotation is None:
            for ax in g.axes[-1, :]:
                ax.tick_params(axis='x', labelsize=default_tick_fontsize)
        if ytick_rotation is None:
            for ax in g.axes[:, 0]:
                ax.tick_params(axis='y', labelsize=default_tick_fontsize)

        # Set axis labels if provided
        default_label_fontsize = label_fontsize if label_fontsize is not None else 12
        if resolved_axis_labels is not None:
            for i, label in enumerate(resolved_axis_labels):
                g.axes[-1, i].set_xlabel(label, fontsize=default_label_fontsize)
                g.axes[i, 0].set_ylabel(label, fontsize=default_label_fontsize)
        else:
            # Set fontsize for existing axis labels
            for i in range(len(g.axes)):
                if g.axes[-1, i].get_xlabel():
                    g.axes[-1, i].set_xlabel(g.axes[-1, i].get_xlabel(), fontsize=default_label_fontsize)
                if g.axes[i, 0].get_ylabel():
                    g.axes[i, 0].set_ylabel(g.axes[i, 0].get_ylabel(), fontsize=default_label_fontsize)

        plt.tight_layout()
        if wspace is not None or hspace is not None:
            plt.subplots_adjust(wspace=wspace, hspace=hspace)
        # Save the figure if required
        if save:
            plt.savefig(filename, dpi=600)
        
        # Show the figure if required
        if show:
            plt.show()

    def _remove_outliers_from_dataframe(self, df, method='iqr', factor=1.5):
        """
        Remove outliers from dataframe using specified method.
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        method : str
            Method for outlier detection ('iqr', 'zscore', 'percentile')
        factor : float
            Factor for outlier detection
            
        Returns:
        --------
        pandas.DataFrame
            Dataframe with outliers removed
        """
        if method == 'iqr':
            # Interquartile Range method
            mask = np.ones(len(df), dtype=bool)
            for col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - factor * IQR
                upper_bound = Q3 + factor * IQR
                mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
            
        elif method == 'zscore':
            # Z-score method
            mask = np.ones(len(df), dtype=bool)
            for col in df.columns:
                z_scores = np.abs(stats.zscore(df[col]))
                mask &= (z_scores < factor)
                
        elif method == 'percentile':
            # Percentile method
            mask = np.ones(len(df), dtype=bool)
            lower_percentile = factor
            upper_percentile = 100 - factor
            for col in df.columns:
                lower_bound = df[col].quantile(lower_percentile / 100)
                upper_bound = df[col].quantile(upper_percentile / 100)
                mask &= (df[col] >= lower_bound) & (df[col] <= upper_bound)
                
        else:
            raise ValueError(f"Unknown outlier detection method: {method}")
        
        return df[mask]

    def extract_and_plot_bayesian_results(self, rank=0, filename='samples_100_50.h5', 
                                          plot_faults=True, plot_std=False, plot_sigmas=True, plot_data=True,
                                          antisymmetric=True, res_use_data_norm=True, cmap='RdBu_r', azimuth=None, elevation=None,
                                          slip_cmap='cmc.roma_r', depth_range=None, z_ticks=None, 
                                          axis_shape=(1.0, 1.0, 0.6), zratio=None, best_model='median', 
                                          gps_title=True, sar_title=True, sar_cbaxis=[0.1, 0.15, 0.35, 0.04], # [0.15, 0.25, 0.25, 0.02],
                                          gps_figsize=None, sar_figsize='double', gps_scale=0.05, gps_legendscale=0.2,
                                          file_type='png', fault_cbaxis=[0.15, 0.22, 0.15, 0.02], fault_style=['notebook'],
                                          remove_direction_labels=False, cbticks=None, cblinewidth=None, cbfontsize=None, cb_label_side='opposite',
                                          map_cbaxis=None, data_poly="config", print_fit_statistics=True, print_fault_statistics=True,
                                          pdf_fonttype=None, gps_fontsize=None, sar_fontsize=None, gps_xticks=None, gps_yticks=None,
                                          sar_xticks=None, sar_yticks=None,
                                          gps_kwargs=None, sar_kwargs=None,
                                          fault_outdir='output', data_outdir='Modeling', show=True,
                                          model=None):
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
        model: canonical alias for ``best_model``. New scripts should use this
            spelling; ``best_model`` remains accepted for existing scripts.
        gps_title: whether to show title for GPS data plots (default is True)
        sar_title: whether to show title for SAR data plots (default is True)
        sar_cbaxis: colorbar axis position for SAR data plots (default is [0.1, 0.15, 0.35, 0.04])
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
        data_poly: "config" (default) follows each dataset's parsed
            geodata.polys value; "include" includes solved corrections;
            None explicitly plots source/slip-only results.
        print_fit_statistics: whether to print fit statistics (default is True)
        print_fault_statistics: whether to print fault statistics (default is True)
        pdf_fonttype: PDF font type (default is None)
        gps_fontsize: font size for GPS plots (default is None)
        sar_fontsize: font size for SAR plots (default is None)
        gps_xticks: custom x-ticks for GPS plots (default is None)
        gps_yticks: custom y-ticks for GPS plots (default is None)
        sar_xticks: custom x-ticks for SAR plots (default is None)
        sar_yticks: custom y-ticks for SAR plots (default is None)
        gps_kwargs: additional keyword arguments for GPS plotting (default is empty dict)
        sar_kwargs: additional keyword arguments for SAR plotting (default is empty dict)
        fault_outdir: directory for posterior and fault-field figures
            (default is 'output')
        data_outdir: directory for GPS/InSAR/leveling/cross-fault figures
            (default is 'Modeling')
        show: whether underlying plotting methods call their interactive show
            path (default is True)
        """
        if model is not None:
            best_model = model
        if (
            rank == 0
            and isinstance(best_model, str)
            and best_model.lower() == 'std'
        ):
            raise ValueError(
                "best_model='std' is descriptive, not predictive. Use "
                "plot_std=True and select mean, median, MAP, max_prob, or a "
                "custom vector as the active result model."
            )
        if rank == 0:
            import cmcrameri
            from ..getcpt import get_cpt 

            file_type = normalize_image_format(file_type)
            fault_output_path = pathlib.Path(fault_outdir)
            fault_output_path.mkdir(parents=True, exist_ok=True)
    
            if slip_cmap is not None and slip_cmap.endswith('.cpt'):
                # 'precip3_16lev_change.cpt'
                cmap_slip = get_cpt.get_cmap(slip_cmap, method='list', N=15)
            else:
                cmap_slip = slip_cmap
            if slip_cmap is None:
                cmap_slip = get_cpt.get_cmap('precip3_16lev_change.cpt', method='list', N=15)
            self.load_from_h5(filename)

            fault_plot_kwargs = dict(
                drawCoastlines=False,
                cblabel='Slip (m)',
                style=fault_style,
                cbaxis=fault_cbaxis,
                xtickpad=5,
                ytickpad=5,
                ztickpad=5,
                xlabelpad=15,
                ylabelpad=15,
                zlabelpad=15,
                shape=axis_shape,
                zratio=zratio,
                elevation=elevation,
                azimuth=azimuth,
                depth=depth_range,
                zticks=z_ticks,
                fault_expand=0.0,
                plot_faultEdges=False,
                remove_direction_labels=remove_direction_labels,
                cbticks=cbticks,
                cbfontsize=cbfontsize,
                cblinewidth=cblinewidth,
                cb_label_side=cb_label_side,
                map_cbaxis=map_cbaxis,
            )
    
            if plot_std:
                self.returnModel(model='std', print_fit_statistics=False)  # std mean
                self.plot_fault_fields(
                    fields=('total',),
                    outdir=fault_output_path,
                    file_type=file_type,
                    slip_cmap=cmap_slip,
                    show=show,
                    suffix='std',
                    **fault_plot_kwargs,
                )
            
            # Print hyperparameters summary table
            self.returnModel(
                model=best_model,
                print_fit_statistics=print_fit_statistics,
            )  # best model
            self._print_hyperparameters_summary()
            if print_fault_statistics:
                self._print_fault_statistics()

            if plot_sigmas:
                self.plot_kde_matrix(plot_sigmas=True, plot_alpha=True, fill=True, save=True,
                                        scatter=False, filename=str(
                                            fault_output_path / f'kde_matrix_sigmas.{file_type}'
                                        ), show=show)

            if plot_faults:
                self.plot_fault_fields(
                    fields=('total',),
                    outdir=fault_output_path,
                    file_type=file_type,
                    slip_cmap=cmap_slip,
                    show=show,
                    suffix=best_model if isinstance(best_model, str) else 'custom',
                    **fault_plot_kwargs,
                )

            if file_type == 'pdf':
                pdf_fonttype = pdf_fonttype if pdf_fonttype is not None else 42  # Use Type 42 (TrueType) for better compatibility
            else:
                pdf_fonttype = None

            resolved_gps_kwargs = {
                'color': ['k', 'r'],
                'xticks': gps_xticks,
                'yticks': gps_yticks,
            }
            resolved_gps_kwargs.update(dict(gps_kwargs or {}))

            # Reuse the same prediction specifications as the legacy flow.
            # The product preserves GPS/InSAR/opticorr/leveling/cross-fault
            # buildsynth arguments and only centralizes plotting/output.
            self.plot_data_fits(
                outdir=data_outdir,
                file_type=file_type,
                plot_data=plot_data,
                data_poly=data_poly,
                antisymmetric=antisymmetric,
                res_use_data_norm=res_use_data_norm,
                cmap=cmap,
                gps_title=gps_title,
                sar_title=sar_title,
                gps_figsize=gps_figsize,
                sar_figsize=sar_figsize,
                gps_scale=gps_scale,
                gps_legendscale=gps_legendscale,
                sar_cbaxis=sar_cbaxis,
                remove_direction_labels=remove_direction_labels,
                gps_kwargs=resolved_gps_kwargs,
                sar_kwargs=sar_kwargs,
                gps_fault_color='b',
                sar_fault_color='k',
                fault_linewidth=None,
                pdf_fonttype=pdf_fonttype,
                gps_fontsize=gps_fontsize,
                sar_fontsize=sar_fontsize,
                show=show,
            )
    
    def _collect_scale_parameter_rows(self):
        """Return physical sigma/alpha rows for the currently active model.

        This reporting adapter intentionally reuses the canonical group layout
        and the same physical-scale resolvers as the likelihood.  It never
        changes ``self.model``, samples, geometry, Green functions, Laplacian,
        or the conditional linear solution.
        """

        posterior = np.asarray(self.sampler.allsamples, dtype=float)
        rows = []

        sigma_layout = self.config.sigmas.get('group_layout', {})
        sigma_by_dataset = getattr(self, 'current_data_sigmas', {})
        if not sigma_by_dataset:
            raise RuntimeError(
                "No active physical sigma context. Activate a predictive "
                "mean/median/MAP/custom model with returnModel() before "
                "printing hyperparameter summaries; the posterior standard "
                "deviation vector is descriptive and is not a model."
            )
        sigma_scales = {
            group: float(sigma_by_dataset[members[0]])
            for group, members in sigma_layout.get('members_by_group', {}).items()
            if members
        }
        sigma_samples = None
        sigma_offset = None
        if self.sigmas_position is not None:
            sigma_offset = self.sigmas_position[0]
            sigma_samples = posterior[:, self.sigmas_position[0]:self.sigmas_position[1]]
        rows.extend(
            build_scale_parameter_rows(
                kind='sigma',
                layout=sigma_layout,
                active_scales_by_group=sigma_scales,
                update_state='sampled',
                log_scaled=bool(self.config.sigmas.get('log_scaled', False)),
                posterior_samples=sigma_samples,
                sample_index_offset=sigma_offset,
            )
        )

        if self.config.alpha_enabled:
            alpha_layout = self.config.alpha.get('group_layout', {})
            alpha_scales = getattr(self, 'current_alpha_group_values', {})
            if not alpha_scales:
                raise RuntimeError(
                    "No active physical alpha context. Activate a predictive "
                    "mean/median/MAP/custom model with returnModel() before "
                    "printing hyperparameter summaries."
                )
            alpha_samples = None
            alpha_offset = None
            if self.alpha_position is not None:
                alpha_offset = self.alpha_position[0]
                alpha_samples = posterior[:, self.alpha_position[0]:self.alpha_position[1]]
            rows.extend(
                build_scale_parameter_rows(
                    kind='alpha',
                    layout=alpha_layout,
                    active_scales_by_group=alpha_scales,
                    update_state='sampled',
                    log_scaled=bool(self.config.alpha.get('log_scaled', False)),
                    posterior_samples=alpha_samples,
                    sample_index_offset=alpha_offset,
                )
            )
        return rows

    def _print_hyperparameters_summary(self):
        """Print geometry and scale parameters without mixing coordinate spaces."""

        posterior = np.asarray(self.sampler.allsamples, dtype=float)
        resolved_updates = getattr(self.config, '_resolved_geometry_updates', ())
        geometry_rows = build_geometry_parameter_rows(
            resolved_updates,
            active_vector=self.model,
            posterior_samples=posterior,
        )
        scale_rows = self._collect_scale_parameter_rows()

        print("\n" + "=" * 80)
        print("Bayesian Hyperparameter Summary")
        print("=" * 80)
        if geometry_rows:
            print(format_geometry_parameter_report(geometry_rows))
            print()
        print(
            format_scale_parameter_report(
                scale_rows,
                title="Bayesian scale parameters",
                show_index=True,
                show_posterior_uncertainty=True,
                tablefmt='simple',
            )
        )
        print(
            "Scale (s) and Row mult. (1/s) are physical active-model values; "
            "Sampling identifies the stored Bayesian coordinate."
        )
        print("=" * 80)

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
    
    @property
    def lsq_parameters(self):
        """Calculate the total number of least-squares samples required for the inversion."""
        rake_fixed = self.config.slip_sampling_mode == 'rake_fixed'
        mcmc_samples = self._calculate_samples(rake_fixed=rake_fixed)
        return mcmc_samples - self.linear_sample_start_position
    
    # alias of lsq_parameters using lsq_samples
    @property
    def lsq_samples(self):
        return self.lsq_parameters
    
    # alias of lsq_parameters using linear_parameters
    @property
    def linear_parameters(self):
        return self.lsq_parameters

    def _calculate_samples(self, rake_fixed):
        """Calculate the total number of samples required for the inversion based on the configuration whether rake is fixed or not."""
        total_samples = 0
        for fault in self.multifaults.faults:
            if hasattr(self.multifaults, 'adapters') and fault.name in self.multifaults.adapters:
                adapter = self.multifaults.adapters[fault.name]
                num_slip_samples = adapter.get_n_source_params()
                # rake_fixed halving only applies to Fault sources
                if rake_fixed and adapter.source_type == 'Fault':
                    num_slip_samples //= 2
            else:
                npatches = len(fault.patch) # Number of patches
                num_slip_samples = len(FaultAdapter._canonicalize_slipdir(fault.slipdir)) * npatches
                if rake_fixed:
                    num_slip_samples //= 2
            # print(fault.poly)
            # print(fault.numberofpolys) fault.numberofpolys is a dict defined in fault.assebleGFs()
            num_poly_samples = np.sum([fault.numberofpolys[ikey] for ikey in fault.numberofpolys], dtype=int)
            # num_poly_samples = np.sum([npoly for npoly in fault.poly.values() if npoly is not None], dtype=int)
            total_samples += num_slip_samples + num_poly_samples

            if self.config.nonlinear_inversion and self.config.faults[fault.name]['geometry']['update']:
                if not self.config.faults[fault.name]['geometry'].get('follows'):
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
        if self._sigma_update_flag:
            print(f"Sigmas position: {self.sigmas_position}")
        if self._alpha_update_flag:
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

        if self._sigma_update_flag:
            sigmas_start, sigmas_end = self.sigmas_position
            print(f"Sigmas position: [{sigmas_start}, {sigmas_end}]")
        if self._alpha_update_flag:
            alpha_start, alpha_end = self.alpha_position
            print(f"Alpha position: [{alpha_start}, {alpha_end}]")

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
        # n_datasets = len(self.multifaults.faults[0].d)  # Number of data points
        n_sigmas_to_update = self.config.sigmas['updatable_params']  # Number of sigmas to update
        n = n_sigmas_to_update
        if self._sigma_update_flag:
            self.sigmas_position = (self.total_geometry_parameters, self.total_geometry_parameters + n)
        else:
            self.sigmas_position = None
            n = 0
    
        # Calculate the positions for alpha
        if self._alpha_update_flag:
            n_alpha = self.config.alpha['updatable_params']  # Number of alpha parameters to update
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
        start_position = self.total_geometry_parameters
        if self.sigmas_position is not None:
            start_position += self.sigmas_position[1] - self.sigmas_position[0]
        if self.alpha_position is not None:
            start_position += self.alpha_position[1] - self.alpha_position[0]
        for fault in self.multifaults.faults:
            # Use adapter if available for type-safe parameter counting
            if hasattr(self.multifaults, 'adapters') and fault.name in self.multifaults.adapters:
                adapter = self.multifaults.adapters[fault.name]
                num_slip_samples = adapter.get_n_source_params()
            else:
                npatches = len(fault.patch)
                num_slip_samples = len(FaultAdapter._canonicalize_slipdir(fault.slipdir)) * npatches
            num_poly_samples = np.sum([fault.numberofpolys[ikey] for ikey in fault.numberofpolys], dtype=int)
            self.slip_positions[fault.name] = (start_position, start_position + num_slip_samples)
            self.poly_positions[fault.name] = (start_position + num_slip_samples, start_position + num_slip_samples + num_poly_samples)
            start_position += num_slip_samples + num_poly_samples

    @property
    def full_slip_positions(self):
        """Source slip slices in the full physical model vector."""
        return self.slip_positions

    @property
    def full_poly_positions(self):
        """Source polynomial slices in the full physical model vector."""
        return self.poly_positions

    def calculate_linear_sample_start_position(self):
        start_position = self.total_geometry_parameters
        if self._sigma_update_flag:
            start_position += self.sigmas_position[1] - self.sigmas_position[0]
        if self._alpha_update_flag:
            start_position += self.alpha_position[1] - self.alpha_position[0]
        self.linear_sample_start_position = start_position
        return start_position

    def calculate_sample_slip_only_positions(self):
        slip_only_positions = []
        smoothing_slip_only_positions = []
        for fault_name in self.faultnames:
            slip_start, slip_end = self.slip_positions[fault_name]
            positions = list(range(slip_start, slip_end))
            slip_only_positions.extend(positions)

            # Check if this source supports smoothing (has GL)
            fault = next((f for f in self.multifaults.faults if f.name == fault_name), None)
            if fault is not None and hasattr(fault, 'GL') and fault.GL is not None:
                smoothing_slip_only_positions.extend(positions)

        slip_only_positions = np.array(slip_only_positions)
        self.sample_slip_only_positions = slip_only_positions
        self.smoothing_slip_only_positions = np.array(smoothing_slip_only_positions)
        return slip_only_positions

    def compute_slip(self, samples, fault):
        """Compute scalar slip magnitude from samples for a Fault source.
        
        This method is Fault-specific: magnitude_rake / rake_fixed modes and the
        2-component vector-norm are all Fault slip decomposition concepts.
        For non-Fault sources, raw absolute parameter values are returned.
        """
        slip_start, slip_end = self.slip_positions[fault.name]

        # Non-Fault sources: return raw absolute values (no slip decomposition)
        adapter = None
        if hasattr(self.multifaults, 'adapters') and fault.name in self.multifaults.adapters:
            adapter = self.multifaults.adapters[fault.name]
            if adapter.source_type != 'Fault':
                return np.abs(samples[slip_start:slip_end].copy())

        if self.config.slip_sampling_mode == 'magnitude_rake':
            slip_magnitude_and_rake = samples[slip_start:slip_end].copy()
            half = len(slip_magnitude_and_rake) // 2
            slip = slip_magnitude_and_rake[:half]
            return slip
        elif self.config.slip_sampling_mode == 'rake_fixed':
            compact_start, compact_end = self.constraint_manager.sample_slip_positions[
                fault.name
            ]
            slip = samples[compact_start:compact_end].copy()
            return slip
        else:
            slip = samples[slip_start:slip_end].copy()  # Create a copy of slip to avoid modifying samples
            if adapter is not None:
                n_comp = len(adapter.get_param_names())
            else:
                n_comp = len(FaultAdapter._canonicalize_slipdir(fault.slipdir))
            if n_comp == 2:  # If both components of slip (dip or strikeslip)
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
            return self._expand_rake_fixed_samples(samples)
        else:
            return samples

    def _expand_rake_fixed_samples(self, samples):
        """Expand compact fixed-rake magnitudes into the full linear layout."""
        samples = np.asarray(samples)
        full_ends = [self.linear_sample_start_position]
        full_ends.extend(end for _, end in self.full_slip_positions.values())
        full_ends.extend(end for _, end in self.full_poly_positions.values())
        expanded = np.zeros(max(full_ends), dtype=samples.dtype)
        linear_start = self.linear_sample_start_position
        expanded[:linear_start] = samples[:linear_start]

        for source in self.multifaults.faults:
            name = source.name
            adapter = self.multifaults.adapters[name]
            compact_slip = slice(*self.constraint_manager.sample_slip_positions[name])
            full_slip = slice(*self.full_slip_positions[name])
            compact_poly = slice(*self.constraint_manager.sample_poly_positions[name])
            full_poly = slice(*self.full_poly_positions[name])

            if adapter.source_type != 'Fault':
                expanded[full_slip] = samples[compact_slip]
            else:
                magnitude = samples[compact_slip]
                rake = np.full_like(magnitude, self.config.rake_angle)
                expanded[full_slip] = self.transfer_magnitude_rake_to_ss_ds(
                    magnitude, rake
                )
            expanded[full_poly] = samples[compact_poly]
        return expanded
    
    def compute_magnitude_log_prior(self, samples, decay_rate=0.1):
        moment_magnitude_threshold = self.moment_magnitude_threshold
        magnitude_tolerance = self.magnitude_tolerance

        # ``self.patch_areas`` is a baseline/compatibility snapshot used by
        # magnitude-aware initialization. A posterior candidate must consume
        # areas belonging to its current geometry.
        current_patch_areas = self.multifaults.get_fault_areas()

        # Only Fault sources contribute to moment magnitude (Pressure/Sbarbot have no patch areas)
        fault_sources = [fault for fault in self.multifaults.faults
                         if fault.name in current_patch_areas]

        total_moment = 0.0
        for fault in fault_sources:
            areas = np.asarray(current_patch_areas[fault.name], dtype=float)
            slip = np.asarray(self.compute_slip(samples, fault), dtype=float)
            if areas.shape != slip.shape:
                raise ValueError(
                    f"Fault '{fault.name}' has {areas.size} patch areas but "
                    f"{slip.size} slip magnitudes."
                )
            total_moment += self.shear_modulus * np.sum(areas * slip)

        total_moment *= 1e6  # km^2 to m^2
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
        lb (dict): Lower bounds for each fault. If None, use lb in self.constraint_manager.bounds.
        ub (dict): Upper bounds for each fault. If None, use ub in self.constraint_manager.bounds.

        Returns:
        dict: A dictionary where keys are fault names and values are slip samples.
        float: The moment magnitude of the generated sample.
        """

        # If faults is not provided, use all faults
        if faults is None:
            faults = list(self.patch_areas)
        unknown = [name for name in faults if name not in self.patch_areas]
        if unknown:
            raise ValueError(
                f"Magnitude-based initialization only supports Fault sources; "
                f"unsupported sources: {unknown}"
            )

        # Resolve physical non-negative magnitude ranges from the active bounds.
        if lb is None or ub is None:
            lb = {}
            ub = {}
            full_lb, full_ub = self.constraint_manager.get_bounds_for_fullsmc()
            if self.config.slip_sampling_mode == 'ss_ds':
                for name in faults:
                    fault = self.multifaults.faults_dict[name]
                    adapter = self.multifaults.adapters[name]
                    start, _ = self.constraint_manager.sample_slip_positions[name]
                    slices = self.constraint_manager._source_component_slices(
                        fault, start, adapter=adapter
                    )
                    npatch = len(fault.patch)
                    max_ss = np.zeros(npatch)
                    max_ds = np.zeros(npatch)
                    if 'strikeslip' in slices:
                        slc = slices['strikeslip']
                        max_ss = np.maximum(
                            np.abs(full_lb[slc]), np.abs(full_ub[slc])
                        )
                    if 'dipslip' in slices:
                        slc = slices['dipslip']
                        max_ds = np.maximum(
                            np.abs(full_lb[slc]), np.abs(full_ub[slc])
                        )
                    lb[name] = np.zeros(npatch)
                    ub[name] = np.hypot(max_ss, max_ds)
            else:
                for name in faults:
                    start, end = self.constraint_manager.sample_slip_positions[name]
                    if self.config.slip_sampling_mode == 'magnitude_rake':
                        end = start + (end - start) // 2
                    lb[name] = np.maximum(full_lb[start:end], 0.0)
                    ub[name] = full_ub[start:end].copy()
                    if np.any(ub[name] < lb[name]):
                        raise ValueError(
                            f"Non-positive slip_magnitude bounds for '{name}'"
                        )

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

        if faults is None:
            faults = list(self.patch_areas)
        # Initialize only the requested Fault sources.
        samples = {name: [] for name in faults}
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
                    compact_start, compact_end = (
                        self.constraint_manager.sample_slip_positions[name]
                    )
                    sampzero[:, compact_start:compact_end] = samples[name]
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

    def _require_bayesian_sampling_mode(self, expected, caller):
        """Keep target construction aligned with the configured mode."""
        actual = self.config.bayesian_sampling_mode
        if actual != expected:
            raise RuntimeError(
                f"{caller} requires bayesian_sampling_mode='{expected}', "
                f"got '{actual}'. Use walk() for mode-aware dispatch."
            )

    def _validate_sampling_ready(self):
        """Run configuration-owned nonlinear preflight once per target build."""
        if not self.nonlinear_inversion:
            return
        validator = getattr(self.config, 'validate_sampling_ready', None)
        if validator is not None:
            validator()

    def _freeze_fullsmc_bounds(self):
        """Freeze one effective bounds snapshot for target and proposal use."""
        self.constraint_manager._require_activation_flags_reconciled(
            "FULLSMC target construction"
        )
        lb, ub = self.constraint_manager.get_bounds_for_fullsmc()
        snapshot = {
            'lb': np.asarray(lb, dtype=float).copy(),
            'ub': np.asarray(ub, dtype=float).copy(),
            'state_revision': self.constraint_manager.state_revision,
        }
        self._fullsmc_bounds_snapshot = snapshot
        return snapshot

    def _build_fullsmc_prior_guard(self):
        """Return frozen bounds and a revision guard for a FULLSMC target."""
        snapshot = self._freeze_fullsmc_bounds()
        lb = snapshot['lb']
        ub = snapshot['ub']
        constraint_revision = snapshot['state_revision']

        def ensure_current_bounds():
            if self.constraint_manager.state_revision != constraint_revision:
                raise RuntimeError(
                    "FULLSMC bounds changed after target construction. "
                    "Rebuild the target or call walk_smc() again before "
                    "sampling."
                )

        return lb, ub, ensure_current_bounds

    def make_target_for_parallel(self, log_enabled=False):
        self._require_bayesian_sampling_mode('FULLSMC', 'make_target_for_parallel')
        self._validate_sampling_ready()
        lb, ub, ensure_current_bounds = self._build_fullsmc_prior_guard()
        if self.nonlinear_inversion:
            def target(samples):
                ensure_current_bounds()
                # Compute log prior
                log_prior = compute_log_prior(samples, lb, ub)
                if log_prior == -np.inf:
                    return -np.inf

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        # self._update_fault(fault_name, fault_config, samples)
                        if not self._try_update_fault_geometry_and_mesh(
                                fault_name, fault_config, samples,
                                log_enabled=log_enabled):
                            return -np.inf
                        self._update_fault_GFs_and_Laplacian(
                            fault_name, fault_config,
                            update_laplacian=self.config.alpha_enabled,
                            log_enabled=log_enabled,
                        )

                new_samples = self.transfer_samples(samples)
                return log_prior + self._compute_likelihoods(new_samples)
        else:
            def target(samples):
                ensure_current_bounds()
                # Compute log prior
                # start_time = time.time()
                log_prior = compute_log_prior(samples, lb, ub)
                # end_time = time.time()
                # print(f"Execution time for computing log prior: {end_time - start_time} seconds")

                if log_prior == -np.inf:
                    return -np.inf
                
                new_samples = self.transfer_samples(samples)
                return log_prior + self._compute_likelihoods(new_samples, GL_combined=self.GL_combined)
        self.target = target
        return target

    def make_magnitude_target_for_parallel(self, decay_rate=0.1, log_enabled=False):
        self._require_bayesian_sampling_mode(
            'FULLSMC', 'make_magnitude_target_for_parallel'
        )
        self._validate_sampling_ready()
        lb, ub, ensure_current_bounds = self._build_fullsmc_prior_guard()
        if self.nonlinear_inversion:
            def target(samples):
                ensure_current_bounds()
                # Compute log prior
                log_prior = compute_log_prior(samples, lb, ub)
                if log_prior == -np.inf:
                    return -np.inf

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        if not self._try_update_fault_geometry_and_mesh(
                                fault_name, fault_config, samples,
                                update_areas=True, log_enabled=log_enabled):
                            return -np.inf

                # Compute log magnitude prior
                # start_time_magnitude_log_prior = time.time()
                magnitude_log_prior = self.compute_magnitude_log_prior(samples, decay_rate=decay_rate)
                # end_time_magnitude_log_prior = time.time()
                # print(f"Execution time for computing magnitude log prior: {end_time_magnitude_log_prior - start_time_magnitude_log_prior} seconds")

                # if magnitude_log_prior != 0.0:
                #     return magnitude_log_prior

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        self._update_fault_GFs_and_Laplacian(
                            fault_name, fault_config,
                            update_laplacian=self.config.alpha_enabled,
                            log_enabled=log_enabled,
                        )

                new_samples = self.transfer_samples(samples)
                return log_prior + magnitude_log_prior + self._compute_likelihoods(new_samples)
        else:
            def target(samples):
                ensure_current_bounds()
                # Compute log prior
                log_prior = compute_log_prior(samples, lb, ub)

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

    def make_smc_fj_target_for_parallel(self, log_enabled=False, x0=None, opts=None,
                smooth_prior_weight=1.0,
                magnitude_log_prior=False, decay_rate=0.1):
        self._require_bayesian_sampling_mode(
            'SMC_FJ', 'make_smc_fj_target_for_parallel'
        )
        self._validate_sampling_ready()

        self.constraint_manager._require_activation_flags_reconciled(
            "SMC_FJ target construction"
        )

        # The manager is the single source of truth for the linear problem.
        # This prevents call-site arrays from bypassing named groups, ownership
        # checks, active-space column validation, or config/runtime precedence.
        A, b = self.constraint_manager.get_combined_inequality_constraints()
        Aeq, beq = self.constraint_manager.get_combined_equality_constraints()
        n_linear = int(self.constraint_manager.get_linear_parameter_layout()['width'])

        has_manager_bounds = self.constraint_manager.has_active_linear_bounds()
        lb = ub = None
        if self.config.use_bounds_constraints or has_manager_bounds:
            lb, ub = self.constraint_manager.get_bounds_for_linear_parameters()
            lb, ub = self.constraint_manager._normalise_active_bounds_pair(
                lb,
                ub,
                expected_length=n_linear,
                label='SMC_FJ linear bounds',
            )

        # Get hyperparameter bounds
        hyper_lb, hyper_ub = self.constraint_manager.get_bounds_for_hyperparameters()
        hyper_lb, hyper_ub = self.constraint_manager._normalise_active_bounds_pair(
            hyper_lb,
            hyper_ub,
            expected_length=int(np.asarray(hyper_lb).size),
            label='SMC_FJ hyperparameter bounds',
        )
        constraint_revision = self.constraint_manager.state_revision

        def ensure_current_constraints():
            if self.constraint_manager.state_revision != constraint_revision:
                raise RuntimeError(
                    "SMC_FJ constraints changed after target construction. "
                    "Rebuild the target or call walk_smc_fj() again before sampling."
                )

        if self.nonlinear_inversion:
            def target(samples):
                ensure_current_constraints()
                # Compute log prior
                log_prior = compute_log_prior(samples, hyper_lb, hyper_ub)
                if log_prior == -np.inf:
                    return -np.inf

                for fault_name, fault_config in self.config.faults.items():
                    if fault_name in self.faultnames and fault_config['geometry']['update']:
                        if not self._try_update_fault_geometry_and_mesh(
                                fault_name, fault_config, samples,
                                log_enabled=log_enabled):
                            return -np.inf
                        self._update_fault_GFs_and_Laplacian(
                            fault_name, fault_config,
                            update_laplacian=self.config.alpha_enabled,
                            log_enabled=log_enabled,
                        )

                return log_prior + self._compute_likelihoods_smc_fj(samples, A=A, b=b, Aeq=Aeq, beq=beq, \
                                                                 lb=lb, ub=ub, x0=x0, opts=opts, smooth_prior_weight=smooth_prior_weight,
                                                                 magnitude_log_prior=magnitude_log_prior, decay_rate=decay_rate)
        else:
            # Geometry, covariance, observations, and Laplacian remain fixed
            # for this target.  Prepare their exact Gram/cross products once;
            # candidates only change sigma/alpha scalar weights.
            quadratic_workspace = self._build_smc_fj_quadratic_workspace(
                GL_combined=self.GL_combined
            )

            def target(samples):
                ensure_current_constraints()
                # Compute log prior
                log_prior = compute_log_prior(samples, hyper_lb, hyper_ub)

                if log_prior == -np.inf:
                    return -np.inf
                
                return log_prior + self._compute_likelihoods_smc_fj(samples, GL_combined=self.GL_combined, A=A, b=b, Aeq=Aeq, beq=beq, \
                                                                 lb=lb, ub=ub, x0=x0, opts=opts, smooth_prior_weight=smooth_prior_weight,
                                                                 magnitude_log_prior=magnitude_log_prior, decay_rate=decay_rate,
                                                                 quadratic_workspace=quadratic_workspace)
        self.target = target
        return target

    def _try_update_fault_geometry_and_mesh(self, *args, **kwargs):
        """Update one Bayesian candidate, rejecting only invalid geometry.

        Returns False for InvalidFaultGeometryError so the target can assign
        -inf. Other exceptions remain visible because configuration, indexing,
        and numerical-programming errors must not be hidden as an ordinary
        rejected sample.
        """
        try:
            self._update_fault_geometry_and_mesh(*args, **kwargs)
        except InvalidFaultGeometryError:
            return False
        return True

    def _update_fault_geometry_and_mesh(self, fault_name, fault_config, samples, update_areas=False, log_enabled=False):
        # Followers share geometry via SharedFaultInfo; nothing to update.
        if fault_config['geometry'].get('follows'):
            return
        start, end = fault_config['geometry']['sample_positions']
        # print(f"Updating fault {fault_name} geometry with samples from position {start} to {end}")
        sample_values = samples[start:end]
        # Update fault geometry
        start_time_geometry = time.time()
        # Followers returned above.  Every master candidate must replay its
        # perturbation from the frozen reference; there is no persistent
        # "geometry already updated" state across candidates.
        self.multifaults.update_fault_geometry(
            fault_names=[fault_name],
            perturbations=sample_values,
            **fault_config['method_parameters']['update_fault_geometry'],
        )
        end_time_geometry = time.time()
        log_time(start_time_geometry, end_time_geometry, "Execution time for updating fault geometry", log_enabled)
        # Update mesh
        start_time_mesh = time.time()
        if not self.multifaults.faults_dict[fault_name].mesh_valid:
            self.multifaults.update_mesh(
                fault_names=[fault_name],
                **fault_config['method_parameters']['update_mesh'],
            )
        end_time_mesh = time.time()
        log_time(start_time_mesh, end_time_mesh, "Execution time for updating mesh", log_enabled)
        # Optionally update fault areas
        if update_areas:
            self.multifaults.get_fault_areas(fault_names=[fault_name])

    def _update_fault_GFs_and_Laplacian(
            self, fault_name, fault_config, update_laplacian=True,
            log_enabled=False):
        """Refresh observation physics and requested candidate smoothing.

        Green's functions depend on source position and are always refreshed
        after nonlinear geometry updates. Laplacian construction is gated
        separately because alpha-disabled targets do not consume smoothing.
        """
        # Update GFs
        start_time_GFs = time.time()
        self.multifaults.update_GFs(fault_names=[fault_name], **fault_config['method_parameters']['update_GFs'])
        end_time_GFs = time.time()
        log_time(start_time_GFs, end_time_GFs, "Execution time for updating GFs", log_enabled)
        # Update Laplacian
        start_time_Laplacian = time.time()
        if (
            update_laplacian
            and (
                not self.multifaults.faults_dict[fault_name].laplacian_valid
                or getattr(
                    self.multifaults.faults_dict[fault_name], 'GL', None
                ) is None
            )
        ):
            self.multifaults.update_Laplacian(
                fault_names=[fault_name],
                **fault_config['method_parameters']['update_Laplacian'],
            )
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
            gl_list = [fault.GL for fault in self.multifaults.faults if hasattr(fault, 'GL') and fault.GL is not None]
            if gl_list:
                GL_combined = block_diag(gl_list).toarray()
            else:
                GL_combined = np.zeros((0, G_combined.shape[1]))
    
        sigmas = self._dataset_sigmas_from_samples(samples)
        if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            return -np.inf
    
        # Score each independent data block directly. The fixed log|C_k|
        # constant remains omitted here to preserve this solver's established
        # absolute-likelihood convention; sigma-dependent terms are retained.
        linear_sample = samples[self.linear_sample_start_position:]
        data_log_likelihood = 0.0
        st = 0
        for ind, idataname in enumerate(self.datanames):
            ed = st + len(self.obs_dict[idataname])
            metric = self.data_covariance_metrics[idataname]
            residual = (
                G_combined[st:ed, :] @ linear_sample
                - self.observations[st:ed]
            )
            data_log_likelihood += gaussian_log_likelihood(
                residual,
                metric,
                sigmas[ind],
                include_base_logdet=False,
            )
            st = ed
    
        # Check if alpha smoothing is enabled
        if not self.config.alpha_enabled:
            return data_log_likelihood
    
        # Resolve the same complete physical alpha state used by result
        # reporting; non-smoothing sources are excluded by this adapter.
        alpha_faults = self._smoothing_alphas_from_samples(samples)
        size_faults = self._get_smoothing_source_param_sizes()
        alpha = np.hstack([[alpha_faults[ind]] * size_faults[ind] for ind in range(len(alpha_faults))])
    
        # Compute smoothness log-likelihood (only smoothing sources)
        linear_sample_slip_only = samples[self.smoothing_slip_only_positions]
        smooth_log_likelihood = compute_smooth_log_likelihood(GL_combined, linear_sample_slip_only, alpha)
    
        # Return the total log-likelihood
        return data_log_likelihood + smooth_log_likelihood

    def _build_smc_fj_quadratic_workspace(self, GL_combined=None):
        """Prepare exact residual/Gram blocks for the current geometry.

        Fixed-geometry targets retain this immutable workspace across all
        candidates.  Nonlinear targets build it transiently after publishing
        each candidate's Green functions and Laplacian, so no geometry sample
        is cached across candidates or MPI ranks.
        """
        G_combined = np.hstack([
            fault.Gassembled for fault in self.multifaults.faults
        ])
        data_blocks = []
        start = 0
        for data_name in self.datanames:
            end = start + len(self.obs_dict[data_name])
            metric = self.data_covariance_metrics[data_name]
            data_blocks.append(LeastSquaresBlock.prepare(
                metric.whiten(G_combined[start:end, :]),
                metric.whiten(self.observations[start:end]),
                name=f"data:{data_name}",
            ))
            start = end

        smoothing_blocks = []
        if self.config.alpha_enabled:
            if GL_combined is None:
                gl_list = [
                    fault.GL for fault in self.multifaults.faults
                    if hasattr(fault, 'GL') and fault.GL is not None
                ]
                GL_combined = (
                    block_diag(gl_list).toarray()
                    if gl_list
                    else np.zeros((0, G_combined.shape[1]))
                )
            if hasattr(GL_combined, 'toarray'):
                GL_combined = GL_combined.toarray()
            GL_combined = np.asarray(GL_combined, dtype=float)
            GL_combined_poly = self.combine_GL_poly(GL_combined)
            sizes = self._get_smoothing_source_param_sizes()
            if sum(sizes) != GL_combined_poly.shape[0]:
                raise ValueError(
                    "SMC_FJ smoothing row layout does not match the resolved "
                    "per-source alpha layout"
                )
            row_start = 0
            for source_index, size in enumerate(sizes):
                row_end = row_start + size
                smoothing_blocks.append(LeastSquaresBlock.prepare(
                    GL_combined_poly[row_start:row_end, :],
                    name=f"smoothing:{source_index}",
                ))
                row_start = row_end

        return _SMCFJQuadraticWorkspace(
            G_combined=G_combined,
            data_blocks=tuple(data_blocks),
            smoothing_blocks=tuple(smoothing_blocks),
        )

    def _compute_likelihoods_smc_fj(
            self, samples, GL_combined=None, A=None, b=None, Aeq=None,
            beq=None, lb=None, ub=None, x0=None, opts=None,
            smooth_prior_weight=1.0, magnitude_log_prior=False,
            decay_rate=0.1, quadratic_workspace=None):
        """Solve and score one SMC-FJ candidate using an exact QP objective.

        ``quadratic_workspace`` is supplied only by a fixed-geometry target.
        Otherwise the current candidate geometry is prepared transiently.
        Posterior data and smoothing scores are evaluated from the same frozen
        residual blocks that assemble the conditional quadratic objective.
        This keeps source/poly column placement and alpha-group row placement
        identical between the solve and the score.

        SMC-FJ eliminates the conditional linear parameters with the same
        Gaussian/Laplace curvature contract whether smoothing is enabled or
        not.  Alpha controls only the presence of smoothing blocks; it never
        switches the target to a profile likelihood.  The curvature term is
        therefore computed from the complete candidate Hessian before the QP
        solve and requires that Hessian to be positive definite.

        Bounds and linear constraints are enforced by the conditional QP.
        The curvature is still the full-space Gaussian term and does not
        include a truncated-Gaussian probability mass, so constrained results
        retain the documented FJ/Laplace approximation rather than claiming an
        exact constrained marginal density.
        """
        workspace = quadratic_workspace
        if workspace is None:
            workspace = self._build_smc_fj_quadratic_workspace(GL_combined)

        sigmas = self._dataset_sigmas_from_samples(samples)
        if np.any(~np.isfinite(sigmas)) or np.any(sigmas <= 0.0):
            return -np.inf
        if len(sigmas) != len(workspace.data_blocks):
            raise ValueError("SMC_FJ sigma/data block cardinality mismatch")

        weighted_blocks = [
            (block, 1.0 / sigma**2)
            for block, sigma in zip(workspace.data_blocks, sigmas)
        ]
        alpha_faults = None
        if self.config.alpha_enabled:
            alpha_faults = self._smoothing_alphas_from_samples(samples)
            if (
                len(alpha_faults) != len(workspace.smoothing_blocks)
                or np.any(~np.isfinite(alpha_faults))
                or np.any(alpha_faults <= 0.0)
            ):
                return -np.inf
            weighted_blocks.extend(
                (block, 1.0 / alpha_value**2)
                for block, alpha_value in zip(
                    workspace.smoothing_blocks, alpha_faults
                )
            )

        H, q = assemble_quadratic_objective(
            weighted_blocks,
            n_parameters=workspace.G_combined.shape[1],
        )
        # Start each candidate transaction with no publishable linear result.
        # A curvature or QP failure must never leave the previous candidate's
        # model marked as current.
        self.G_combined = workspace.G_combined
        self.mpost = None
        self._last_linear_solve_valid = False
        try:
            curvature_log_term = gaussian_curvature_log_term(
                H, name="SMC_FJ conditional Hessian"
            )
        except ValueError as error:
            return self._record_smc_fj_candidate_failure(
                error, stage="conditional curvature evaluation"
            )
        try:
            mpost = self.least_squares_quadratic_inversion(
                H, q, reg=0, A=A, b=b, Aeq=Aeq, beq=beq,
                lb=lb, ub=ub, x0=x0, opts=opts,
            )
        except Exception as error:
            return self._record_smc_fj_candidate_failure(
                error, stage="conditional QP solve"
            )
        self._last_linear_solve_valid = True
        self.mpost = mpost
        complete_sample = np.hstack((
            samples[:self.linear_sample_start_position], mpost
        ))

        data_quadratic = sum(
            weighted_residual_quadratic(block, mpost, 1.0 / sigma**2)
            for block, sigma in zip(workspace.data_blocks, sigmas)
        )
        sigma_logdet = sum(
            block.matrix.shape[0] * np.log(sigma**2)
            for block, sigma in zip(workspace.data_blocks, sigmas)
        )
        data_log_likelihood = -0.5 * (data_quadratic + sigma_logdet)

        magnitude_log_prior_value = 0.0
        if magnitude_log_prior:
            magnitude_log_prior_value = self.compute_magnitude_log_prior(
                complete_sample, decay_rate
            )

        # Score the exact source/poly blocks used in H.  Reconstructing a
        # separate slip-only GL representation here would duplicate the row
        # and column layout contract and could let the solve and posterior
        # silently diverge when non-smoothing or polynomial columns exist.
        smooth_log_likelihood = 0.0
        if self.config.alpha_enabled:
            smooth_log_likelihood = -0.5 * sum(
                weighted_residual_quadratic(
                    block, mpost, 1.0 / alpha_value**2
                )
                + block.matrix.shape[0] * np.log(alpha_value**2)
                for block, alpha_value in zip(
                    workspace.smoothing_blocks, alpha_faults
                )
            )
        return (
            data_log_likelihood
            + smooth_log_likelihood * smooth_prior_weight
            + magnitude_log_prior_value
            + curvature_log_term
        )

    def _record_smc_fj_candidate_failure(self, error, *, stage):
        """Reject one invalid conditional candidate without aborting SMC.

        Geometry, curvature, and constrained-QP failures belong to individual
        candidates.  They must clear transient linear state and receive the
        shared invalid likelihood; configuration/programming failures outside
        this conditional transaction remain visible to the caller.
        """
        self.mpost = None
        self._last_linear_solve_valid = False
        self.invalid_smc_fj_candidate_count = (
            getattr(self, 'invalid_smc_fj_candidate_count', 0) + 1
        )
        parallel_rank = getattr(self.config, 'parallel_rank', None)
        if (
            (parallel_rank is None or parallel_rank == 0)
            and not getattr(self, '_smc_fj_candidate_failure_warned', False)
        ):
            warnings.warn(
                f"SMC_FJ candidate rejected during {stage} "
                f"({type(error).__name__}: {error}). The full constraint set "
                f"was retained and this sample receives log-likelihood "
                f"{_INVALID_CONSTRAINED_SOLVE_LOGLIKE:g}. Inspect Hessian "
                "conditioning, sigma/alpha bounds, and "
                "get_constraint_snapshot(validate=True).",
                RuntimeWarning,
                stacklevel=2,
            )
            self._smc_fj_candidate_failure_warned = True
        return _INVALID_CONSTRAINED_SOLVE_LOGLIKE

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
        # Compute using lsqlin equivalent to the lsqlin in matlab.
        opts = {'show_progress': False} if opts is None else dict(opts)
        opts.setdefault('show_progress', False)
        ret = lsqlin.lsqlin(C, d, reg, A, b, Aeq, beq, lb, ub, x0, opts)
        _validate_lsqlin_status(ret, context="SMC_FJ constrained least squares")
        mpost = ret['x']
        # Store mpost
        self.mpost = lsqlin.cvxopt_to_numpy_matrix(mpost)

        return self.mpost

    def least_squares_quadratic_inversion(
            self, H, q, reg=0, A=None, b=None, Aeq=None, beq=None,
            lb=None, ub=None, x0=None, opts=None):
        """Solve the exact prepared form of the SMC-FJ conditional problem."""
        opts = {'show_progress': False} if opts is None else dict(opts)
        opts.setdefault('show_progress', False)
        ret = lsqlin.lsqlin_quadratic(
            H, q, reg, A, b, Aeq, beq, lb, ub, x0, opts
        )
        _validate_lsqlin_status(
            ret, context="SMC_FJ constrained quadratic solve"
        )
        self.mpost = lsqlin.cvxopt_to_numpy_matrix(ret['x'])
        return self.mpost
    
    def _get_source_param_sizes(self):
        """Return list of source parameter counts for each fault, using adapters when available."""
        sizes = []
        for fault in self.multifaults.faults:
            if hasattr(self.multifaults, 'adapters') and fault.name in self.multifaults.adapters:
                sizes.append(self.multifaults.adapters[fault.name].get_n_source_params())
            else:
                sizes.append(len(fault.patch) * len(FaultAdapter._canonicalize_slipdir(fault.slipdir)))
        return sizes

    def _get_smoothing_source_param_sizes(self):
        """Return list of source parameter counts for smoothing-capable sources only.

        Sources are considered smoothing-capable when they have a non-None GL
        (Laplacian) matrix after GF construction.
        """
        sizes = []
        for fault in self.multifaults.faults:
            has_gl = hasattr(fault, 'GL') and fault.GL is not None
            if not has_gl:
                continue
            if hasattr(self.multifaults, 'adapters') and fault.name in self.multifaults.adapters:
                sizes.append(self.multifaults.adapters[fault.name].get_n_source_params())
            else:
                sizes.append(len(fault.patch) * len(FaultAdapter._canonicalize_slipdir(fault.slipdir)))
        return sizes

    def combine_GL_poly(self, GL_combined=None):
        if GL_combined is None:
            GL_combined_poly = []
            for fault in self.multifaults.faults:
                # Check if this source supports smoothing (has GL)
                has_gl = hasattr(fault, 'GL') and fault.GL is not None
                if has_gl:
                    poly_positions = self.poly_positions.get(fault.name, (0, 0))
                    combined = np.zeros((fault.GL.shape[0], fault.GL.shape[1] + poly_positions[1] - poly_positions[0]))
                    combined[:, :fault.GL.shape[1]] = fault.GL.toarray()
                    GL_combined_poly.append(combined)
                else:
                    # For sources without smoothing, create zero-row placeholder
                    slip_st, slip_end = self.slip_positions.get(fault.name, (0, 0))
                    poly_st, poly_end = self.poly_positions.get(fault.name, (0, 0))
                    n_params = (slip_end - slip_st) + (poly_end - poly_st)
                    if n_params > 0:
                        GL_combined_poly.append(np.zeros((0, n_params)))
            if GL_combined_poly:
                self.GL_combined_poly = scipy.linalg.block_diag(*GL_combined_poly)
            else:
                self.GL_combined_poly = np.zeros((0, 0))

            return self.GL_combined_poly

        # Candidate matrices contain only smoothing-capable source blocks.
        # Expand those blocks into the same source/poly column layout used by
        # Gassembled. Consuming the explicit matrix is essential: returning an
        # initialization cache would mix current smooth likelihood with
        # baseline smoothing rows in one nonlinear candidate.
        if hasattr(GL_combined, 'toarray'):
            GL_combined = GL_combined.toarray()
        GL_combined = np.asarray(GL_combined, dtype=float)
        if GL_combined.ndim != 2:
            raise ValueError("GL_combined must be a two-dimensional matrix.")

        expanded_blocks = []
        row_start = 0
        col_start = 0
        for fault in self.multifaults.faults:
            has_gl = hasattr(fault, 'GL') and fault.GL is not None
            poly_start, poly_end = self.poly_positions.get(
                fault.name, (0, 0)
            )
            n_poly = poly_end - poly_start
            if has_gl:
                n_rows, n_cols = fault.GL.shape
                row_end = row_start + n_rows
                col_end = col_start + n_cols
                if (
                    row_end > GL_combined.shape[0]
                    or col_end > GL_combined.shape[1]
                ):
                    raise ValueError(
                        "GL_combined shape is inconsistent with current "
                        f"smoothing source '{fault.name}'."
                    )
                block = np.zeros((n_rows, n_cols + n_poly))
                block[:, :n_cols] = GL_combined[
                    row_start:row_end, col_start:col_end
                ]
                expanded_blocks.append(block)
                row_start = row_end
                col_start = col_end
            else:
                slip_start, slip_end = self.slip_positions.get(
                    fault.name, (0, 0)
                )
                n_params = (slip_end - slip_start) + n_poly
                if n_params > 0:
                    expanded_blocks.append(np.zeros((0, n_params)))

        if (
            row_start != GL_combined.shape[0]
            or col_start != GL_combined.shape[1]
        ):
            raise ValueError(
                "GL_combined contains rows or columns not described by the "
                "current smoothing sources."
            )
        if expanded_blocks:
            self.GL_combined_poly = scipy.linalg.block_diag(*expanded_blocks)
        else:
            self.GL_combined_poly = np.zeros((0, 0))
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
        self.config.bayesian_sampling_mode = normalize_bayesian_sampling_mode(
            value
        )

    @property
    def geodata(self):
        return self.config.geodata['data']

    @geodata.setter
    def geodata(self, value):
        self.config.geodata['data'] = value

    @property
    def sigmas(self):
        """Retired ambiguous shortcut; use explicit config/result state."""
        raise AttributeError(
            "'sigmas' is not a Bayesian inversion state property. Read or "
            "change configured values through inversion.config.sigmas; after "
            "returnModel(), read current_data_sigmas for the activated "
            "physical per-dataset values."
        )

    @property
    def alpha(self):
        """Retired ambiguous shortcut; use explicit config/result state."""
        raise AttributeError(
            "'alpha' is not a Bayesian inversion state property. Read or "
            "change configured values through inversion.config.alpha; after "
            "returnModel(), read current_smoothing_alphas or "
            "current_alpha_group_values for activated physical values."
        )

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

# EOF
