"""Independent nonlinear geometry SMC inversion.

This module is intentionally separate from ``exploremultifaults_smc``.  It
keeps the same CSI forward-model path and the same SMC sampler, but parameter
bookkeeping is handled by explicit ``ParameterSpec`` records instead of the old
``param_keys`` / ``param_index`` layout.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, namedtuple
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import yaml
from mpi4py import MPI
from numba import njit
from scipy.stats import uniform

try:
    import h5py
except Exception:  # pragma: no cover - optional runtime dependency
    h5py = None

from csi import SourceInv, planarfault

from .config.nonlinear_geometry_config import NonlinearGeometryConfig
from .data_corrections import (
    assign_parameter_slices,
    build_correction_design_matrix,
    correction_coefficients_from_theta,
)
from .logging_utils.mpi_logging import ensure_default_logging
from .smc_mpi_nonlinear import SMC_samples_parallel_mpi_nonlinear
from .smc_convergence import evaluate_smc_convergence, write_convergence_report

logger = logging.getLogger(__name__)


@njit
def logpdf_multivariate_normal(x, mean, inv_cov, logdet):
    norm_const = -0.5 * logdet
    x_mu = np.subtract(x, mean)
    solution = np.dot(inv_cov, x_mu)
    result = -0.5 * np.dot(x_mu, solution) + norm_const
    return result


@njit
def compute_data_log_likelihood(simulations, observations, inv_cov, log_cov_det):
    return logpdf_multivariate_normal(observations, simulations, inv_cov, log_cov_det)


@njit
def compute_log_prior(samples, lb, ub):
    if np.any((samples < lb) | (samples > ub)):
        return -np.inf
    return 0.0


NT1 = namedtuple("NT1", "N Neff target LB UB")
NT2 = namedtuple(
    "NT2",
    "allsamples postval beta stage covsmpl resmpl sample_stats "
    "fault_parameter_stage_summary",
)


@dataclass(frozen=True)
class ParameterSpec:
    """One sampled parameter in the nonlinear geometry parameter vector."""

    name: str
    group: str
    local_name: str
    index: int
    prior: Sequence[Any]
    display_name: Optional[str] = None
    fault_name: Optional[str] = None
    data_name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def label(self) -> str:
        return self.display_name or self.name


@dataclass
class LikelihoodEntry:
    data: Any
    dobs: np.ndarray
    cd_inv: np.ndarray
    log_cd_det: float
    vertical: bool


class NonlinearGeometrySMCInversion(SourceInv):
    """SMC nonlinear geometry inversion with explicit parameter registry."""

    config_class = NonlinearGeometryConfig

    def __init__(
        self,
        name,
        mode=None,
        num_faults=None,
        utmzone=None,
        ellps="WGS84",
        lon0=None,
        lat0=None,
        verbose=True,
        fixed_params=None,
        config_file="default_nonlinear_geometry_config.yml",
        geodata=None,
        parallel_rank=None,
    ):
        self.parallel_rank = (
            parallel_rank if parallel_rank is not None else MPI.COMM_WORLD.Get_rank()
        )
        self.verbose = verbose and self.parallel_rank == 0
        ensure_default_logging(verbose=self.verbose)
        self.logger = logging.getLogger(__name__)

        if lon0 is None or lat0 is None:
            lon0, lat0 = self._read_lon_lat_0(config_file, lon0, lat0)
        if lon0 is None or lat0 is None:
            raise ValueError(
                "lon0 and lat0 must be set by arguments or config lon_lat_0"
            )

        super().__init__(
            name,
            utmzone=utmzone,
            ellps=ellps,
            lon0=lon0,
            lat0=lat0,
        )

        self._load_and_set_config(
            config_file=config_file,
            fixed_params=fixed_params,
            geodata=geodata,
            mode=mode,
            num_faults=num_faults,
        )

        self.logger.info(
            "[OK] Nonlinear geometry SMC '%s' initialized with %d parameters",
            name,
            len(self.parameter_specs),
        )

    @staticmethod
    def _read_lon_lat_0(config_file, lon0, lat0):
        if config_file is None:
            return lon0, lat0
        with open(config_file, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
        lon_lat_0 = config.get("lon_lat_0")
        if lon_lat_0:
            if lon0 is None:
                lon0 = lon_lat_0[0]
            if lat0 is None:
                lat0 = lon_lat_0[1]
        return lon0, lat0

    def _load_and_set_config(
        self,
        *,
        config_file,
        fixed_params=None,
        geodata=None,
        mode=None,
        num_faults=None,
    ):
        self.config = self.config_class(
            config_file,
            geodata=geodata,
            verbose=self.verbose,
            parallel_rank=self.parallel_rank,
        )

        self._rename_magnitude_to_slip(self.config.bounds)
        self._rename_magnitude_to_slip(self.config.initial)
        self._rename_magnitude_to_slip(self.config.fixed_params)

        self.nchains = self.config.nchains
        self.chain_length = self.config.chain_length
        self.bounds = self.config.bounds
        self.geodata = self.config.geodata
        self.nfaults = int(num_faults or self.config.nfaults)
        self.faultnames = [f"fault_{i}" for i in range(self.nfaults)]
        self.ndatas = len(self.geodata.get("data", []))
        self.dataFaults = self.config.dataFaults
        self.slip_sampling_mode = self.config.slip_sampling_mode
        self.mode = mode or self.slip_sampling_mode
        self.fault_parameter_order = self._fault_parameter_order(self.mode)

        self.fixed_params = self.config.fixed_params or {}
        if fixed_params:
            self._merge_fixed_params(fixed_params)

        self.fault_alias_map = getattr(self.config, "fault_id_to_alias", None) or {
            fault_name: fault_name.replace("fault_", "F")
            for fault_name in self.faultnames
        }

        self.faults = {
            fault_name: planarfault(
                f"mcmc {fault_name}",
                utmzone=self.utmzone,
                lon0=self.lon0,
                lat0=self.lat0,
                ellps=self.ellps,
                verbose=False,
            )
            for fault_name in self.faultnames
        }

        self.data_correction_specs = list(self.config.data_correction_specs)
        self.sigmas = self._ensure_sigma_config(self.config.sigmas)
        self._build_parameter_registry()

    @staticmethod
    def _rename_magnitude_to_slip(config_section):
        if not isinstance(config_section, dict):
            return
        if "magnitude" in config_section:
            config_section["slip"] = config_section.pop("magnitude")
        for value in config_section.values():
            NonlinearGeometrySMCInversion._rename_magnitude_to_slip(value)

    def _merge_fixed_params(self, fixed_params):
        for fault_name, values in fixed_params.items():
            if isinstance(values, Mapping):
                target = self.fixed_params.setdefault(fault_name, {})
                target.update(values)
            else:
                self.fixed_params[fault_name] = values

    @staticmethod
    def _fault_parameter_order(mode):
        if mode == "ss_ds":
            return [
                "lon",
                "lat",
                "depth",
                "dip",
                "width",
                "length",
                "strike",
                "strikeslip",
                "dipslip",
            ]
        if mode == "mag_rake":
            return [
                "lon",
                "lat",
                "depth",
                "dip",
                "width",
                "length",
                "strike",
                "slip",
                "rake",
            ]
        raise ValueError("Invalid slip_sampling_mode. Expected 'ss_ds' or 'mag_rake'.")

    def _ensure_sigma_config(self, sigmas):
        if sigmas is not None:
            return sigmas
        values = np.ones(self.ndatas, dtype=float)
        return {
            "mode": "individual",
            "update": np.zeros(self.ndatas, dtype=bool),
            "values": values,
            "dataset_param_indices": np.arange(self.ndatas, dtype=int),
            "updatable_param_indices": np.full(self.ndatas, -1, dtype=int),
            "log_scaled": False,
            "num_datasets": self.ndatas,
            "total_params": self.ndatas,
            "updatable_params": 0,
            "bounds": {"defaults": ["Uniform", 0.1, 9.9]},
            "groups": None,
        }

    def _build_parameter_registry(self):
        self.parameter_specs: List[ParameterSpec] = []
        self.parameter_index: Dict[str, int] = {}
        self.parameter_groups: Dict[str, List[ParameterSpec]] = defaultdict(list)
        self.fault_parameter_specs: Dict[str, List[ParameterSpec]] = {
            fault_name: [] for fault_name in self.faultnames
        }
        self.data_correction_parameter_specs: Dict[str, List[ParameterSpec]] = {}
        self.sigma_parameter_specs: List[ParameterSpec] = []

        for fault_name in self.faultnames:
            merged_bounds = self._merged_fault_bounds(fault_name)
            for local_name in self.fault_parameter_order:
                if local_name in self.fixed_params.get(fault_name, {}):
                    continue
                if local_name not in merged_bounds:
                    raise ValueError(
                        f"No prior bound for {fault_name}.{local_name}; "
                        "set it in bounds.defaults or bounds.<fault_name>"
                    )
                self._register_parameter(
                    name=f"faults.{fault_name}.{local_name}",
                    group="faults",
                    local_name=local_name,
                    prior=merged_bounds[local_name],
                    display_name=f"{self.fault_alias_map.get(fault_name, fault_name)}.{local_name}",
                    fault_name=fault_name,
                )

        self.data_correction_specs = assign_parameter_slices(
            self.data_correction_specs,
            start_index=len(self.parameter_specs),
        )
        self.data_correction_specs_by_name = {
            spec.data_name: spec for spec in self.data_correction_specs
        }
        for correction_spec in self.data_correction_specs:
            correction_params = []
            if correction_spec.sampled:
                for offset, local_name in enumerate(correction_spec.parameter_names):
                    index = correction_spec.parameter_slice.start + offset
                    spec = self._register_parameter(
                        name=f"data_corrections.{correction_spec.data_name}.{local_name}",
                        group="data_corrections",
                        local_name=local_name,
                        prior=correction_spec.priors[local_name],
                        display_name=correction_spec.display_names.get(local_name),
                        data_name=correction_spec.data_name,
                        index=index,
                        metadata={"transform": correction_spec.transform},
                    )
                    correction_params.append(spec)
            self.data_correction_parameter_specs[correction_spec.data_name] = (
                correction_params
            )

        self._register_sigma_parameters()

    def _register_parameter(
        self,
        *,
        name,
        group,
        local_name,
        prior,
        display_name=None,
        fault_name=None,
        data_name=None,
        index=None,
        metadata=None,
    ):
        if index is None:
            index = len(self.parameter_specs)
        if index != len(self.parameter_specs):
            raise ValueError(
                f"Parameter index mismatch for {name}: got {index}, "
                f"expected {len(self.parameter_specs)}"
            )
        spec = ParameterSpec(
            name=name,
            group=group,
            local_name=local_name,
            index=index,
            prior=prior,
            display_name=display_name,
            fault_name=fault_name,
            data_name=data_name,
            metadata=dict(metadata or {}),
        )
        self.parameter_specs.append(spec)
        self.parameter_index[name] = index
        self.parameter_groups[group].append(spec)
        if fault_name is not None:
            self.fault_parameter_specs.setdefault(fault_name, []).append(spec)
        return spec

    def _merged_fault_bounds(self, fault_name):
        default_bounds = self.bounds.get("defaults", {})
        fault_bounds = self.bounds.get(fault_name, {})
        return {**default_bounds, **fault_bounds}

    def _register_sigma_parameters(self):
        update = np.asarray(self.sigmas.get("update", []), dtype=bool)
        if update.size == 0:
            return

        bounds = self.sigmas.get("bounds", {})
        names = self._sigma_group_names()
        for sigma_param_index, should_update in enumerate(update):
            if not should_update:
                continue
            local_name = names[sigma_param_index]
            bound_key = f"sigma_{sigma_param_index}"
            prior = bounds.get(bound_key, bounds.get("defaults"))
            if prior is None:
                raise ValueError(
                    f"No sigma prior bound for {bound_key}; set geodata.sigmas.bounds.defaults"
                )
            spec = self._register_parameter(
                name=f"sigmas.{local_name}",
                group="sigmas",
                local_name=local_name,
                prior=prior,
                display_name=local_name,
                metadata={"sigma_param_index": sigma_param_index},
            )
            self.sigma_parameter_specs.append(spec)

    def _sigma_group_names(self):
        mode = self.sigmas.get("mode", "individual")
        total_params = int(self.sigmas.get("total_params", 0))
        if mode == "single":
            return ["all"]
        if mode == "individual":
            datanames = [data.name for data in self.geodata.get("data", [])]
            return datanames[:total_params]
        if mode == "grouped":
            groups = self.sigmas.get("groups") or {}
            return list(groups.keys())
        return [f"sigma_{i}" for i in range(total_params)]

    def setPriors(self, bounds=None, datas=None, initialSample=None, sigmas=None):
        """Create prior distributions in ``ParameterSpec`` order."""
        if bounds is not None:
            self.bounds.update(bounds)
        if sigmas is not None:
            self.sigmas.update(sigmas)
        if bounds is not None or sigmas is not None:
            self._build_parameter_registry()

        initialSample = dict(initialSample or {})
        self.Priors = []
        self.initSampleVec = []
        self.initialSample = {}

        for spec in self.parameter_specs:
            pm_func = self._make_prior_distribution(spec.prior)
            value = self._initial_value_for_spec(spec, initialSample, pm_func)
            self.Priors.append(pm_func)
            self.initSampleVec.append(value)
            self.initialSample[spec.name] = value

        self.lb, self.ub = self.parameter_bounds()
        return None

    def _make_prior_distribution(self, bound):
        if not isinstance(bound, Sequence) or isinstance(bound, str) or len(bound) < 3:
            raise ValueError(f"Invalid prior bound: {bound!r}")
        if bound[0] != "Uniform":
            raise ValueError(
                "NonlinearGeometrySMCInversion currently requires Uniform priors "
                "because the SMC proposal uses finite lower/upper bounds."
            )
        return uniform(*bound[1:])

    def _initial_value_for_spec(self, spec, initial_sample, pm_func):
        candidates = [spec.name, spec.label]
        if spec.fault_name:
            candidates.append(f"{spec.fault_name}_{spec.local_name}")
        for key in candidates:
            if key in initial_sample:
                return float(initial_sample[key])
        return float(pm_func.rvs())

    def parameter_names(self):
        return [spec.name for spec in self.parameter_specs]

    def parameter_display_names(self):
        return [spec.label for spec in self.parameter_specs]

    def parameter_bounds(self):
        lb = []
        ub = []
        for spec in self.parameter_specs:
            dist_name, lower, scale = spec.prior[:3]
            if dist_name != "Uniform":
                raise ValueError(f"{spec.name}: only Uniform priors have finite SMC bounds")
            lb.append(float(lower))
            ub.append(float(lower) + float(scale))
        return np.asarray(lb, dtype=float), np.asarray(ub, dtype=float)

    def setLikelihood(self, datas=None, verticals=None):
        """Build likelihood entries from geodetic data covariance matrices."""
        self.datas = list(datas if datas is not None else self.geodata["data"])
        if verticals is None:
            verticals = self.geodata.get("verticals", [True] * len(self.datas))
        if isinstance(verticals, bool):
            verticals = [verticals] * len(self.datas)
        if len(verticals) != len(self.datas):
            raise ValueError("Length of verticals must match number of data sets")
        self.verticals = list(verticals)

        self.Likelihoods: List[LikelihoodEntry] = []
        for data, vertical in zip(self.datas, self.verticals):
            if not hasattr(data, "Cd"):
                raise ValueError(f"No data covariance for data set {data.name}")
            dobs = self._observation_vector(data, vertical)
            cd = np.asarray(data.Cd, dtype=float)
            cd_inv = np.linalg.inv(cd)
            sign, logdet = np.linalg.slogdet(cd)
            log_cd_det = float(sign * logdet)
            self.Likelihoods.append(
                LikelihoodEntry(
                    data=data,
                    dobs=np.asarray(dobs, dtype=float).reshape(-1),
                    cd_inv=cd_inv,
                    log_cd_det=log_cd_det,
                    vertical=bool(vertical),
                )
            )
        return None

    @staticmethod
    def _observation_vector(data, vertical=True):
        dtype = str(data.dtype).lower()
        if dtype == "gps":
            vel = np.asarray(data.vel_enu)
            return vel.flatten() if vertical else vel[:, :-1].flatten()
        if dtype in {"insar", "leveling"}:
            return np.asarray(data.vel).reshape(-1)
        if dtype == "crossfaultoffset":
            return np.asarray(data.data_vector).reshape(-1)
        raise ValueError(f"Unsupported data type: {data.dtype}")

    def build_fault_params(self, theta, fault_name):
        theta = np.asarray(theta, dtype=float)
        params = {
            spec.local_name: float(theta[spec.index])
            for spec in self.fault_parameter_specs.get(fault_name, [])
        }
        fixed = {
            key: value
            for key, value in self.fixed_params.get(fault_name, {}).items()
            if key in self.fault_parameter_order
        }
        params.update(fixed)
        missing = [key for key in self.fault_parameter_order if key not in params]
        if missing:
            raise ValueError(f"{fault_name}: missing parameters {missing}")
        return params

    def Predict(self, theta, data, vertical=True, faultnames=None, updatepatch=True):
        """Predict one data set and apply its configured data correction."""
        theta = np.asarray(theta, dtype=float)
        faultnames = self._normalize_faultnames(faultnames)
        faults_for_data = []

        for fault_name in faultnames:
            fault = self.faults[fault_name]
            self._update_fault_for_data(
                fault=fault,
                fault_name=fault_name,
                theta=theta,
                data=data,
                vertical=vertical,
                updatepatch=updatepatch,
            )
            faults_for_data.append(fault)

        for fault_name in set(self.faultnames).difference(faultnames):
            self.faults[fault_name].buildGFs(
                data,
                vertical=vertical,
                slipdir="sd",
                verbose=False,
                method="empty",
            )

        data.buildsynth(faults_for_data)
        simulation = self._synthetic_vector(data, vertical)
        correction = self._data_correction_vector(theta, data, vertical)
        if correction is not None:
            simulation = simulation + correction
            self._write_synthetic_vector(data, simulation, vertical)
        return simulation

    def _normalize_faultnames(self, faultnames):
        if faultnames is None:
            return list(self.faultnames)
        if isinstance(faultnames, str):
            faultnames = [faultnames]
        unknown = set(faultnames) - set(self.faultnames)
        if unknown:
            raise ValueError(f"Unknown fault names: {sorted(unknown)}")
        return list(faultnames)

    def _update_fault_for_data(
        self,
        *,
        fault,
        fault_name,
        theta,
        data,
        vertical,
        updatepatch,
    ):
        params = self.build_fault_params(theta, fault_name)
        lon = params["lon"]
        lat = params["lat"]
        depth = params["depth"]
        strike = params["strike"]
        dip = params["dip"]
        length = params["length"]
        width = params["width"]

        if dip > 90:
            dip = 180 - dip
            strike = (strike + 180) % 360
        elif dip < 0:
            dip = -dip
            strike = (strike + 180) % 360

        if updatepatch:
            fault.buildPatches(
                lon,
                lat,
                depth,
                strike,
                dip,
                length,
                width,
                1,
                1,
                verbose=False,
            )

        fault.buildGFs(data, vertical=vertical, slipdir="sd", verbose=False)
        strikeslip, dipslip = self._slip_components(params)
        fault.slip[:, 0] = strikeslip
        fault.slip[:, 1] = dipslip

    def _slip_components(self, params):
        if self.mode == "ss_ds":
            return params["strikeslip"], params["dipslip"]
        slip = params["slip"]
        rake = np.radians(params["rake"])
        return slip * np.cos(rake), slip * np.sin(rake)

    @staticmethod
    def _synthetic_vector(data, vertical=True):
        dtype = str(data.dtype).lower()
        if dtype == "gps":
            synth = np.asarray(data.synth)
            return synth.flatten() if vertical else synth[:, :-1].flatten()
        if dtype in {"insar", "leveling"}:
            return np.asarray(data.synth).reshape(-1)
        if dtype == "crossfaultoffset":
            if hasattr(data, "synth_vector"):
                return np.asarray(data.synth_vector).reshape(-1)
            return np.asarray(data.synth).reshape(-1)
        raise ValueError(f"Unsupported data type: {data.dtype}")

    @staticmethod
    def _write_synthetic_vector(data, vector, vertical=True):
        dtype = str(data.dtype).lower()
        vector = np.asarray(vector, dtype=float).reshape(-1)
        if dtype == "gps":
            synth = np.asarray(data.synth, dtype=float).copy()
            if vertical:
                data.synth = vector.reshape(synth.shape)
            else:
                synth[:, :-1] = vector.reshape(synth[:, :-1].shape)
                data.synth = synth
            return
        if dtype in {"insar", "leveling"}:
            data.synth = vector.reshape(np.asarray(data.synth).shape)
            return
        if dtype == "crossfaultoffset":
            data.synth = vector.reshape(np.asarray(data.synth).shape)
            data.synth_vector = vector
            return
        raise ValueError(f"Unsupported data type: {data.dtype}")

    def _data_correction_vector(self, theta, data, vertical=True):
        spec = self.data_correction_specs_by_name.get(data.name)
        if spec is None:
            return None
        design_matrix = build_correction_design_matrix(
            data,
            spec.transform,
            vertical=vertical,
        )
        coefficients = correction_coefficients_from_theta(spec, theta)
        return design_matrix.dot(coefficients)

    def _dataset_sigmas_from_theta(self, theta):
        values = np.asarray(self.sigmas["values"], dtype=float).copy()
        for spec in self.sigma_parameter_specs:
            sigma_param_index = spec.metadata["sigma_param_index"]
            values[sigma_param_index] = theta[spec.index]
        dataset_indices = np.asarray(self.sigmas["dataset_param_indices"], dtype=int)
        sigmas = values[dataset_indices]
        if self.sigmas.get("log_scaled", False):
            sigmas = np.power(10.0, sigmas)
        return np.asarray(sigmas, dtype=float)

    def make_target(self, updatepatches=None, dataFaults=None):
        if not hasattr(self, "Priors"):
            self.setPriors()
        if not hasattr(self, "Likelihoods"):
            self.setLikelihood()

        self.lb, self.ub = self.parameter_bounds()
        dataFaults = self._resolved_data_faults(dataFaults)
        updatepatches = self._resolved_updatepatches(updatepatches, dataFaults)

        def target(samples):
            samples = np.asarray(samples, dtype=np.float64)
            log_prior = compute_log_prior(samples, self.lb, self.ub)
            if log_prior == -np.inf:
                return -np.inf

            log_likelihood = 0.0
            sigmas = self._dataset_sigmas_from_theta(samples)
            for i, entry in enumerate(self.Likelihoods):
                simulations = self.Predict(
                    samples,
                    entry.data,
                    vertical=entry.vertical,
                    faultnames=dataFaults[i],
                    updatepatch=updatepatches[i],
                )
                isigma2 = sigmas[i] ** 2
                cd_inv_sigma = np.divide(entry.cd_inv, isigma2)
                log_cd_det_sigma = entry.log_cd_det + np.log(isigma2) * len(entry.dobs)
                log_likelihood += compute_data_log_likelihood(
                    simulations,
                    entry.dobs,
                    cd_inv_sigma,
                    log_cd_det_sigma,
                )
            return log_prior + log_likelihood

        return target

    def _resolved_data_faults(self, dataFaults=None):
        if dataFaults is None:
            dataFaults = self.dataFaults
        if dataFaults is None:
            return [list(self.faultnames) for _ in self.Likelihoods]
        if len(dataFaults) != len(self.Likelihoods):
            raise ValueError("Length of dataFaults must match number of likelihoods")
        return [self._normalize_faultnames(item) for item in dataFaults]

    def _resolved_updatepatches(self, updatepatches, dataFaults):
        if updatepatches is not None:
            if len(updatepatches) != len(dataFaults):
                raise ValueError("Length of updatepatches must match number of data sets")
            return list(updatepatches)

        updatepatches = [False] * len(dataFaults)
        initialized_faults = set()
        for i, faultnames in enumerate(dataFaults):
            new_faults = set(faultnames).difference(initialized_faults)
            if new_faults:
                updatepatches[i] = True
                initialized_faults.update(new_faults)
        if not any(updatepatches) and updatepatches:
            updatepatches[0] = True
        return updatepatches

    def walk(
        self,
        nchains=None,
        chain_length=None,
        comm=None,
        filename="samples.h5",
        save_every=1,
        save_at_interval=True,
        save_at_final=True,
        covariance_epsilon=1e-9,
        amh_a=1.0 / 9.0,
        amh_b=8.0 / 9.0,
        updatepatches=None,
        dataFaults=None,
        diagnose=True,
        diagnose_detail=False,
        convergence_report_file=None,
    ):
        nchains = nchains if nchains is not None else self.nchains
        chain_length = chain_length if chain_length is not None else self.chain_length
        self.dataFaults = dataFaults or self.dataFaults

        target = self.make_target(updatepatches=updatepatches, dataFaults=self.dataFaults)
        opt = NT1(nchains, chain_length, target, self.lb, self.ub)
        samples = NT2(None, None, None, None, None, None, None, None)
        comm = comm or MPI.COMM_WORLD
        rank = comm.Get_rank()
        diagnostic_info = self._fault_parameter_diagnostic_info()

        if rank == 0:
            self.logger.info("Starting nonlinear geometry SMC sampling...")

        final = SMC_samples_parallel_mpi_nonlinear(
            opt,
            samples,
            NT1,
            NT2,
            comm,
            save_at_final,
            save_every,
            save_at_interval,
            covariance_epsilon,
            amh_a,
            amh_b,
            diagnostic_indices=diagnostic_info["indices"],
            diagnostic_parameter_names=diagnostic_info["parameter_names"],
            diagnostic_display_names=diagnostic_info["display_names"],
            diagnostic_lower_bounds=diagnostic_info["lower_bounds"],
            diagnostic_upper_bounds=diagnostic_info["upper_bounds"],
        )

        if rank == 0:
            self.sampler = final._asdict()
            self.save2h5(filename)
            if diagnose:
                self.report_convergence(
                    filename=convergence_report_file,
                    sample_filename=filename,
                    print_report=True,
                    print_detail=diagnose_detail,
                )
            self.logger.info("Finished nonlinear geometry SMC sampling.")
        return None

    def evaluate_convergence(self, **kwargs):
        if not hasattr(self, "sampler"):
            raise ValueError("No sampler results available")
        lb, ub = self.parameter_bounds()
        return evaluate_smc_convergence(
            self.sampler["allsamples"],
            postval=self.sampler.get("postval"),
            beta=self.sampler.get("beta"),
            sample_stats=self.sampler.get("sample_stats"),
            fault_parameter_stage_summary=self.sampler.get(
                "fault_parameter_stage_summary"
            ),
            lower_bounds=lb,
            upper_bounds=ub,
            parameter_names=self.parameter_names(),
            **kwargs,
        )

    def _fault_parameter_diagnostic_info(self):
        specs = [
            spec
            for spec in self.parameter_specs
            if spec.group == "faults"
        ]
        lb, ub = self.parameter_bounds()
        indices = np.asarray([spec.index for spec in specs], dtype=int)
        return {
            "indices": indices,
            "parameter_names": [spec.name for spec in specs],
            "display_names": [spec.label for spec in specs],
            "lower_bounds": lb[indices] if indices.size else None,
            "upper_bounds": ub[indices] if indices.size else None,
        }

    def report_convergence(
        self,
        *,
        filename=None,
        sample_filename=None,
        print_report=True,
        print_detail=False,
        write_text_report=True,
        text_report_file=None,
        **kwargs,
    ):
        """Evaluate, optionally print, and save the lightweight SMC diagnostics."""
        report = self.evaluate_convergence(**kwargs)
        self.convergence_report = report
        sample_key = self._sample_filename_key(sample_filename)
        if filename is None:
            filename = self._default_convergence_report_filename(sample_filename)
        if filename:
            write_convergence_report(report, str(filename))
        if write_text_report:
            if text_report_file is None:
                text_report_file = self._default_convergence_text_report_filename(
                    filename
                )
            if text_report_file:
                self._write_convergence_text_report(report, text_report_file)
        self._last_convergence_sample_key = sample_key
        self._last_convergence_report_file = str(filename) if filename else None
        self._last_convergence_text_report_file = (
            str(text_report_file) if text_report_file else None
        )
        if print_report:
            self._print_convergence_report(
                report,
                filename=filename,
                text_filename=text_report_file,
                print_detail=print_detail,
            )
        return report

    @staticmethod
    def _default_convergence_report_filename(sample_filename=None):
        if sample_filename:
            path = Path(sample_filename)
            return path.with_name(f"{path.stem}_convergence.yml")
        return Path("smc_convergence_report.yml")

    @staticmethod
    def _default_convergence_text_report_filename(report_filename=None):
        if report_filename:
            return Path(report_filename).with_suffix(".txt")
        return Path("smc_convergence_report.txt")

    @staticmethod
    def _sample_filename_key(sample_filename=None):
        if sample_filename is None:
            return None
        path = Path(sample_filename)
        try:
            return str(path.resolve())
        except OSError:
            return str(path)

    def _convergence_report_already_current(self, sample_filename):
        sample_key = self._sample_filename_key(sample_filename)
        if sample_key is None:
            return False
        return (
            hasattr(self, "convergence_report")
            and getattr(self, "_last_convergence_sample_key", None) == sample_key
        )

    def _print_convergence_report(
        self,
        report,
        *,
        filename=None,
        text_filename=None,
        print_detail=False,
    ):
        completed = report.get("completed", {})
        samples = report.get("samples", {})
        warnings = report.get("warnings", [])
        process = report.get("smc_process", {})
        trend = report.get("fault_parameter_trend", {})
        checks = report.get("fault_parameter_checks", {})
        print("\n" + "=" * 72)
        print(f"SMC Convergence Diagnostics | {report.get('status')}")
        print("=" * 72)
        finite_ratio = report.get("postval", {}).get("finite_ratio")
        completed_text = (
            "completed" if completed.get("beta_reached_1") else "incomplete"
        )
        completed_text += f" (beta={completed.get('beta_final')}"
        if finite_ratio is not None:
            completed_text += f", finite postval={finite_ratio:.3f}"
        completed_text += ")"
        print(
            f"Run: {completed_text} | "
            f"Particles: {samples.get('particles')} "
            f"/ recommended >= {samples.get('min_particles')}"
        )

        median = self._fault_metric_summary(
            checks,
            "median_trend_ratio",
            threshold_key="median_trend_ratio_max",
            warning_threshold_key="median_trend_ratio_warning",
            direction="above",
        )
        print("")
        print("Main checks")
        self._print_screen_item(
            self._section_status(median),
            "Median stability",
            self._screen_metric_text(
                "worst drift",
                median,
                threshold_op="<=",
                value_suffix=" x final CI width",
                threshold_text=self._median_stability_threshold_text(median),
            ),
        )
        self._print_screen_exceeded("also over threshold", median)
        if not median.get("available"):
            self._print_screen_detail("note", trend.get("note", "not available"))

        final_prior = self._fault_metric_summary(
            checks,
            "final_ci_to_prior",
            threshold_key="final_ci_to_prior_max",
            direction="above",
        )
        final_max = self._fault_metric_summary(
            checks,
            "final_ci_to_max_ci",
            threshold_key="final_ci_to_max_ci_max",
            direction="above",
        )
        uncertainty_status = self._multi_metric_status([final_prior, final_max])
        self._print_screen_item(
            uncertainty_status,
            "Uncertainty",
            self._screen_metric_text(
                "widest final/prior",
                final_prior,
                threshold_op="<=",
            ),
        )
        self._print_screen_exceeded("also over threshold", final_prior)
        self._print_screen_detail(
            "least CI shrinkage",
            self._screen_metric_value_text(final_max, threshold_op="<="),
        )
        self._print_screen_exceeded("also over threshold", final_max)

        ci_widening = self._fault_metric_summary(
            checks,
            "ci_widening_ratio",
            threshold_key="ci_widening_ratio_max",
            direction="above",
        )
        ci_narrowing = self._fault_metric_summary(
            checks,
            "ci_narrowing_ratio",
            threshold_key="ci_narrowing_ratio_display_min",
            direction="above",
        )
        bound_edge = self._fault_metric_summary(
            checks,
            "bound_distance",
            threshold_key="bound_distance_min",
            direction="below",
        )
        boundary_mass = self._fault_metric_summary(
            checks,
            "boundary_mass",
            threshold_key="boundary_mass_max",
            direction="above",
        )
        bound_status = self._bound_proximity_status(bound_edge, boundary_mass)
        self._print_screen_item(
            bound_status,
            "Bound proximity",
            self._screen_metric_text(
                "closest CI edge",
                bound_edge,
                threshold_op=">=",
                value_suffix=" / prior width",
                threshold_text=(
                    f"info < {self._format_metric(bound_edge.get('threshold'))}"
                ),
            ),
        )
        self._print_screen_exceeded("also near bound", bound_edge)
        self._print_screen_detail(
            "boundary mass",
            self._boundary_mass_text(boundary_mass, checks, bound_edge),
        )
        self._print_screen_exceeded("also mass near", boundary_mass)
        if bound_status != "OK":
            self._print_screen_detail(
                "meaning",
                self._bound_proximity_interpretation(bound_edge, boundary_mass),
            )

        print("")
        print("Additional notes")
        self._print_late_ci_screen_summary(ci_widening, ci_narrowing)
        self._print_screen_item(
            self._process_status_label(process),
            "SMC process",
            self._screen_process_note(process),
        )

        print("")
        print("Action")
        print(
            self._wrap_screen_text(
                self._screen_user_action(
                    self._section_status(median),
                    uncertainty_status,
                    bound_status,
                    self._process_status_label(process),
                ),
                indent="",
            )
        )

        print("")
        print("Reports")
        if text_filename:
            print(f"TXT: {text_filename}")
        if filename:
            print(f"YML: {filename}")
        if warnings:
            print(f"Warnings: {', '.join(warnings)}")
        else:
            print("Warnings: none")
        if print_detail:
            self._print_fault_parameter_check_table(report)
        print("=" * 72)
        import sys

        sys.stdout.flush()

    @classmethod
    def _fault_metric_summary(
        cls,
        checks,
        metric,
        *,
        threshold_key,
        warning_threshold_key=None,
        direction,
    ):
        rows = checks.get("rows", []) if checks else []
        threshold = checks.get("thresholds", {}).get(threshold_key) if checks else None
        warning_threshold = (
            checks.get("thresholds", {}).get(warning_threshold_key)
            if checks and warning_threshold_key
            else None
        )
        valid = [
            row for row in rows
            if cls._metric_is_number(row.get(metric))
        ]
        if not valid:
            return {
                "available": False,
                "metric": metric,
                "threshold": threshold,
                "warning_threshold": warning_threshold,
                "direction": direction,
                "row": None,
                "label": None,
                "value": None,
                "exceeded": [],
                "also_exceeded": [],
            }

        if direction == "below":
            worst = min(valid, key=lambda row: float(row.get(metric)))
            exceeded = [
                row for row in valid
                if threshold is not None and float(row.get(metric)) < float(threshold)
            ]
        else:
            worst = max(valid, key=lambda row: float(row.get(metric)))
            exceeded = [
                row for row in valid
                if threshold is not None and float(row.get(metric)) > float(threshold)
            ]

        return {
            "available": True,
            "metric": metric,
            "threshold": threshold,
            "warning_threshold": warning_threshold,
            "direction": direction,
            "row": worst,
            "label": worst.get("label", worst.get("name")),
            "value": float(worst.get(metric)),
            "exceeded": exceeded,
            "warning_exceeded": [
                row for row in valid
                if (
                    warning_threshold is not None
                    and float(row.get(metric)) > float(warning_threshold)
                )
            ] if direction == "above" else [],
            "also_exceeded": [row for row in exceeded if row is not worst],
            "valid": valid,
        }

    @staticmethod
    def _metric_is_number(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return False
        return np.isfinite(value)

    @classmethod
    def _section_status(cls, summary):
        if not summary.get("available"):
            return "not available"
        if summary.get("warning_exceeded"):
            return "WARNING"
        return "REVIEW" if summary.get("exceeded") else "OK"

    @staticmethod
    def _multi_metric_status(summaries):
        if not any(summary.get("available") for summary in summaries):
            return "not available"
        return "REVIEW" if any(summary.get("exceeded") for summary in summaries) else "OK"

    @classmethod
    def _bound_proximity_status(cls, bound_edge, boundary_mass):
        if boundary_mass.get("exceeded"):
            return "REVIEW"
        if bound_edge.get("exceeded"):
            return "INFO"
        if bound_edge.get("available") or boundary_mass.get("available"):
            return "OK"
        return "not available"

    @staticmethod
    def _process_status_label(process):
        status = str(process.get("status", "")).lower()
        if status == "warning":
            return "WARNING"
        if status == "review":
            return "REVIEW"
        if status == "ok":
            return "OK"
        return "not available"

    @staticmethod
    def _late_ci_status(widening, narrowing):
        if widening.get("exceeded"):
            return "REVIEW"
        if narrowing.get("exceeded"):
            return "INFO"
        if widening.get("available") or narrowing.get("available"):
            return "OK"
        return "not available"

    @staticmethod
    def _late_ci_interpretation(widening, narrowing):
        if widening.get("exceeded"):
            return "credible intervals widened near final beta; review with trend plot"
        if narrowing.get("exceeded"):
            return "uncertainty decreased near final beta"
        if widening.get("available") or narrowing.get("available"):
            return "final-stage CI widths are stable"
        return "late CI behavior is unavailable"

    @classmethod
    def _print_screen_item(cls, status, title, text):
        tag = f"[{status}]"
        prefix = f"{tag:<9} {title:<17}: "
        cls._print_wrapped_screen_line(prefix, text)

    @classmethod
    def _print_screen_detail(cls, label, text):
        prefix = f"{'':9} {label:<17}: "
        cls._print_wrapped_screen_line(prefix, text)

    @staticmethod
    def _print_wrapped_screen_line(prefix, text, *, width=120):
        import textwrap

        body = str(text)
        body_width = max(24, width - len(prefix))
        lines = textwrap.wrap(body, width=body_width) or [""]
        print(prefix + lines[0])
        for line in lines[1:]:
            print(" " * len(prefix) + line)

    @classmethod
    def _screen_metric_text(cls, label, summary, **kwargs):
        value_text = cls._screen_metric_value_text(summary, **kwargs)
        if value_text == "not available":
            return f"{label} not available"
        return f"{label} {value_text}"

    @classmethod
    def _screen_metric_value_text(
        cls,
        summary,
        *,
        threshold_op,
        value_suffix="",
        threshold_label="threshold",
        threshold_text=None,
    ):
        if not summary.get("available"):
            return "not available"
        parameter = cls._metric_parameter_label(summary)
        value = cls._format_metric(summary.get("value"))
        threshold = cls._format_metric(summary.get("threshold"))
        threshold_part = threshold_text or f"{threshold_label} {threshold_op} {threshold}"
        return f"{parameter} = {value}{value_suffix} ({threshold_part})"

    @classmethod
    def _print_metric_summary(
        cls,
        label,
        summary,
        *,
        threshold_op,
        value_suffix="",
        threshold_label="threshold",
        threshold_text=None,
    ):
        if not summary.get("available"):
            print(f"  {label:<19}: not available")
            return
        parameter = cls._metric_parameter_label(summary)
        value = cls._format_metric(summary.get("value"))
        threshold = cls._format_metric(summary.get("threshold"))
        threshold_part = threshold_text or f"{threshold_label} {threshold_op} {threshold}"
        print(
            f"  {label:<19}: {parameter} = {value}{value_suffix} "
            f"({threshold_part})"
        )

    @classmethod
    def _boundary_mass_text(cls, boundary_mass, checks, bound_edge):
        if not boundary_mass.get("available"):
            return "not available"
        threshold = cls._format_metric(boundary_mass.get("threshold"))
        row = None
        if boundary_mass.get("exceeded"):
            row = boundary_mass.get("row")
        elif bound_edge.get("exceeded"):
            edge_name = (bound_edge.get("row") or {}).get("name")
            row = cls._matching_metric_row(boundary_mass, edge_name)
        if row is None:
            return f"none over threshold (review >= {threshold})"
        label = row.get("label", row.get("name", "-"))
        direction = row.get("boundary_mass_direction") or row.get("bound_direction")
        if direction:
            label = f"{label} {direction}"
        return (
            f"{label} = {cls._format_metric(row.get('boundary_mass'))} "
            f"within {cls._boundary_tol_percent(checks)}% prior width "
            f"(review >= {threshold})"
        )

    @classmethod
    def _print_boundary_mass_summary(cls, boundary_mass, checks, bound_edge):
        if not boundary_mass.get("available"):
            print(f"  {'boundary mass':<19}: not available")
            return
        row = None
        if boundary_mass.get("exceeded"):
            row = boundary_mass.get("row")
        elif bound_edge.get("exceeded"):
            edge_name = (bound_edge.get("row") or {}).get("name")
            row = cls._matching_metric_row(boundary_mass, edge_name)
        if row is None:
            print(f"  {'boundary mass':<19}: none over threshold")
            return
        label = row.get("label", row.get("name", "-"))
        direction = row.get("boundary_mass_direction") or row.get("bound_direction")
        if direction:
            label = f"{label} {direction}"
        print(
            f"  {'boundary mass':<19}: "
            f"{label} = {cls._format_metric(row.get('boundary_mass'))} "
            f"within {cls._boundary_tol_percent(checks)}% prior width "
            f"(review >= {cls._format_metric(boundary_mass.get('threshold'))})"
        )

    @staticmethod
    def _matching_metric_row(summary, name):
        if not name:
            return None
        for row in summary.get("valid", []):
            if row.get("name") == name:
                return row
        return None

    @classmethod
    def _boundary_tol_percent(cls, checks):
        threshold = (checks or {}).get("thresholds", {}).get("boundary_tol_fraction")
        try:
            value = float(threshold) * 100.0
        except (TypeError, ValueError):
            value = 1.0
        if not np.isfinite(value):
            value = 1.0
        text = f"{value:.3g}"
        return text

    @classmethod
    def _bound_proximity_interpretation(cls, bound_edge, boundary_mass):
        edge_hit = bool(bound_edge.get("exceeded"))
        mass_hit = bool(boundary_mass.get("exceeded"))
        if edge_hit and mass_hit:
            return (
                "posterior mass is close to a prior bound; expand the bound "
                "and rerun if it is not a physical hard limit"
            )
        if edge_hit:
            return (
                "final CI approaches a prior bound; inspect the trend plot "
                "before changing bounds"
            )
        if mass_hit:
            return (
                "some final posterior samples accumulate near a prior bound; "
                "review the configured prior range"
            )
        if bound_edge.get("available") or boundary_mass.get("available"):
            return "final posterior is not close to configured prior bounds"
        return "prior-bound proximity is unavailable"

    @classmethod
    def _median_stability_threshold_text(cls, summary):
        review = summary.get("threshold")
        warning = summary.get("warning_threshold")
        if warning is None:
            return f"threshold <= {cls._format_metric(review)}"
        return (
            f"OK <= {cls._format_metric(review)}; "
            f"WARNING > {cls._format_metric(warning)}"
        )

    @classmethod
    def _print_exceeded_parameters(cls, label, summary, *, only_if_any=False):
        if not summary.get("available"):
            return
        if summary.get("threshold") is None:
            if only_if_any:
                return
            print(f"  {label:<19}: threshold unavailable")
            return
        rows = summary.get("also_exceeded", [])
        if not rows:
            if only_if_any:
                return
            print(f"  {label:<19}: none")
            return
        print(
            f"  {label:<19}: "
            f"{cls._format_metric_row_list(rows, summary['metric'])}"
        )

    @classmethod
    def _print_screen_exceeded(cls, label, summary):
        if not summary.get("available") or summary.get("threshold") is None:
            return
        rows = summary.get("also_exceeded", [])
        if rows:
            cls._print_screen_detail(
                label,
                cls._format_metric_row_list(rows, summary["metric"]),
            )

    @classmethod
    def _print_late_ci_screen_summary(cls, ci_widening, ci_narrowing):
        status = cls._late_ci_status(ci_widening, ci_narrowing)
        if ci_widening.get("exceeded"):
            text = cls._screen_metric_text(
                "largest widening",
                ci_widening,
                threshold_op="<=",
            )
        elif ci_narrowing.get("exceeded"):
            text = cls._screen_metric_text(
                "strongest narrowing",
                ci_narrowing,
                threshold_op=">=",
                threshold_label="display",
            )
        elif ci_widening.get("available"):
            text = cls._screen_metric_text(
                "largest widening",
                ci_widening,
                threshold_op="<=",
            )
        else:
            text = "not available"
        cls._print_screen_item(status, "Late CI behavior", text)
        if ci_widening.get("exceeded"):
            cls._print_screen_exceeded("also over threshold", ci_widening)
        if status in {"INFO", "REVIEW"}:
            cls._print_screen_detail(
                "meaning",
                cls._late_ci_interpretation(ci_widening, ci_narrowing),
            )

    @staticmethod
    def _screen_process_note(process):
        note = process.get("note", "not available")
        if isinstance(note, str) and note.startswith("early particle degeneracy only"):
            return f"{note}; usually inspect, not fatal"
        return note

    @staticmethod
    def _screen_user_action(median_status, uncertainty_status, bound_status, process_status):
        if median_status == "WARNING":
            return (
                "Inspect fault_parameter_trends.png, then increase particles or "
                "mutation length and compare with an independent run."
            )
        if median_status == "REVIEW":
            return (
                "Inspect fault_parameter_trends.png and compare with an independent "
                "seed before transferring geometry to linear slip inversion."
            )
        if bound_status == "REVIEW":
            return (
                "Inspect fault_parameter_trends.png. Expand a prior bound only when "
                "boundary mass is REVIEW or the bound is not a physical hard limit."
            )
        if bound_status == "INFO":
            return (
                "Inspect fault_parameter_trends.png. The CI edge is relatively close "
                "to a bound, but boundary mass is below the review threshold."
            )
        if uncertainty_status == "REVIEW":
            return (
                "Inspect wide-uncertainty parameters before transferring geometry to "
                "linear slip inversion."
            )
        if process_status == "REVIEW":
            return (
                "Inspect fault_parameter_trends.png and the text report. Early SMC "
                "process warnings are usually not fatal when parameter trends are stable."
            )
        return (
            "Inspect fault_parameter_trends.png. Proceed if the geometry is "
            "geologically reasonable."
        )

    @staticmethod
    def _wrap_screen_text(text, *, indent="", width=76):
        import textwrap

        return "\n".join(
            textwrap.wrap(
                str(text),
                width=width,
                initial_indent=indent,
                subsequent_indent=indent,
            )
        )

    @classmethod
    def _metric_parameter_label(cls, summary):
        row = summary.get("row") or {}
        label = summary.get("label") or "-"
        direction = cls._metric_direction(row, summary.get("metric"))
        if direction:
            return f"{label} {direction}"
        return label

    @classmethod
    def _metric_direction(cls, row, metric):
        if metric == "bound_distance":
            return row.get("bound_direction")
        if metric == "boundary_mass":
            return row.get("boundary_mass_direction")
        return None

    @classmethod
    def _format_metric_row_list(cls, rows, metric, max_items=6):
        items = []
        for row in rows[:max_items]:
            label = row.get("label", row.get("name", "-"))
            direction = cls._metric_direction(row, metric)
            if direction:
                label = f"{label} {direction}"
            items.append(f"{label}={cls._format_metric(row.get(metric))}")
        if len(rows) > max_items:
            items.append(f"+{len(rows) - max_items} more")
        return ", ".join(items)

    @staticmethod
    def _fault_check_summary_note(checks, key):
        if not checks:
            return "not available"
        summary = checks.get("summaries", {}).get(key)
        if not summary:
            return "not available"
        return summary.get("note", "not available")

    @classmethod
    def _print_fault_parameter_check_table(cls, report):
        rows = report.get("fault_parameter_checks", {}).get("rows", [])
        if not rows:
            return
        print("")
        print("Fault Parameter Detail")
        print("-" * 60)
        print(
            f"{'Parameter':<14} {'final/max':>9} {'final/prior':>11} "
            f"{'medTrend':>9} {'CIwide':>8} {'CInarrow':>8} {'boundDist':>9} "
            f"{'bound':>6} {'bMass':>7} {'mBound':>6} {'medPos':>7} {'flags'}"
        )
        for row in rows:
            print(
                f"{row.get('label', row.get('name')):<14} "
                f"{cls._format_metric(row.get('final_ci_to_max_ci')):>9} "
                f"{cls._format_metric(row.get('final_ci_to_prior')):>11} "
                f"{cls._format_metric(row.get('median_trend_ratio')):>9} "
                f"{cls._format_metric(row.get('ci_widening_ratio')):>8} "
                f"{cls._format_metric(row.get('ci_narrowing_ratio')):>8} "
                f"{cls._format_metric(row.get('bound_distance')):>9} "
                f"{str(row.get('bound_direction') or '-'):>6} "
                f"{cls._format_metric(row.get('boundary_mass')):>7} "
                f"{str(row.get('boundary_mass_direction') or '-'):>6} "
                f"{cls._format_metric(row.get('median_prior_position')):>7} "
                f"{','.join(row.get('flags') or ['-'])}"
            )

    @staticmethod
    def _format_metric(value):
        if value is None:
            return "-"
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(value):
            return "-"
        return f"{value:.3f}"

    @classmethod
    def _write_convergence_text_report(cls, report, filename):
        filename = Path(filename)
        lines = cls._format_convergence_text_report(report)
        filename.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @classmethod
    def _format_convergence_text_report(cls, report):
        checks = report.get("fault_parameter_checks", {})
        thresholds = checks.get("thresholds", {})
        rows = checks.get("rows", [])
        process = report.get("smc_process", {})
        completed = report.get("completed", {})
        samples = report.get("samples", {})
        postval = report.get("postval", {})
        warnings = report.get("warnings", [])
        median = cls._fault_metric_summary(
            checks,
            "median_trend_ratio",
            threshold_key="median_trend_ratio_max",
            warning_threshold_key="median_trend_ratio_warning",
            direction="above",
        )
        final_prior = cls._fault_metric_summary(
            checks,
            "final_ci_to_prior",
            threshold_key="final_ci_to_prior_max",
            direction="above",
        )
        final_max = cls._fault_metric_summary(
            checks,
            "final_ci_to_max_ci",
            threshold_key="final_ci_to_max_ci_max",
            direction="above",
        )
        ci_widening = cls._fault_metric_summary(
            checks,
            "ci_widening_ratio",
            threshold_key="ci_widening_ratio_max",
            direction="above",
        )
        ci_narrowing = cls._fault_metric_summary(
            checks,
            "ci_narrowing_ratio",
            threshold_key="ci_narrowing_ratio_display_min",
            direction="above",
        )
        bound_edge = cls._fault_metric_summary(
            checks,
            "bound_distance",
            threshold_key="bound_distance_min",
            direction="below",
        )
        boundary_mass = cls._fault_metric_summary(
            checks,
            "boundary_mass",
            threshold_key="boundary_mass_max",
            direction="above",
        )
        median_status = cls._section_status(median)
        uncertainty_status = cls._multi_metric_status([final_prior, final_max])
        bound_status = cls._bound_proximity_status(bound_edge, boundary_mass)
        process_status = cls._process_status_label(process)

        lines = [
            "SMC Convergence Diagnostics Report",
            "=" * 80,
            f"Status: {report.get('status')}",
            (
                "Run completed: "
                f"{completed.get('beta_reached_1')} "
                f"(beta={completed.get('beta_final')}, "
                f"finite postval={cls._format_metric(postval.get('finite_ratio'))})"
            ),
            (
                "Particles: "
                f"{samples.get('particles')} / recommended >= "
                f"{samples.get('min_particles')}"
            ),
            "",
            "1. Quick Verdict",
            "-" * 80,
            cls._run_completion_sentence(completed),
            cls._quick_verdict_sentence(
                median_status,
                uncertainty_status,
                bound_status,
            ),
            "Recommended next step:",
            f"  {cls._recommended_convergence_action(median_status, uncertainty_status, bound_status)}",
            "",
            "2. What To Check First",
            "-" * 80,
        ]
        lines.extend(
            cls._format_text_table(
                ["Check", "Status", "Main result", "Meaning"],
                [
                    [
                        "Median stability",
                        median_status,
                        cls._metric_result_text(
                            median,
                            value_suffix=" x final CI width",
                        ),
                        "Are final-stage medians still drifting?",
                    ],
                    [
                        "Bound proximity",
                        bound_status,
                        cls._bound_result_text(
                            bound_edge,
                            boundary_mass,
                            checks,
                        ),
                        "Is posterior close to configured prior bounds?",
                    ],
                    [
                        "Uncertainty",
                        uncertainty_status,
                        cls._metric_result_text(final_prior),
                        "Is final CI still broad relative to prior range?",
                    ],
                    [
                        "Late CI behavior",
                        cls._late_ci_status(ci_widening, ci_narrowing),
                        cls._metric_result_text(ci_widening),
                        "Did CI widen or narrow near final beta?",
                    ],
                    [
                        "SMC process",
                        process_status,
                        process.get("note", "not available"),
                        "Were particle degeneracy signals observed?",
                    ],
                ],
            )
        )
        lines.extend([
            "",
            "3. Flagged Parameters",
            "-" * 80,
        ])
        lines.extend(cls._format_flagged_parameter_text_table(rows, thresholds))
        lines.extend([
            "",
            "4. Key Metric Definitions",
            "-" * 80,
            (
                "late_median_drift_ratio = median_trend_ratio = "
                "posterior-near stage median max shift / final CI width. "
                f"OK <= {cls._format_metric(thresholds.get('median_trend_ratio_max'))}; "
                f"REVIEW > {cls._format_metric(thresholds.get('median_trend_ratio_max'))}; "
                f"WARNING > {cls._format_metric(thresholds.get('median_trend_ratio_warning'))}."
            ),
            (
                "closest_CI_edge = min(final_CI_lower - prior_lower, "
                "prior_upper - final_CI_upper) / prior_width. "
                f"INFO < {cls._format_metric(thresholds.get('bound_distance_min'))}."
            ),
            (
                "boundary_mass = fraction of final posterior samples within "
                f"{cls._boundary_tol_percent(checks)}% prior width of a bound. "
                f"REVIEW >= {cls._format_metric(thresholds.get('boundary_mass_max'))}."
            ),
            (
                "final_ci_to_prior = final CI width / prior width. "
                f"REVIEW > {cls._format_metric(thresholds.get('final_ci_to_prior_max'))}."
            ),
            "",
            "5. Detailed Fault Parameter Table",
            "-" * 80,
        ])
        if rows:
            lines.extend(cls._format_fault_parameter_text_table(rows))
        else:
            lines.append("Fault parameter table: not available")
        lines.extend([
            "",
            "6. Interpretation Notes",
            "-" * 80,
            "- REVIEW does not mean the inversion failed; it means the result needs inspection.",
            (
                "- Bound proximity REVIEW means final posterior samples "
                "accumulate near a configured prior bound."
            ),
            (
                "- Closest CI edge below the threshold is INFO when boundary "
                "mass is below threshold; it is not treated as non-convergence."
            ),
            (
                "- If a REVIEW bound is only a search limit, expand that bound "
                "and rerun to test sensitivity."
            ),
            (
                "- If a flagged bound is a physical hard limit, keep it and "
                "state that constraint when interpreting the model."
            ),
            "- Use fault_parameter_trends.png to inspect the parameter evolution.",
            "",
            "Warnings:",
            ", ".join(warnings) if warnings else "none",
        ])
        return lines

    @classmethod
    def _format_fault_parameter_text_table(cls, rows):
        table_rows = []
        for row in rows:
            flags = ", ".join(row.get("flags") or ["-"])
            table_rows.append([
                row.get("label", row.get("name")),
                cls._format_metric(row.get("median_trend_ratio")),
                cls._format_metric(row.get("final_ci_to_prior")),
                cls._format_value_with_direction(
                    row,
                    "bound_distance",
                    "bound_direction",
                ),
                cls._format_value_with_direction(
                    row,
                    "boundary_mass",
                    "boundary_mass_direction",
                ),
                cls._format_metric(row.get("median_prior_position")),
                flags,
            ])
        return cls._format_text_table(
            [
                "Parameter",
                "Drift",
                "Final/Prior CI",
                "Closest CI Edge",
                "Boundary Mass",
                "Median Pos",
                "Flags",
            ],
            table_rows,
        )

    @staticmethod
    def _run_completion_sentence(completed):
        if completed.get("beta_reached_1"):
            return "Run completed successfully: final beta reached 1.0."
        return "Run did not reach beta=1.0; inspect the sampler output before using the model."

    @staticmethod
    def _quick_verdict_sentence(median_status, uncertainty_status, bound_status):
        parts = []
        if median_status == "OK":
            parts.append("geometry medians are stable near final beta")
        elif median_status == "not available":
            parts.append("geometry median trend diagnostics are unavailable")
        else:
            parts.append("some geometry medians still move near final beta")
        if uncertainty_status == "not available":
            parts.append("uncertainty diagnostics are unavailable")
        elif uncertainty_status != "OK":
            parts.append("some final credible intervals remain broad")
        if bound_status == "not available":
            parts.append("prior-bound diagnostics are unavailable")
        elif bound_status == "REVIEW":
            parts.append("some posterior mass is close to prior bounds")
        elif bound_status == "INFO":
            parts.append("a CI edge is relatively close to a prior bound")
        if not parts:
            return "Main conclusion: no foreground diagnostic issue was detected."
        return "Main conclusion: " + "; ".join(parts) + "."

    @staticmethod
    def _recommended_convergence_action(median_status, uncertainty_status, bound_status):
        if median_status == "not available":
            return "Use a sample file with fault_parameter_stage_summary for full trend diagnostics."
        if median_status == "WARNING":
            return "Increase particles or mutation length, and compare with an independent run."
        if median_status == "REVIEW":
            return "Inspect fault_parameter_trends.png and compare with an independent seed if needed."
        if bound_status == "REVIEW":
            return "Inspect the trend plot; expand flagged bounds and rerun if they are not physical hard limits."
        if bound_status == "INFO":
            return "Inspect the trend plot; boundary expansion is optional unless the edge proximity is scientifically concerning."
        if uncertainty_status != "OK":
            return "Inspect wide-uncertainty parameters before transferring geometry to linear slip inversion."
        return "Inspect the trend plot and proceed if the geometry is geologically reasonable."

    @classmethod
    def _metric_result_text(cls, summary, value_suffix=""):
        if not summary.get("available"):
            return "not available"
        parameter = cls._metric_parameter_label(summary)
        return (
            f"{parameter} = {cls._format_metric(summary.get('value'))}"
            f"{value_suffix}"
        )

    @classmethod
    def _bound_result_text(cls, bound_edge, boundary_mass, checks):
        if boundary_mass.get("exceeded"):
            return cls._metric_result_text(
                boundary_mass,
                value_suffix=(
                    f" within {cls._boundary_tol_percent(checks)}% prior width"
                ),
            )
        if bound_edge.get("exceeded") or bound_edge.get("available"):
            return cls._metric_result_text(bound_edge, value_suffix=" / prior width")
        if boundary_mass.get("available"):
            return cls._metric_result_text(
                boundary_mass,
                value_suffix=(
                    f" within {cls._boundary_tol_percent(checks)}% prior width"
                ),
            )
        return "not available"

    @classmethod
    def _format_flagged_parameter_text_table(cls, rows, thresholds):
        flagged_rows = []
        for row in rows:
            label = row.get("label", row.get("name"))
            for flag in row.get("flags") or []:
                issue = cls._flag_issue_name(flag)
                if issue is None:
                    continue
                value = cls._flag_issue_value(row, flag)
                threshold = cls._flag_issue_threshold(thresholds, flag)
                flagged_rows.append([
                    label,
                    issue,
                    value,
                    threshold,
                    cls._flag_issue_suggestion(flag),
                ])
        if not flagged_rows:
            return ["none"]
        return cls._format_text_table(
            ["Parameter", "Issue", "Value", "Threshold", "Suggested check"],
            flagged_rows,
        )

    @classmethod
    def _flag_issue_name(cls, flag):
        names = {
            "median_trend": "late median drift",
            "ci_width_widening": "late CI widening",
            "ci_not_contracted": "weak CI shrinkage",
            "broad_ci": "broad final CI",
            "ci_touches_bound": "CI near prior bound",
            "boundary_mass": "posterior mass near bound",
        }
        return names.get(flag)

    @classmethod
    def _flag_issue_value(cls, row, flag):
        if flag == "median_trend":
            return cls._format_metric(row.get("median_trend_ratio"))
        if flag == "ci_width_widening":
            return cls._format_metric(row.get("ci_widening_ratio"))
        if flag == "ci_not_contracted":
            return cls._format_metric(row.get("final_ci_to_max_ci"))
        if flag == "broad_ci":
            return cls._format_metric(row.get("final_ci_to_prior"))
        if flag == "ci_touches_bound":
            return cls._format_value_with_direction(
                row,
                "bound_distance",
                "bound_direction",
            )
        if flag == "boundary_mass":
            return cls._format_value_with_direction(
                row,
                "boundary_mass",
                "boundary_mass_direction",
            )
        return "-"

    @classmethod
    def _flag_issue_threshold(cls, thresholds, flag):
        if flag == "median_trend":
            return (
                f"REVIEW > {cls._format_metric(thresholds.get('median_trend_ratio_max'))}; "
                f"WARNING > {cls._format_metric(thresholds.get('median_trend_ratio_warning'))}"
            )
        if flag == "ci_width_widening":
            return f"> {cls._format_metric(thresholds.get('ci_widening_ratio_max'))}"
        if flag == "ci_not_contracted":
            return f"> {cls._format_metric(thresholds.get('final_ci_to_max_ci_max'))}"
        if flag == "broad_ci":
            return f"> {cls._format_metric(thresholds.get('final_ci_to_prior_max'))}"
        if flag == "ci_touches_bound":
            return f"< {cls._format_metric(thresholds.get('bound_distance_min'))}"
        if flag == "boundary_mass":
            return f">= {cls._format_metric(thresholds.get('boundary_mass_max'))}"
        return "-"

    @staticmethod
    def _flag_issue_suggestion(flag):
        suggestions = {
            "median_trend": "inspect trend plot; consider more particles or another seed",
            "ci_width_widening": "inspect trend plot",
            "ci_not_contracted": "check whether data constrain this parameter",
            "broad_ci": "treat transferred geometry as uncertain",
            "ci_touches_bound": "expand bound if it is only a search limit",
            "boundary_mass": "expand bound and rerun if not physically fixed",
        }
        return suggestions.get(flag, "inspect")

    @classmethod
    def _format_value_with_direction(cls, row, metric, direction_key):
        value = cls._format_metric(row.get(metric))
        direction = row.get(direction_key)
        if direction:
            return f"{value} {direction}"
        return value

    @staticmethod
    def _format_text_table(headers, rows):
        if not rows:
            return []
        string_rows = [[str(item) for item in row] for row in rows]
        widths = [
            max(len(str(headers[i])), *(len(row[i]) for row in string_rows))
            for i in range(len(headers))
        ]
        header = "  ".join(
            str(headers[i]).ljust(widths[i]) for i in range(len(headers))
        )
        rule = "  ".join("-" * width for width in widths)
        lines = [header, rule]
        for row in string_rows:
            lines.append(
                "  ".join(row[i].ljust(widths[i]) for i in range(len(headers)))
            )
        return lines

    @staticmethod
    def _parameter_check_summary(report, max_items=4):
        trend_items = report.get("fault_parameter_trend", {}).get(
            "flagged_parameters",
            [],
        )
        display_by_name = {
            item.get("name"): item.get("label")
            for item in trend_items
            if item.get("name") and item.get("label")
        }
        checks = {}
        for name, item in report.get("parameters", {}).items():
            if item.get("status") == "WARNING":
                direction = item.get("boundary_direction")
                label = display_by_name.get(name, name)
                if direction:
                    label += f" ({direction} boundary)"
                checks[name] = label
        for item in trend_items:
            name = item.get("name")
            label = item.get("label") or name
            reasons = item.get("reasons") or []
            if name and label and reasons and "boundary_limited" in reasons:
                checks.setdefault(name, f"{label} (boundary)")
        if not checks:
            return "none"
        unique = list(checks.values())
        text = ", ".join(unique[:max_items])
        if len(unique) > max_items:
            text += f", +{len(unique) - max_items} more"
        return text

    def returnModel(self, model="median", print_stats=True):
        theta = self._select_model_vector(model)
        self.model = theta
        faults = self._faults_from_theta(theta, build_geometry=model not in {"STD", "std", "Std"})

        self.model_dict = getattr(self, "model_dict", {})
        self.model_dict[model] = {
            "faults": {
                fault_name: self.build_fault_params(theta, fault_name)
                for fault_name in self.faultnames
            },
            "data_corrections": self._data_correction_values(theta),
            "sigmas": self._sigma_values(theta),
        }

        if model not in {"STD", "std", "Std"} and hasattr(self, "Likelihoods"):
            dataFaults = self._resolved_data_faults(self.dataFaults)
            for i, entry in enumerate(self.Likelihoods):
                self.Predict(
                    theta,
                    entry.data,
                    vertical=entry.vertical,
                    faultnames=dataFaults[i],
                    updatepatch=True,
                )
            if print_stats:
                self.calculate_and_print_fit_statistics(model=model)
                self.print_mcmc_parameter_positions(print_table=True)
        return faults

    def _select_model_vector(self, model):
        if not hasattr(self, "sampler"):
            raise ValueError("No sampler results available")
        samples = np.asarray(self.sampler["allsamples"])
        if model in {"Mean", "mean"}:
            return samples.mean(axis=0)
        if model in {"Median", "median"}:
            return np.median(samples, axis=0)
        if model in {"Std", "std", "STD"}:
            return samples.std(axis=0)
        if model in {"MAP", "map", "Map"}:
            key = "postval" if "postval" in self.sampler else "log_posterior"
            idx = int(np.argmax(self.sampler[key]))
            return samples[idx, :]
        if isinstance(model, int):
            return samples[model, :]
        raise ValueError(f"Unknown model type: {model}")

    def _faults_from_theta(self, theta, *, build_geometry=True):
        faults = []
        for fault_name in self.faultnames:
            fault = self.faults[fault_name]
            params = self.build_fault_params(theta, fault_name)
            strike = params["strike"]
            dip = params["dip"]
            if dip > 90:
                dip = 180 - dip
                strike = (strike + 180) % 360
            elif dip < 0:
                dip = -dip
                strike = (strike + 180) % 360
            if build_geometry:
                fault.buildPatches(
                    params["lon"],
                    params["lat"],
                    params["depth"],
                    strike,
                    dip,
                    params["length"],
                    params["width"],
                    1,
                    1,
                    verbose=False,
                )
            else:
                fault.slip = np.zeros((1, 2), dtype=float)
            strikeslip, dipslip = self._slip_components(params)
            fault.slip[:, 0] = strikeslip
            fault.slip[:, 1] = dipslip
            faults.append(fault)
        return faults

    def _data_correction_values(self, theta):
        values = {}
        for spec in self.data_correction_specs:
            coeffs = correction_coefficients_from_theta(spec, theta)
            values[spec.data_name] = {
                name: float(coeffs[i]) for i, name in enumerate(spec.parameter_names)
            }
        return values

    def _sigma_values(self, theta):
        dataset_sigmas = self._dataset_sigmas_from_theta(theta)
        return {
            data.name: float(dataset_sigmas[i])
            for i, data in enumerate(self.geodata.get("data", []))
        }

    def calculate_data_fit_metrics(self, data, vertical=True):
        observed = self._observation_vector(data, vertical)
        synthetic = self._synthetic_vector(data, vertical)
        residuals = synthetic - observed
        rms = float(np.sqrt(np.mean(residuals**2)))
        ss_res = float(np.sum(residuals**2))
        ss_tot = float(np.sum(observed**2))
        vr = (1.0 - ss_res / ss_tot) * 100.0 if ss_tot != 0 else 0.0
        return rms, vr

    def calculate_and_print_fit_statistics(self, model="median"):
        if not hasattr(self, "model_dict") or model not in self.model_dict:
            self.returnModel(model=model, print_stats=False)

        print("\n" + "=" * 60)
        print(f"Data Fit Statistics ({str(model).upper()} model)")
        print("=" * 60)
        total_rms = 0.0
        total_vr = 0.0
        count = 0
        for data, vertical in zip(self.datas, self.verticals):
            rms, vr = self.calculate_data_fit_metrics(data, vertical)
            print(f"{data.name:<15} | RMS: {rms:8.4f} | VR: {vr:6.2f}%")
            total_rms += rms
            total_vr += vr
            count += 1
        if count:
            print("-" * 60)
            print(f"{'Average':<15} | RMS: {total_rms / count:8.4f} | VR: {total_vr / count:6.2f}%")
        print("=" * 60)

    def save_model_to_file(
        self,
        filename=None,
        model="median",
        recalculate=False,
        output_to_screen=True,
        screen_format="compact",
    ):
        """Write a readable model summary for the selected posterior model."""
        if filename is None:
            filename = f"model_results_{model}.txt"
        if recalculate or not hasattr(self, "model_dict") or model not in self.model_dict:
            self.returnModel(model=model, print_stats=False)

        model_entry = self.model_dict[model]
        lines = self._format_model_summary(model, model_entry)
        text = "\n".join(lines) + "\n"
        Path(filename).write_text(text, encoding="utf-8")
        if output_to_screen:
            if screen_format in {"full", "detailed", "detail"}:
                print(text)
            else:
                print(self._format_compact_model_summary(model, model_entry, report_file=filename))
        return text

    def _format_model_summary(self, model, model_entry):
        theta = self._summary_model_vector(model)
        std_by_index = self._summary_std_by_index()
        parameter_specs = list(getattr(self, "parameter_specs", []))
        lines = [
            f"Nonlinear Geometry Model Summary ({str(model).upper()})",
            "=" * 60,
            "",
        ]
        lines.extend(
            self._format_fault_model_summary(
                model_entry.get("faults", {}),
                std_by_index=std_by_index,
            )
        )
        lines.extend(
            self._format_data_correction_model_summary(
                model_entry.get("data_corrections", {}),
                std_by_index=std_by_index,
            )
        )
        lines.extend(
            self._format_sigma_model_summary(
                model_entry.get("sigmas", {}),
                std_by_index=std_by_index,
                theta=theta,
            )
        )
        lines.extend(
            self._format_sample_vector_summary(
                theta,
                parameter_specs=parameter_specs,
            )
        )
        return lines

    def _format_compact_model_summary(self, model, model_entry, *, report_file=None):
        theta = self._summary_model_vector(model)
        std_by_index = self._summary_std_by_index()
        rows = self._compact_model_summary_rows(model_entry, theta=theta, std_by_index=std_by_index)
        lines = [
            "",
            "=" * 60,
            f"SMC Nonlinear Geometry Model Summary ({str(model).upper()})",
            "=" * 60,
        ]
        if report_file is not None:
            lines.append(f"Detailed report       : {report_file}")
        lines.extend([
            "Note                  : '*' marks fixed values; [data] rows are derived values.",
            "",
        ])
        headers = ["Index", "Category", "Name", "Parameter", str(model).upper(), "STD", "Note"]
        lines.extend(self._format_compact_table(headers, rows))
        lines.append("")
        lines.append(f"Total sampled parameters: {len(getattr(self, 'parameter_specs', []) or [])}")
        lines.extend([
            "",
            f"Sample vector values ({str(model).upper()}):",
            self._format_value_list(theta) if theta is not None else "not available",
            "=" * 60,
        ])
        return "\n".join(lines)

    def _compact_model_summary_rows(self, model_entry, *, theta, std_by_index):
        rows = []
        rows.extend(self._compact_fault_rows(model_entry.get("faults", {}), std_by_index))
        rows.extend(self._compact_data_correction_rows(model_entry.get("data_corrections", {}), std_by_index))
        rows.extend(self._compact_sigma_rows(model_entry.get("sigmas", {}), theta, std_by_index))
        return rows

    def _compact_fault_rows(self, faults, std_by_index):
        rows = []
        fault_spec_lookup = self._fault_spec_lookup()
        fixed_params = getattr(self, "fixed_params", {}) or {}
        model_name = ""
        for fault_name in self._ordered_fault_names(faults):
            params = faults[fault_name]
            alias = self._fault_alias(fault_name)
            model_name = f"{fault_name} ({alias})"
            fixed_fault_names = self._fixed_fault_param_names(fault_name, fixed_params)
            for local_name in self._fault_summary_order(params):
                if local_name not in params:
                    continue
                spec = fault_spec_lookup.get((fault_name, local_name))
                fixed = spec is None and local_name in fixed_fault_names
                rows.append(self._compact_summary_row(
                    index=self._compact_index(spec, fixed=fixed),
                    group="Fault",
                    name=model_name,
                    parameter=local_name,
                    value=params[local_name],
                    std=self._compact_std(spec, std_by_index, fixed=fixed),
                    note="fixed *" if fixed else "",
                ))
        return rows

    def _compact_data_correction_rows(self, data_corrections, std_by_index):
        rows = []
        lookup = self._data_correction_spec_lookup()
        fixed_lookup = self._fixed_data_correction_lookup()
        for data_name, params in data_corrections.items():
            for local_name, value in params.items():
                spec = lookup.get((data_name, local_name))
                fixed = spec is None and (data_name, local_name) in fixed_lookup
                rows.append(self._compact_summary_row(
                    index=self._compact_index(spec, fixed=fixed),
                    group="Data correction",
                    name=data_name,
                    parameter=local_name,
                    value=value,
                    std=self._compact_std(spec, std_by_index, fixed=fixed),
                    note="fixed *" if fixed else "",
                ))
        return rows

    def _compact_sigma_rows(self, sigmas, theta, std_by_index):
        rows = []
        sampled_note = "sampled log10" if self._sigma_log_scaled() else "sampled"
        for row in self._sigma_sample_summary_rows(theta, std_by_index):
            fixed = row["fixed"]
            note = "fixed *" if fixed else sampled_note
            rows.append(self._compact_summary_row(
                index=self._compact_index_from_row(row),
                group="Sigma",
                name=row["name"],
                parameter="sigma",
                value=row["value"],
                std=row["std"] if row["std"] is not None else (0.0 if fixed else None),
                note=note,
            ))
        if self._sigma_log_scaled():
            physical_std = self._physical_sigma_std_by_dataset()
            for data_name, value in sigmas.items():
                rows.append(self._compact_summary_row(
                    index="[data]",
                    group="Sigma physical",
                    name=data_name,
                    parameter="sigma",
                    value=value,
                    std=physical_std.get(data_name),
                    note="10**sampled",
                ))
        return rows

    def _compact_summary_row(self, *, index, group, name, parameter, value, std, note):
        return [
            str(index),
            str(group),
            str(name),
            str(parameter),
            self._format_model_float(value),
            self._format_model_float(std) if std is not None else "-",
            str(note),
        ]

    @staticmethod
    def _compact_index(spec, *, fixed=False):
        if fixed:
            return "[fixed]"
        if spec is None:
            return "[-]"
        return f"[{spec.index}]"

    @staticmethod
    def _compact_index_from_row(row):
        if row["fixed"]:
            return "[fixed]"
        if row["index"] is None:
            return "[-]"
        return f"[{row['index']}]"

    @staticmethod
    def _compact_std(spec, std_by_index, *, fixed=False):
        if fixed:
            return 0.0
        if spec is None:
            return None
        return std_by_index.get(spec.index)

    @staticmethod
    def _format_compact_table(headers, rows):
        rows = [list(row) for row in rows]
        try:
            from tabulate import tabulate

            return tabulate(
                rows,
                headers=headers,
                tablefmt="grid",
                stralign="left",
            ).splitlines()
        except Exception:
            pass

        widths = [
            max(len(str(headers[i])), *(len(str(row[i])) for row in rows))
            for i in range(len(headers))
        ]

        def fmt_row(values):
            return " | ".join(str(values[i]).ljust(widths[i]) for i in range(len(headers)))

        lines = [fmt_row(headers), "-+-".join("-" * width for width in widths)]
        lines.extend(fmt_row(row) for row in rows)
        return lines

    def _summary_model_vector(self, model):
        if hasattr(self, "sampler") and "allsamples" in self.sampler:
            try:
                return np.asarray(self._select_model_vector(model), dtype=float)
            except Exception:
                return None
        if hasattr(self, "model"):
            try:
                return np.asarray(self.model, dtype=float)
            except Exception:
                return None
        return None

    def _summary_std_by_index(self):
        if not hasattr(self, "sampler") or "allsamples" not in self.sampler:
            return {}
        samples = np.asarray(self.sampler["allsamples"], dtype=float)
        if samples.ndim != 2:
            return {}
        std = np.std(samples, axis=0)
        return {idx: float(value) for idx, value in enumerate(std)}

    def _format_fault_model_summary(self, faults, *, std_by_index):
        if not faults:
            return []
        lines = [
            "Fault parameters",
            "=" * 60,
            "",
        ]
        fault_spec_lookup = self._fault_spec_lookup()
        fixed_params = getattr(self, "fixed_params", {}) or {}
        for fault_name in self._ordered_fault_names(faults):
            params = faults[fault_name]
            alias = self._fault_alias(fault_name)
            fixed_fault_names = self._fixed_fault_param_names(fault_name, fixed_params)
            lines.extend([
                f"Fault: {fault_name} ({alias})",
                "-" * 30,
            ])
            order = self._fault_summary_order(params)
            rows = []
            for local_name in order:
                if local_name not in params:
                    continue
                spec = fault_spec_lookup.get((fault_name, local_name))
                row = self._parameter_summary_row(
                    local_name,
                    params[local_name],
                    spec=spec,
                    std_by_index=std_by_index,
                    fixed=spec is None and local_name in fixed_fault_names,
                )
                rows.append(row)

            if rows:
                lines.append("Parameters in fault order:")
                lines.extend(self._format_parameter_rows(rows))

            list_names = [row["name"] for row in rows]
            list_indices = [self._format_ordered_index(row) for row in rows]
            list_values = [row["value"] for row in rows]
            lines.extend([
                "",
                "Ordered list:",
                f"  names : {self._format_name_list(list_names)}",
                f"  index : {self._format_name_list(list_indices)}",
                f"  values: {self._format_value_list(list_values)}",
                "",
            ])
        return lines

    def _format_data_correction_model_summary(self, data_corrections, *, std_by_index):
        if not data_corrections:
            return []
        lines = [
            "Data correction parameters",
            "=" * 60,
        ]
        lookup = self._data_correction_spec_lookup()
        fixed_lookup = self._fixed_data_correction_lookup()
        for data_name, params in data_corrections.items():
            lines.append(f"Dataset: {data_name}")
            rows = []
            for local_name, value in params.items():
                spec = lookup.get((data_name, local_name))
                rows.append(
                    self._parameter_summary_row(
                        local_name,
                        value,
                        spec=spec,
                        std_by_index=std_by_index,
                        fixed=spec is None and (data_name, local_name) in fixed_lookup,
                    )
                )
            lines.extend(self._format_parameter_rows(rows))
            lines.append("")
        return lines

    def _format_sigma_model_summary(self, sigmas, *, std_by_index, theta=None):
        if not sigmas:
            return []
        lines = [
            "Sigma parameters",
            "=" * 60,
        ]
        sample_rows = self._sigma_sample_summary_rows(theta, std_by_index)
        if sample_rows:
            lines.extend(self._format_parameter_rows(sample_rows))
        else:
            # Fallback for partially constructed test/user objects without a parsed
            # geodata.sigmas config. New nonlinear objects should use the branch above.
            lookup = self._sigma_spec_lookup()
            for data_name, value in sigmas.items():
                spec = lookup.get(data_name)
                fixed = spec is None
                row = self._parameter_summary_row(
                    data_name,
                    value,
                    spec=spec,
                    std_by_index=std_by_index,
                    fixed=fixed,
                )
                lines.extend(self._format_parameter_rows([row]))
        if self._sigma_log_scaled():
            lines.extend(["", "Physical sigma values used in likelihood (10**sampled sigma)"])
            physical_std = self._physical_sigma_std_by_dataset()
            for data_name, value in sigmas.items():
                std = physical_std.get(data_name)
                lines.append(
                    f"  {data_name:<12} {'[data]':<8}: "
                    f"{self._format_model_float(value):>12} {self._format_uncertainty(std)}"
                )
        lines.append("")
        return lines

    def _sigma_sample_summary_rows(self, theta, std_by_index):
        sigmas = getattr(self, "sigmas", None)
        if not isinstance(sigmas, Mapping) or "values" not in sigmas:
            return []
        values = np.asarray(sigmas.get("values", []), dtype=float)
        if values.size == 0:
            return []
        names = list(self._sigma_group_names())
        if len(names) < values.size:
            names.extend(f"sigma_{i}" for i in range(len(names), values.size))
        spec_by_param_index = {}
        for spec in getattr(self, "sigma_parameter_specs", []):
            metadata = getattr(spec, "metadata", {}) or {}
            sigma_param_index = metadata.get("sigma_param_index")
            if sigma_param_index is None:
                try:
                    sigma_param_index = names.index(spec.local_name)
                except (AttributeError, ValueError):
                    continue
            if sigma_param_index is not None:
                spec_by_param_index[int(sigma_param_index)] = spec
        theta = None if theta is None else np.asarray(theta, dtype=float)
        rows = []
        for sigma_param_index, initial_value in enumerate(values):
            spec = spec_by_param_index.get(sigma_param_index)
            if spec is not None and theta is not None and spec.index < theta.size:
                value = theta[spec.index]
            else:
                value = initial_value
            rows.append(
                self._parameter_summary_row(
                    names[sigma_param_index],
                    value,
                    spec=spec,
                    std_by_index=std_by_index,
                    fixed=spec is None,
                )
            )
        return rows

    def _physical_sigma_std_by_dataset(self):
        if not self._sigma_log_scaled():
            return {}
        if not hasattr(self, "sampler") or "allsamples" not in self.sampler:
            return {}
        data_list = getattr(self, "geodata", {}).get("data", [])
        if not data_list:
            return {}
        try:
            samples = np.asarray(self.sampler["allsamples"], dtype=float)
        except (TypeError, ValueError):
            return {}
        if samples.ndim != 2 or samples.shape[0] == 0:
            return {}
        values = []
        for sample in samples:
            try:
                values.append(self._dataset_sigmas_from_theta(sample))
            except Exception:
                return {}
        values = np.asarray(values, dtype=float)
        if values.ndim != 2:
            return {}
        std = np.std(values, axis=0)
        return {
            data.name: float(std[i])
            for i, data in enumerate(data_list)
            if i < std.size
        }

    def _sigma_log_scaled(self):
        sigmas = getattr(self, "sigmas", None)
        return isinstance(sigmas, Mapping) and bool(sigmas.get("log_scaled", False))

    def _format_sample_vector_summary(self, theta, *, parameter_specs):
        lines = [
            "Sample vector in ParameterSpec order",
            "=" * 60,
        ]
        if theta is None or not parameter_specs:
            lines.extend(["not available", ""])
            return lines
        specs = sorted(parameter_specs, key=lambda spec: spec.index)
        names = [spec.name for spec in specs]
        labels = [spec.label for spec in specs]
        values = [
            theta[spec.index] if spec.index < theta.size else np.nan
            for spec in specs
        ]
        lines.extend([
            f"canonical names: {self._format_name_list(names)}",
            f"display names  : {self._format_name_list(labels)}",
            f"values         : {self._format_value_list(values)}",
            "",
        ])
        return lines

    def _parameter_summary_row(
        self,
        name,
        value,
        *,
        spec=None,
        std_by_index=None,
        fixed=False,
    ):
        std_by_index = std_by_index or {}
        index = spec.index if spec is not None else None
        return {
            "name": name,
            "index": index,
            "fixed": bool(fixed),
            "value": float(value),
            "std": std_by_index.get(index) if index is not None else None,
        }

    @classmethod
    def _format_parameter_rows(cls, rows):
        lines = []
        for row in rows:
            index = "[fixed]" if row["fixed"] else cls._format_index(row["index"])
            std = cls._format_uncertainty(row["std"])
            fixed_marker = " *" if row["fixed"] else ""
            lines.append(
                f"  {row['name']:<12} {index:<8}: "
                f"{cls._format_model_float(row['value']):>12} {std}{fixed_marker}"
            )
        return lines

    @staticmethod
    def _format_ordered_index(row):
        if row["fixed"]:
            return "fixed"
        if row["index"] is None:
            return "-"
        return str(row["index"])

    @staticmethod
    def _format_index(index):
        if index is None:
            return "[-]"
        return f"[{index}]"

    @classmethod
    def _format_uncertainty(cls, value):
        if value is None:
            return "+/- -"
        return f"+/- {cls._format_model_float(value)}"

    @staticmethod
    def _format_model_float(value):
        try:
            value = float(value)
        except (TypeError, ValueError):
            return "-"
        if not np.isfinite(value):
            return "-"
        return f"{value:.6f}"

    @classmethod
    def _format_value_list(cls, values):
        return "[" + ", ".join(cls._format_model_float(value) for value in values) + "]"

    @staticmethod
    def _format_name_list(names):
        return "[" + ", ".join(str(name) for name in names) + "]"

    def _fault_alias(self, fault_name):
        alias_map = getattr(self, "fault_alias_map", {}) or {}
        return alias_map.get(fault_name, fault_name)

    def _fault_summary_order(self, params):
        order = list(getattr(self, "fault_parameter_order", []) or [])
        if not order:
            order = list(params.keys())
        for key in params:
            if key not in order:
                order.append(key)
        return order

    def _ordered_fault_names(self, faults):
        names = []
        for fault_name in getattr(self, "faultnames", []) or []:
            if fault_name in faults and fault_name not in names:
                names.append(fault_name)
        for fault_name in faults:
            if fault_name not in names:
                names.append(fault_name)
        return names

    @staticmethod
    def _fixed_fault_param_names(fault_name, fixed_params):
        values = fixed_params.get(fault_name, {}) if isinstance(fixed_params, Mapping) else {}
        if isinstance(values, Mapping):
            return set(values.keys())
        try:
            return set(values)
        except TypeError:
            return set()

    def _fault_spec_lookup(self):
        lookup = {}
        for spec in getattr(self, "parameter_specs", []):
            if spec.group == "faults" and spec.fault_name is not None:
                lookup[(spec.fault_name, spec.local_name)] = spec
        return lookup

    def _data_correction_spec_lookup(self):
        lookup = {}
        for spec in getattr(self, "parameter_specs", []):
            if spec.group == "data_corrections" and spec.data_name is not None:
                lookup[(spec.data_name, spec.local_name)] = spec
        return lookup

    def _fixed_data_correction_lookup(self):
        lookup = set()
        for spec in getattr(self, "data_correction_specs", []):
            if getattr(spec, "sampled", False):
                continue
            for local_name in getattr(spec, "parameter_names", []):
                lookup.add((spec.data_name, local_name))
        return lookup

    def _sigma_spec_lookup(self):
        lookup = {}
        for spec in getattr(self, "sigma_parameter_specs", []):
            lookup[spec.local_name] = spec
            lookup[spec.label] = spec
            lookup[spec.name] = spec
        return lookup

    def extract_and_plot_bayesian_results(
        self,
        rank=0,
        filename="samples_mag_rake_multifaults.h5",
        plot_faults=True,
        plot_sigmas=True,
        plot_data=True,
        antisymmetric=True,
        res_use_data_norm=True,
        cmap="jet",
        model="median",
        fault_figsize=(7.5, 6.5),
        sigmas_figsize=(2.625, 2.625),
        save_data=True,
        sar_corner=None,
        plot_data_corrections=True,
        data_corrections_figsize=(3.5, 3.5),
        modeling_dir="Modeling",
        show=True,
        screen_dpi=200,
        diagnose=True,
        diagnose_detail=False,
        convergence_report_file=None,
        force_diagnose=False,
    ):
        """Load samples, plot posterior summaries, and rebuild a selected model.

        This is a compatibility entry for old example scripts.  It keeps the
        old high-level workflow name, while using the new parameter registry
        and plotting helpers internally.
        """
        if rank != 0:
            return None
        if model in {"std", "Std", "STD"}:
            plot_data = False

        self.load_samples_from_h5(filename=filename)
        if diagnose and (
            force_diagnose or not self._convergence_report_already_current(filename)
        ):
            self.report_convergence(
                filename=convergence_report_file,
                sample_filename=filename,
                print_report=True,
                print_detail=diagnose_detail,
            )
        self.print_mcmc_parameter_positions(print_table=False)

        if plot_faults or (plot_sigmas and self.sigma_parameter_specs) or (
            plot_data_corrections and any(self.data_correction_parameter_specs.values())
        ):
            print("\n" + "=" * 60)
            print("Output Figures")
            print("=" * 60)

        if plot_faults:
            self._plot_fault_kde_matrices(
                fault_figsize=fault_figsize,
                show=show,
                screen_dpi=screen_dpi,
            )

        if plot_sigmas and self.sigma_parameter_specs:
            self.plot_kde_matrix(
                save=True,
                plot_faults=False,
                plot_sigmas=True,
                fill=True,
                scatter=False,
                filename="kde_matrix_sigmas.png",
                figsize=sigmas_figsize,
                hspace=0.05,
                wspace=0.05,
                show=show,
                screen_dpi=screen_dpi,
            )

        if plot_data_corrections and any(self.data_correction_parameter_specs.values()):
            self.plot_kde_matrix(
                save=True,
                plot_faults=False,
                plot_sigmas=False,
                plot_data_corrections=True,
                fill=True,
                scatter=False,
                filename="kde_matrix_data_corrections.png",
                figsize=data_corrections_figsize,
                hspace=0.05,
                wspace=0.05,
                xtick_rotation=45,
                show=show,
                screen_dpi=screen_dpi,
            )

        if not hasattr(self, "Likelihoods"):
            self.setLikelihood()

        faults = self.returnModel(model=model, print_stats=False)
        self.save_model_to_file(
            f"model_results_{model}.txt",
            model=model,
            output_to_screen=True,
        )
        if hasattr(self, "datas") and hasattr(self, "verticals"):
            self.calculate_and_print_fit_statistics(model=model)

        grouped_data = self._group_data_by_type()
        if save_data:
            self._save_modeled_data_files(
                grouped_data,
                modeling_dir=modeling_dir,
                sar_corner=sar_corner,
            )

        if plot_data:
            self._plot_modeled_data(
                faults,
                grouped_data,
                modeling_dir=modeling_dir,
                antisymmetric=antisymmetric,
                res_use_data_norm=res_use_data_norm,
                cmap=cmap,
                show=show,
            )
        return faults

    def _plot_fault_kde_matrices(self, *, fault_figsize, show, screen_dpi=200):
        fault_figsize = fault_figsize if fault_figsize is not None else (7.5, 6.5)
        for fault_name in self.faultnames:
            fault_alias = self.fault_alias_map.get(fault_name, fault_name)
            self.plot_kde_matrix(
                save=True,
                plot_faults=True,
                faults=fault_name,
                fill=True,
                scatter=False,
                filename=f"kde_matrix_{fault_alias}.png",
                figsize=fault_figsize,
                hspace=0.05,
                wspace=0.05,
                xtick_rotation=45,
                show=show,
                screen_dpi=screen_dpi,
            )

    def _group_data_by_type(self):
        grouped = {
            "gps": [],
            "insar": [],
            "leveling": [],
            "crossfaultoffset": [],
        }
        for data, vertical in zip(getattr(self, "datas", []), getattr(self, "verticals", [])):
            dtype = str(getattr(data, "dtype", "")).lower()
            if dtype == "gps":
                grouped["gps"].append((data, vertical))
            elif dtype in grouped:
                grouped[dtype].append(data)
        return grouped

    def _save_modeled_data_files(self, grouped_data, *, modeling_dir, sar_corner=None):
        out_dir = Path(modeling_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for sardata in grouped_data["insar"]:
            if sar_corner is not None and hasattr(sardata, "writeDecim2file"):
                triangular = sar_corner == "tri"
                for data_type in ["data", "synth", "resid"]:
                    sardata.writeDecim2file(
                        f"{sardata.name}_{data_type}.txt",
                        data_type,
                        outDir=str(out_dir),
                        triangular=triangular,
                    )
            elif hasattr(sardata, "write2file"):
                for data_type in ["data", "synth", "resid"]:
                    sardata.write2file(
                        f"{sardata.name}_{data_type}.txt",
                        data_type,
                        outDir=str(out_dir),
                    )

        for gpsdata, _vertical in grouped_data["gps"]:
            if hasattr(gpsdata, "write2file"):
                for data_type in ["data", "synth", "res"]:
                    gpsdata.write2file(
                        f"{gpsdata.name}_{data_type}.txt",
                        data_type,
                        outDir=str(out_dir),
                    )

        for levdata in grouped_data["leveling"]:
            if hasattr(levdata, "write2file"):
                for data_type in ["data", "synth"]:
                    levdata.write2file(
                        f"{levdata.name}_{data_type}.txt",
                        outDir=str(out_dir),
                        data=data_type,
                    )

        for cfdata in grouped_data["crossfaultoffset"]:
            if hasattr(cfdata, "write2file"):
                for data_type in ["data", "synth"]:
                    cfdata.write2file(
                        f"{cfdata.name}_{data_type}.txt",
                        outDir=str(out_dir),
                        data=data_type,
                    )

    def _plot_modeled_data(
        self,
        faults,
        grouped_data,
        *,
        modeling_dir,
        antisymmetric,
        res_use_data_norm,
        cmap,
        show,
    ):
        out_dir = Path(modeling_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        for fault in faults:
            fault.color = "b"
        for gpsdata, _vertical in grouped_data["gps"]:
            if not hasattr(gpsdata, "plot"):
                continue
            box = [gpsdata.lon.min(), gpsdata.lon.max(), gpsdata.lat.min(), gpsdata.lat.max()]
            gpsdata.plot(
                faults=faults,
                drawCoastlines=True,
                data=["data", "synth"],
                scale=0.2,
                legendscale=0.05,
                color=["k", "r"],
                seacolor="lightblue",
                box=box,
                titleyoffset=1.02,
            )
            if hasattr(gpsdata, "fig"):
                gpsdata.fig.savefig(
                    f"gps_{gpsdata.name}",
                    ftype="png",
                    dpi=600,
                    bbox_inches="tight",
                    mapaxis=None,
                    saveFig=["map"],
                )

        for fault in faults:
            fault.color = "k"
        for sardata in grouped_data["insar"]:
            if not hasattr(sardata, "plot_fit_comparison"):
                continue
            datamin, datamax = float(sardata.vel.min()), float(sardata.vel.max())
            absmax = max(abs(datamin), abs(datamax))
            norm = [-absmax, absmax] if antisymmetric else [datamin, datamax]
            sardata.plot_fit_comparison(
                faults=faults,
                cmap=cmap,
                vmin=norm[0],
                vmax=norm[1],
                share_colorbar=res_use_data_norm,
                save_path=str(out_dir / f"{sardata.name}_fit_comparison.pdf"),
                show=show,
            )

        try:
            from .data_plot_utils import _plot_crossfaultoffset_fit, _plot_leveling_fit
        except Exception:
            return
        for levdata in grouped_data["leveling"]:
            _plot_leveling_fit(levdata, save_dir=str(out_dir), file_type="png", show=show)
        for cfdata in grouped_data["crossfaultoffset"]:
            _plot_crossfaultoffset_fit(cfdata, save_dir=str(out_dir), file_type="png", show=show)

    def print_mcmc_parameter_positions(self, print_table=True):
        rows = [
            [
                spec.group,
                spec.fault_name or spec.data_name or "",
                spec.local_name,
                spec.name,
                spec.label,
                spec.index,
            ]
            for spec in self.parameter_specs
        ]
        rows.sort(key=lambda row: row[-1])

        estimated = defaultdict(list)
        for spec in self.parameter_specs:
            estimated[spec.group].append(spec.name)

        if print_table:
            headers = ["Group", "Owner", "Param", "Canonical", "Display", "Index"]
            try:
                from tabulate import tabulate

                self.logger.info("\n%s", tabulate(rows, headers=headers, tablefmt="grid"))
            except Exception:
                for row in rows:
                    self.logger.info("%s", row)
        return dict(estimated)

    def data_correction_label_map(self):
        return {
            spec.name: spec.label
            for spec in self.parameter_groups.get("data_corrections", [])
        }

    def plot_kde_matrix(self, **kwargs):
        """Plot a KDE matrix with the nonlinear SMC plotting conventions."""
        from .nonlinear_geometry_plots import plot_kde_matrix

        return plot_kde_matrix(self, **kwargs)

    def plot_parameter_marginals(self, **kwargs):
        """Plot posterior marginal distributions using the standalone plot module."""
        from .nonlinear_geometry_plots import plot_parameter_marginals

        return plot_parameter_marginals(self, **kwargs)

    def plot_parameter_pairs(self, **kwargs):
        """Plot a lightweight corner-style parameter matrix."""
        from .nonlinear_geometry_plots import plot_parameter_pairs

        return plot_parameter_pairs(self, **kwargs)

    def plot_sample_traces(self, **kwargs):
        """Plot posterior samples in saved order for selected parameters."""
        from .nonlinear_geometry_plots import plot_sample_traces

        return plot_sample_traces(self, **kwargs)

    def plot_parameter_intervals(self, **kwargs):
        """Plot posterior medians and credible intervals."""
        from .nonlinear_geometry_plots import plot_parameter_intervals

        return plot_parameter_intervals(self, **kwargs)

    def plot_smc_progress(self, **kwargs):
        """Plot saved SMC beta and posterior values when available."""
        from .nonlinear_geometry_plots import plot_smc_progress

        return plot_smc_progress(self, **kwargs)

    def plot_fault_parameter_trends(self, **kwargs):
        """Plot fault-parameter median and credible interval evolution by SMC stage."""
        from .nonlinear_geometry_plots import plot_fault_parameter_trends

        return plot_fault_parameter_trends(self, **kwargs)

    def save2h5(self, filename, datasets=None):
        if h5py is None:
            self.logger.warning("h5py is not available; samples were not saved")
            return
        if datasets is None:
            datasets = [
                "allsamples",
                "postval",
                "beta",
                "stage",
                "covsmpl",
                "resmpl",
                "sample_stats",
                "fault_parameter_stage_summary",
            ]
        with h5py.File(filename, "w") as f:
            for dataset in datasets:
                if dataset in self.sampler and self.sampler[dataset] is not None:
                    value = self.sampler[dataset]
                    if isinstance(value, Mapping):
                        group = f.create_group(dataset)
                        for key, subvalue in value.items():
                            self._write_h5_dataset(group, key, subvalue)
                    else:
                        self._write_h5_dataset(f, dataset, value)
            f.attrs["parameter_names_json"] = json.dumps(self.parameter_names())
            f.attrs["parameter_display_names_json"] = json.dumps(
                self.parameter_display_names()
            )
            lb, ub = self.parameter_bounds()
            f.create_dataset("lower_bounds", data=lb)
            f.create_dataset("upper_bounds", data=ub)

    def load_samples_from_h5(self, filename, datasets=None):
        if h5py is None:
            self.logger.warning("h5py is not available; samples were not loaded")
            return
        if datasets is None:
            datasets = [
                "allsamples",
                "postval",
                "beta",
                "stage",
                "covsmpl",
                "resmpl",
                "sample_stats",
                "fault_parameter_stage_summary",
            ]
        samples = {}
        with h5py.File(filename, "r") as f:
            for dataset in datasets:
                if dataset in f:
                    item = f[dataset]
                    if isinstance(item, h5py.Group):
                        samples[dataset] = {
                            key: self._read_h5_dataset(item[key])
                            for key in item.keys()
                        }
                    else:
                        samples[dataset] = self._read_h5_dataset(item)
        self.sampler = samples

    @staticmethod
    def _write_h5_dataset(group, key, value):
        arr = np.asarray(value)
        if arr.dtype.kind in {"U", "O"}:
            dtype = h5py.string_dtype(encoding="utf-8")
            group.create_dataset(key, data=np.asarray(value, dtype=dtype), dtype=dtype)
            return
        group.create_dataset(key, data=value)

    @staticmethod
    def _read_h5_dataset(dataset):
        value = dataset[()]
        arr = np.asarray(value)
        if arr.dtype.kind == "S":
            return np.char.decode(arr, "utf-8").tolist()
        if arr.dtype.kind == "O":
            return [
                item.decode("utf-8") if isinstance(item, bytes) else str(item)
                for item in arr.reshape(-1)
            ]
        return value


nonlinear_geometry_smc = NonlinearGeometrySMCInversion
