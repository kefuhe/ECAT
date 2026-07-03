#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Nonlinear-geometry SMC backend with process diagnostics.

This module is a focused copy of the generic :mod:`SMC_MPI` sampling path for
the new nonlinear-geometry interface.  The sampling equations and update order
are kept equivalent to ``SMC_MPI.SMC_samples_parallel_mpi``; the added behavior
is limited to recording per-stage diagnostics in ``sample_stats``.
"""

from __future__ import annotations

from datetime import datetime
import time

import h5py
import numpy as np
from numba import njit


def deterministicR_optimized(inIndex, q):
    """Kitagawa deterministic resampling, copied from the generic backend."""

    n_chains = inIndex.shape[0]
    parents = np.arange(n_chains)

    cum_dist = np.cumsum(q)
    aux = np.random.rand(1)
    u = (parents + aux) / n_chains

    N_childs = np.zeros(n_chains, dtype=np.int64)
    j = np.searchsorted(cum_dist, u)
    np.add.at(N_childs, j, 1)

    outindx = np.repeat(parents, N_childs)

    return outindx


@njit
def multivariate_normal(mean, L, size):
    Z = np.random.standard_normal((size, L.shape[0]))
    return mean + np.dot(Z, L.T)


@njit
def adjust_bounds(X_new, LB, UB, avg_acc):
    adjust = avg_acc >= 0.05
    ind1 = X_new < LB
    X_new[ind1] = np.add(LB[ind1], (LB[ind1] - X_new[ind1]) * adjust)

    ind2 = X_new > UB
    X_new[ind2] = np.subtract(UB[ind2], (X_new[ind2] - UB[ind2]) * adjust)

    return X_new


def run_amh(X, covariance_chol, mrun, beta, LB, UB, target, a=1.0 / 9.0, b=8.0 / 9.0):
    Dims = covariance_chol.shape[0]
    logpdf = target(X)
    P = logpdf * beta
    best_P = P
    P0 = P

    sameind = np.where(np.equal(LB, UB))
    dimension = np.array(
        [0.441, 0.352, 0.316, 0.285, 0.275, 0.273, 0.270, 0.268,
         0.267, 0.266, 0.265, 0.255]
    )
    s = a + b * dimension[np.minimum(Dims, 11)]

    L = covariance_chol

    U = np.log(np.random.rand(mrun))
    TH = np.zeros((Dims, mrun))
    THP = np.zeros(mrun)
    avg_acc = 0

    Z = multivariate_normal(np.zeros_like(X), L, mrun)

    for i in range(mrun):
        X_new = X + s * Z[i]
        X_new[sameind] = LB[sameind]

        X_new = adjust_bounds(X_new, LB, UB, avg_acc)

        P_new = beta * target(X_new)

        if P_new > best_P:
            X = X_new
            best_P = P_new
            P0 = P_new
            acc_rate = 1
        else:
            rho = P_new - P0
            acc_rate = 1 if rho > 0 else np.exp(rho)
            if U[i] <= rho:
                X = X_new
                P0 = P_new

        TH[:, i] = X
        THP[i] = best_P
        inv_i_plus_1 = 1.0 / (i + 1)
        avg_acc = avg_acc * i * inv_i_plus_1 + acc_rate * inv_i_plus_1
        s = a + b * avg_acc

    return TH, THP, avg_acc


def AMH_optimized_jit(X, target, covariance_chol, mrun, beta, LB, UB, a=1.0 / 9.0, b=8.0 / 9.0):
    TH, THP, avg_acc = run_amh(X, covariance_chol, mrun, beta, LB, UB, target, a, b)
    G = TH[:, -1]
    GP = THP[-1] / beta
    return G, GP, avg_acc


def _make_samples(
    NT2,
    allsamples,
    postval,
    beta,
    stage,
    covsmpl,
    resmpl,
    sample_stats=None,
    fault_parameter_stage_summary=None,
):
    fields = getattr(NT2, "_fields", ())
    values = {
        "allsamples": allsamples,
        "postval": postval,
        "beta": beta,
        "stage": stage,
        "covsmpl": covsmpl,
        "resmpl": resmpl,
        "sample_stats": sample_stats,
        "fault_parameter_stage_summary": fault_parameter_stage_summary,
    }
    return NT2(**{field: values[field] for field in fields})


def _sample_stats(samples):
    return getattr(samples, "sample_stats", None)


def _fault_parameter_stage_summary(samples):
    return getattr(samples, "fault_parameter_stage_summary", None)


def _empty_sample_stats():
    return {
        "stage": [],
        "beta": [],
        "delta_beta": [],
        "conditional_ess": [],
        "normalized_ess": [],
        "weight_cv": [],
        "max_weight_fraction": [],
        "weight_entropy": [],
        "unique_ancestor_fraction": [],
        "unique_ancestor_count": [],
        "acceptance_rate_mean": [],
        "acceptance_rate_min": [],
        "acceptance_rate_max": [],
        "jump_distance_mean": [],
        "jump_distance_median": [],
        "covariance_condition": [],
        "pre_mcmc_elapsed_seconds": [],
        "mutation_elapsed_seconds": [],
    }


def _append_sample_stats(sample_stats, record):
    if sample_stats is None:
        sample_stats = _empty_sample_stats()
    for key in _empty_sample_stats():
        sample_stats.setdefault(key, []).append(record.get(key, np.nan))
    return sample_stats


def _finalize_sample_stats(sample_stats):
    if sample_stats is None:
        return None
    return {
        key: np.asarray(value, dtype=float)
        for key, value in sample_stats.items()
    }


def _empty_fault_parameter_stage_summary(parameter_names, display_names=None):
    labels = display_names if display_names is not None else parameter_names
    return {
        "stage": [],
        "parameter_names": [str(name) for name in parameter_names],
        "display_names": [str(name) for name in labels],
        "median": [],
        "ci_lower": [],
        "ci_upper": [],
        "ci_width": [],
        "boundary_fraction": [],
    }


def _append_fault_parameter_stage_summary(summary, record):
    if summary is None:
        summary = _empty_fault_parameter_stage_summary(
            record["parameter_names"],
            record.get("display_names"),
        )
    for key in ["stage", "median", "ci_lower", "ci_upper", "ci_width", "boundary_fraction"]:
        summary.setdefault(key, []).append(record[key])
    return summary


def _finalize_fault_parameter_stage_summary(summary):
    if summary is None:
        return None
    finalized = {
        "parameter_names": list(summary.get("parameter_names", [])),
        "display_names": list(summary.get("display_names", summary.get("parameter_names", []))),
    }
    for key in ["stage", "median", "ci_lower", "ci_upper", "ci_width", "boundary_fraction"]:
        finalized[key] = np.asarray(summary.get(key, []), dtype=float)
    return finalized


def _fault_parameter_stage_record(
    samples,
    *,
    stage,
    diagnostic_indices=None,
    diagnostic_parameter_names=None,
    diagnostic_display_names=None,
    diagnostic_lower_bounds=None,
    diagnostic_upper_bounds=None,
    credible_interval=0.95,
    boundary_tol_fraction=0.01,
):
    if diagnostic_indices is None:
        return None
    indices = np.asarray(diagnostic_indices, dtype=int).reshape(-1)
    if indices.size == 0:
        return None
    arr = np.asarray(samples, dtype=float)
    selected = arr[:, indices]
    alpha = (1.0 - credible_interval) / 2.0
    ci_lower = np.percentile(selected, 100.0 * alpha, axis=0)
    ci_upper = np.percentile(selected, 100.0 * (1.0 - alpha), axis=0)
    names = diagnostic_parameter_names or [f"param_{idx}" for idx in indices]
    labels = diagnostic_display_names or names
    return {
        "stage": float(stage),
        "parameter_names": [str(name) for name in names],
        "display_names": [str(name) for name in labels],
        "median": np.median(selected, axis=0),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_width": ci_upper - ci_lower,
        "boundary_fraction": _boundary_fraction_matrix(
            selected,
            diagnostic_lower_bounds,
            diagnostic_upper_bounds,
            boundary_tol_fraction=boundary_tol_fraction,
        ),
    }


def _boundary_fraction_matrix(values, lower_bounds, upper_bounds, *, boundary_tol_fraction):
    if lower_bounds is None or upper_bounds is None:
        return np.full(values.shape[1], np.nan)
    lb = np.asarray(lower_bounds, dtype=float).reshape(-1)
    ub = np.asarray(upper_bounds, dtype=float).reshape(-1)
    fractions = np.full(values.shape[1], np.nan)
    for i in range(values.shape[1]):
        width = ub[i] - lb[i]
        if not np.isfinite(width) or width <= 0:
            continue
        tol = width * boundary_tol_fraction
        near = (values[:, i] <= lb[i] + tol) | (values[:, i] >= ub[i] - tol)
        fractions[i] = np.mean(near)
    return fractions


def _weight_stats(postval, beta_previous, beta_current):
    post = np.asarray(postval, dtype=float).reshape(-1)
    if post.size == 0:
        return {
            "conditional_ess": np.nan,
            "normalized_ess": np.nan,
            "weight_cv": np.nan,
            "max_weight_fraction": np.nan,
            "weight_entropy": np.nan,
        }
    logpst = post - np.max(post)
    logwght = (beta_current - beta_previous) * logpst
    wght = np.exp(logwght)
    wsum = np.sum(wght)
    if not np.isfinite(wsum) or wsum <= 0:
        return {
            "conditional_ess": np.nan,
            "normalized_ess": np.nan,
            "weight_cv": np.nan,
            "max_weight_fraction": np.nan,
            "weight_entropy": np.nan,
        }
    probwght = wght / wsum
    ess = 1.0 / np.sum(probwght ** 2)
    positive = probwght > 0
    entropy = -np.sum(probwght[positive] * np.log(probwght[positive]))
    mean_w = np.mean(wght)
    weight_cv = np.std(wght, ddof=1) / mean_w if post.size > 1 and mean_w != 0 else 0.0
    return {
        "conditional_ess": float(ess),
        "normalized_ess": float(ess / post.size),
        "weight_cv": float(weight_cv),
        "max_weight_fraction": float(np.max(probwght)),
        "weight_entropy": float(entropy),
    }


def _write_samples_h5(filename, samples):
    with h5py.File(filename, "w") as f:
        for key, value in samples._asdict().items():
            if value is None:
                continue
            if isinstance(value, dict):
                group = f.create_group(key)
                for subkey, subvalue in value.items():
                    _write_h5_dataset(group, subkey, subvalue)
            else:
                _write_h5_dataset(f, key, value)


def _write_h5_dataset(group, key, value):
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        dtype = h5py.string_dtype(encoding="utf-8")
        group.create_dataset(key, data=np.asarray(value, dtype=dtype), dtype=dtype)
        return
    group.create_dataset(key, data=value)


class NonlinearSMCclass:
    """SMC state machine for nonlinear geometry diagnostics."""

    def __init__(self, opt, samples, NT1, NT2, verbose=True):
        self.verbose = verbose
        self.opt = opt
        self.samples = samples
        self.NT1 = NT1
        self.NT2 = NT2

    def initialize(self):
        if self.verbose:
            print("-----------------------------------------------------------------------------------------------")
            print("-----------------------------------------------------------------------------------------------")
            print(
                f"Initializing ATMIP with {self.opt.N :8d} Markov chains and "
                f"{self.opt.Neff :8d} chain length."
            )

    def prior_samples_vectorize(self):
        numpars = self.opt.LB.shape[0]
        diffbnd = self.opt.UB - self.opt.LB
        diffbndN = np.tile(diffbnd, (self.opt.N, 1))
        LBN = np.tile(self.opt.LB, (self.opt.N, 1))

        sampzero = LBN + np.random.rand(self.opt.N, numpars) * diffbndN
        beta = np.array([0])
        stage = np.array([1])

        logpost = np.apply_along_axis(self.opt.target, 1, sampzero)
        postval = logpost.reshape(-1, 1)

        return _make_samples(
            self.NT2,
            sampzero,
            postval,
            beta,
            stage,
            None,
            None,
            _empty_sample_stats(),
            None,
        )

    def find_beta_numpy(self):
        beta1 = self.samples.beta[-1]
        max_post = np.max(self.samples.postval)
        logpst = self.samples.postval - max_post
        beta = beta1 + .5

        if beta > 1:
            beta = 1

        refcov = 1

        while beta - beta1 > 1e-6:
            curr_beta = (beta + beta1) / 2
            diffbeta = beta - beta1
            logwght = diffbeta * logpst
            wght = np.exp(logwght)

            covwght = np.std(wght, ddof=1) / np.mean(wght)

            if covwght > refcov:
                beta = curr_beta
            else:
                beta1 = curr_beta

        betanew = np.min(np.array([1, beta]))
        betaarray = np.append(self.samples.beta, betanew)
        newstage = np.arange(1, self.samples.stage[-1] + 2)
        return _make_samples(
            self.NT2,
            self.samples.allsamples,
            self.samples.postval,
            betaarray,
            newstage,
            self.samples.covsmpl,
            self.samples.resmpl,
            _sample_stats(self.samples),
            _fault_parameter_stage_summary(self.samples),
        )

    def resample_stage(self):
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2]) * logpst
        wght = np.exp(logwght)

        probwght = wght / np.sum(wght)
        inind = np.arange(0, self.opt.N)

        outind = deterministicR_optimized(inind, probwght)
        newsmpl = self.samples.allsamples[outind, :]

        samples = _make_samples(
            self.NT2,
            self.samples.allsamples,
            self.samples.postval,
            self.samples.beta,
            self.samples.stage,
            self.samples.covsmpl,
            newsmpl,
            _sample_stats(self.samples),
            _fault_parameter_stage_summary(self.samples),
        )

        return samples, outind

    def make_covariance_numpy(self, epsilon=1e-6):
        dims = self.samples.allsamples.shape[1]
        logpst = self.samples.postval - np.max(self.samples.postval)
        logwght = (self.samples.beta[-1] - self.samples.beta[-2]) * logpst
        wght = np.exp(logwght)

        probwght = wght / np.sum(wght)
        weightmat = np.tile(probwght, (1, dims))
        multmat = weightmat * self.samples.allsamples

        meansmpl = multmat.sum(axis=0, dtype="float")

        smpldiff = self.samples.allsamples - meansmpl
        weighted_diff = smpldiff * np.sqrt(probwght)
        covariance = weighted_diff.T @ weighted_diff

        covariance += epsilon * np.eye(dims)

        return _make_samples(
            self.NT2,
            self.samples.allsamples,
            self.samples.postval,
            self.samples.beta,
            self.samples.stage,
            covariance,
            self.samples.resmpl,
            _sample_stats(self.samples),
            _fault_parameter_stage_summary(self.samples),
        )

    def MCMC_samples_parallel_mpi(self, comm=None, a=1.0 / 9.0, b=8.0 / 9.0):
        rank = comm.Get_rank()
        size = comm.Get_size()

        dims = self.samples.allsamples.shape[1]
        mhsmpl = np.zeros([self.opt.N, dims])
        mhpost = np.zeros([self.opt.N, 1])

        covsmpl_chol = np.linalg.cholesky(self.samples.covsmpl)

        def process_sample(i):
            start = self.samples.resmpl[i, :]
            G, GP, acc = AMH_optimized_jit(
                start,
                self.opt.target,
                covsmpl_chol,
                self.opt.Neff,
                self.samples.beta[-1],
                self.opt.LB,
                self.opt.UB,
                a,
                b,
            )
            jump_distance = np.linalg.norm(np.asarray(G) - np.asarray(start))
            return np.transpose(G), GP, acc, jump_distance

        results = [process_sample(i) for i in range(rank, self.opt.N, size)]

        comm.Barrier()
        gathered_results = comm.gather(results, root=0)

        if rank == 0:
            acceptance_rates = np.full(self.opt.N, np.nan)
            jump_distances = np.full(self.opt.N, np.nan)
            gathered_results = [item for sublist in gathered_results for item in sublist]
            for i, (G, GP, acc, jump_distance) in enumerate(gathered_results):
                mhsmpl[i, :] = G
                mhpost[i] = GP
                acceptance_rates[i] = acc
                jump_distances[i] = jump_distance

            samples = _make_samples(
                self.NT2,
                mhsmpl,
                mhpost,
                self.samples.beta,
                self.samples.stage,
                self.samples.covsmpl,
                self.samples.resmpl,
                _sample_stats(self.samples),
                _fault_parameter_stage_summary(self.samples),
            )
        else:
            samples = None
            acceptance_rates = None
            jump_distances = None

        comm.Barrier()
        samples = comm.bcast(samples, root=0)
        acceptance_rates = comm.bcast(acceptance_rates, root=0)
        jump_distances = comm.bcast(jump_distances, root=0)

        return samples, acceptance_rates, jump_distances


def SMC_samples_parallel_mpi_nonlinear(
    opt,
    samples,
    NT1,
    NT2,
    comm=None,
    save_at_final_stage=True,
    save_interval=1,
    save_at_interval=False,
    covariance_epsilon=1e-6,
    amh_a=1.0 / 9.0,
    amh_b=8.0 / 9.0,
    diagnostic_indices=None,
    diagnostic_parameter_names=None,
    diagnostic_display_names=None,
    diagnostic_lower_bounds=None,
    diagnostic_upper_bounds=None,
    diagnostic_credible_interval=0.95,
    diagnostic_boundary_tol_fraction=0.01,
):
    """Sequential Monte Carlo sampling with stage-level diagnostics."""

    rank = comm.Get_rank()

    current = NonlinearSMCclass(opt, samples, NT1, NT2)
    if rank == 0:
        current.initialize()

    if samples.allsamples is None:
        if rank == 0:
            print("------Calculating the prior posterior values at stage 1-----", flush=True)
            current = NonlinearSMCclass(opt, samples, NT1, NT2)
            samples = current.prior_samples_vectorize()
            start_time = time.time()
        else:
            samples = None
    else:
        if rank == 0:
            start_time = time.time()

    comm.Barrier()
    samples = comm.bcast(samples, root=0)

    while samples.beta[-1] != 1:
        stage_record = None
        if rank == 0:
            beta_previous = float(samples.beta[-1])
            current = NonlinearSMCclass(opt, samples, NT1, NT2)
            samples = current.find_beta_numpy()
            beta_current = float(samples.beta[-1])

            stage_record = {
                "stage": float(samples.stage[-1]),
                "beta": beta_current,
                "delta_beta": beta_current - beta_previous,
            }
            stage_record.update(
                _weight_stats(samples.postval, beta_previous, beta_current)
            )

            current = NonlinearSMCclass(opt, samples, NT1, NT2)
            samples, outind = current.resample_stage()
            unique_ancestors = np.unique(outind).size
            stage_record["unique_ancestor_count"] = float(unique_ancestors)
            stage_record["unique_ancestor_fraction"] = float(unique_ancestors / opt.N)

            current = NonlinearSMCclass(opt, samples, NT1, NT2)
            samples = current.make_covariance_numpy(epsilon=covariance_epsilon)
            try:
                stage_record["covariance_condition"] = float(np.linalg.cond(samples.covsmpl))
            except np.linalg.LinAlgError:
                stage_record["covariance_condition"] = np.inf

            print(
                f"Starting metropolis chains at stage = {samples.stage[-1] :3d} "
                f"and beta = {samples.beta[-1] :.6f}.",
                flush=True,
            )

            end_time = time.time()
            execution_time = end_time - start_time
            current_time = datetime.now().strftime("%y-%m-%d %H:%M:%S")
            stage_record["pre_mcmc_elapsed_seconds"] = float(execution_time)
            print(
                f"The while loop took {execution_time:.6f} seconds to execute. "
                f"Current time: {current_time}",
                flush=True,
            )

            if save_at_interval and samples.stage[-1] % save_interval == 0:
                _write_samples_h5(f"samples_stage_{samples.stage[-1]}.h5", samples)
        else:
            samples = None

        comm.Barrier()
        samples = comm.bcast(samples, root=0)

        mutation_start = time.time()
        current = NonlinearSMCclass(opt, samples, NT1, NT2)
        samples, acceptance_rates, jump_distances = current.MCMC_samples_parallel_mpi(
            comm=comm,
            a=amh_a,
            b=amh_b,
        )

        if rank == 0:
            stage_record["mutation_elapsed_seconds"] = float(time.time() - mutation_start)
            finite_acc = acceptance_rates[np.isfinite(acceptance_rates)]
            finite_jump = jump_distances[np.isfinite(jump_distances)]
            stage_record["acceptance_rate_mean"] = (
                float(np.mean(finite_acc)) if finite_acc.size else np.nan
            )
            stage_record["acceptance_rate_min"] = (
                float(np.min(finite_acc)) if finite_acc.size else np.nan
            )
            stage_record["acceptance_rate_max"] = (
                float(np.max(finite_acc)) if finite_acc.size else np.nan
            )
            stage_record["jump_distance_mean"] = (
                float(np.mean(finite_jump)) if finite_jump.size else np.nan
            )
            stage_record["jump_distance_median"] = (
                float(np.median(finite_jump)) if finite_jump.size else np.nan
            )
            sample_stats = _append_sample_stats(_sample_stats(samples), stage_record)
            fault_record = _fault_parameter_stage_record(
                samples.allsamples,
                stage=samples.stage[-1],
                diagnostic_indices=diagnostic_indices,
                diagnostic_parameter_names=diagnostic_parameter_names,
                diagnostic_display_names=diagnostic_display_names,
                diagnostic_lower_bounds=diagnostic_lower_bounds,
                diagnostic_upper_bounds=diagnostic_upper_bounds,
                credible_interval=diagnostic_credible_interval,
                boundary_tol_fraction=diagnostic_boundary_tol_fraction,
            )
            fault_summary = _fault_parameter_stage_summary(samples)
            if fault_record is not None:
                fault_summary = _append_fault_parameter_stage_summary(
                    fault_summary,
                    fault_record,
                )
            samples = _make_samples(
                NT2,
                samples.allsamples,
                samples.postval,
                samples.beta,
                samples.stage,
                samples.covsmpl,
                samples.resmpl,
                sample_stats,
                fault_summary,
            )

        comm.Barrier()
        samples = comm.bcast(samples, root=0)

    if rank == 0:
        samples = _make_samples(
            NT2,
            samples.allsamples,
            samples.postval,
            samples.beta,
            samples.stage,
            samples.covsmpl,
            samples.resmpl,
            _finalize_sample_stats(_sample_stats(samples)),
            _finalize_fault_parameter_stage_summary(
                _fault_parameter_stage_summary(samples)
            ),
        )
        if save_at_final_stage:
            _write_samples_h5("samples_final.h5", samples)

    samples = comm.bcast(samples if rank == 0 else None, root=0)
    return samples
