"""Shared result-statistics contract for standalone nonlinear SMC solvers.

The legacy and registry-based nonlinear geometry solvers use different
parameter registries and forward-model implementations.  This mixin unifies
only their result boundary: an already activated representative model is read
into structured fit rows, which can then be printed or written without
selecting another posterior sample or rerunning the sampler.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from .fit_statistics import (
    data_fit_vectors,
    fit_metrics_from_vectors,
    fit_statistics_rows_to_dataframe,
    format_fit_statistics_report,
    format_fit_statistics_table,
    weighted_fit_metrics_from_residual,
    write_fit_statistics_report_files,
)


class NonlinearFitStatisticsMixin:
    """Provide structured fit reporting for standalone nonlinear SMC classes.

    Subclasses publish ``_active_result_model`` when ``returnModel()`` has
    materialized a representative model and implement
    ``_fit_dataset_sigmas()`` using the same parameter mapping as their
    likelihood.  Covariance metrics are prepared once by ``setLikelihood()``.
    """

    def _fit_dataset_sigmas(self):  # pragma: no cover - subclass contract
        raise NotImplementedError

    def _require_active_fit_model(self, model):
        active = getattr(self, "_active_result_model", None)
        if active != model:
            raise RuntimeError(
                f"Model {model!r} is not active. Call returnModel(model={model!r}, "
                "print_fit_statistics=False) before collecting fit statistics."
            )
        if str(model).lower() == "std":
            raise ValueError("The posterior standard deviation is not a predictive model.")

    def _fit_sigma_groups(self):
        """Return dataset-to-group labels from the likelihood sigma contract."""
        datas = list(getattr(self, "datas", []))
        names = [str(data.name) for data in datas]
        config = getattr(self, "sigmas", {})
        mode = config.get("mode", "individual")
        if mode == "single":
            members = {"all": names}
        elif mode == "individual":
            members = {f"group_{name}": [name] for name in names}
        elif mode == "grouped":
            members = {
                str(group): list(group_names)
                for group, group_names in (config.get("groups") or {}).items()
            }
        else:
            raise ValueError(f"Unknown sigmas mode: {mode}")
        groups = {
            str(name): str(group)
            for group, group_names in members.items()
            for name in group_names
        }
        missing = [name for name in names if name not in groups]
        if missing:
            raise ValueError(
                "Sigma groups do not cover active datasets: " + ", ".join(missing)
            )
        return groups

    def collect_fit_statistics(
        self,
        *,
        model="median",
        include_dataset=True,
        include_global=False,
        include_dataset_average=False,
        include_weighted=False,
        rebuild_synth=False,
    ):
        """Collect fit rows for the currently activated nonlinear model.

        ``collect_fit_statistics`` never selects a posterior representative.
        Set ``rebuild_synth=True`` only when the active data objects may have
        been overwritten since ``returnModel``; this repeats prediction for the
        same active vector but does not sample or solve a new model.

        Standalone geometry SMC has no assembled global solver vector, so
        ``include_global`` is accepted for interface consistency and produces
        no additional row.
        """
        self._require_active_fit_model(model)
        datas = list(getattr(self, "datas", []))
        verticals = list(getattr(self, "verticals", []))
        if len(datas) != len(verticals):
            raise ValueError("Active data and vertical flag counts do not match.")

        if rebuild_synth:
            self._rebuild_active_fit_synthetics()

        metrics = getattr(self, "data_covariance_metrics", {})
        sigmas = self._fit_dataset_sigmas() if include_weighted else {}
        sigma_groups = self._fit_sigma_groups() if include_weighted else {}
        rows = []
        if include_dataset:
            for data, vertical in zip(datas, verticals):
                observed, synthetic = data_fit_vectors(data, vertical=vertical)
                row = {
                    "scope": "dataset",
                    "model": model,
                    "dataset": getattr(data, "name", None),
                    "data_type": getattr(data, "dtype", None),
                    "vertical": bool(vertical),
                    "poly": None,
                    "sigma_group": sigma_groups.get(str(data.name)),
                    **fit_metrics_from_vectors(observed, synthetic),
                }
                if include_weighted:
                    metric = metrics.get(data.name) if isinstance(metrics, Mapping) else None
                    sigma = sigmas.get(data.name) if isinstance(sigmas, Mapping) else None
                    if metric is not None and sigma is not None:
                        row.update(
                            weighted_fit_metrics_from_residual(
                                synthetic - observed,
                                metric,
                                sigma=sigma,
                            )
                        )
                rows.append(row)

        if include_dataset_average and rows:
            dataset_rows = [row for row in rows if row["scope"] == "dataset"]
            if dataset_rows:
                rows.append(
                    {
                        "scope": "dataset_average",
                        "model": model,
                        "dataset": None,
                        "data_type": None,
                        "vertical": None,
                        "poly": None,
                        "rms": float(np.mean([row["rms"] for row in dataset_rows])),
                        "vr": float(np.mean([row["vr"] for row in dataset_rows])),
                        "ss_res": float(np.sum([row["ss_res"] for row in dataset_rows])),
                        "ss_obs": float(np.sum([row["ss_obs"] for row in dataset_rows])),
                        "n_observations": int(
                            np.sum([row["n_observations"] for row in dataset_rows])
                        ),
                    }
                )
        return rows

    @staticmethod
    def fit_statistics_to_dataframe(rows: Sequence[Mapping]):
        """Convert structured fit rows to a pandas DataFrame."""
        return fit_statistics_rows_to_dataframe(rows)

    @staticmethod
    def format_fit_statistics_report(rows: Sequence[Mapping], *, model=None):
        """Format structured fit rows as a compact text report."""
        return format_fit_statistics_report(rows, model=model)

    def write_fit_statistics_report(
        self,
        outdir="output",
        rows=None,
        *,
        model="median",
        include_dataset=True,
        include_global=False,
        include_dataset_average=False,
        include_weighted=False,
        rebuild_synth=False,
        basename="fit_statistics",
        formats=("txt", "tsv"),
    ):
        """Write the same structured rows used by the console report."""
        if rows is None:
            rows = self.collect_fit_statistics(
                model=model,
                include_dataset=include_dataset,
                include_global=include_global,
                include_dataset_average=include_dataset_average,
                include_weighted=include_weighted,
                rebuild_synth=rebuild_synth,
            )
        return write_fit_statistics_report_files(
            rows,
            outdir,
            basename=basename,
            formats=formats,
            model=model,
        )

    def calculate_and_print_fit_statistics(self, model="median"):
        """Activate when needed, then print one covariance-aware fit table."""
        if getattr(self, "_active_result_model", None) != model:
            self.returnModel(model=model, print_fit_statistics=False)
        rows = self.collect_fit_statistics(
            model=model,
            include_dataset=True,
            include_global=False,
            include_weighted=True,
            rebuild_synth=False,
        )
        print("\n" + format_fit_statistics_table(rows, model=model))
        return rows
