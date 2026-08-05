"""High-level figure products built from existing ECAT/CSI plot methods."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data_plot_utils import _plot_crossfaultoffset_fit, _plot_leveling_fit
from .data_prediction import get_geodata_prediction_specs, resolve_data_poly
from .interseismic_fields import get_fault_by_name, get_faults_from_inversion


def _merge_product_plot_kwargs(
    defaults,
    common=None,
    specific=None,
    *,
    locked=(),
    context="figure product",
):
    """Merge display kwargs while protecting product-owned arguments.

    Precedence is ``defaults < common < specific``.  Arguments that identify
    the scientific field or control the product lifecycle are supplied by the
    wrapper itself and therefore cannot also appear in free-form kwargs.
    """
    common = dict(common or {})
    specific = dict(specific or {})
    conflicts = sorted((set(common) | set(specific)) & set(locked))
    if conflicts:
        names = ", ".join(conflicts)
        raise ValueError(f"{context} owns these keyword(s): {names}")
    merged = dict(defaults or {})
    merged.update(common)
    merged.update(specific)
    return merged


def _as_name_set(values: Sequence[str] | str | None) -> set[str] | None:
    if values is None or values == "all":
        return None
    if isinstance(values, str):
        return {values}
    return {str(value) for value in values}


def _resolve_faults(inversion: Any, faults: Sequence[Any] | str | Any | None = None) -> list[Any]:
    all_faults = list(inversion._get_faults() if hasattr(inversion, "_get_faults") else get_faults_from_inversion(inversion))
    if faults is None or (isinstance(faults, str) and faults == "all"):
        return all_faults
    if isinstance(faults, str):
        return [get_fault_by_name(inversion, faults)]
    if not isinstance(faults, Iterable):
        return [faults]
    resolved = []
    fault_map = {str(getattr(fault, "name", "")): fault for fault in all_faults}
    for fault in faults:
        if isinstance(fault, str):
            try:
                resolved.append(fault_map[fault])
            except KeyError as exc:
                raise ValueError(f"Fault '{fault}' was not found") from exc
        else:
            resolved.append(fault)
    return resolved


def _iter_geodata(inversion: Any, datasets=None, data_types=None):

    dataset_filter = _as_name_set(datasets)
    type_filter = _as_name_set(data_types)
    for spec in get_geodata_prediction_specs(inversion):
        data = spec.data
        name = str(getattr(data, "name", ""))
        dtype = str(getattr(data, "dtype", ""))
        if dataset_filter is not None and name not in dataset_filter:
            continue
        if type_filter is not None and dtype not in type_filter:
            continue
        yield spec


def _prepare_fault_traces(faults: Sequence[Any], *, color="k", linewidth=None):
    for fault in faults:
        if getattr(fault, "lon", None) is None or getattr(fault, "lat", None) is None:
            if hasattr(fault, "setTrace"):
                fault.setTrace(0.1)
        fault.color = color
        if linewidth is not None:
            fault.linewidth = linewidth


def _save_last_fault_plot(
    fault: Any,
    path: Path,
    *,
    dpi=300,
    bbox_inches="tight",
    preferred="fault",
):
    """Save the real CSI/geodeticplot figure created by ``fault.plot``.

    CSI fault plotting stores figures on ``fault.slipfig`` instead of returning
    a Matplotlib ``Figure``.  Saving ``plt.gcf()`` here can capture an unrelated
    or empty current figure, so product helpers save the explicit stored figure.
    """
    plotter = getattr(fault, "slipfig", None)
    candidates = []
    if plotter is not None:
        if preferred == "map":
            candidates.extend([getattr(plotter, "figCarte", None), getattr(plotter, "figFaille", None)])
        else:
            candidates.extend([getattr(plotter, "figFaille", None), getattr(plotter, "figCarte", None)])
    for fig in candidates:
        if fig is not None and hasattr(fig, "savefig"):
            fig.savefig(path, dpi=dpi, bbox_inches=bbox_inches)
            return path

    # Fallback for non-CSI plotters that still rely on pyplot's current figure.
    import matplotlib.pyplot as plt

    plt.gcf().savefig(path, dpi=dpi, bbox_inches=bbox_inches)
    return path


def plot_data_fits_product(
    inversion: Any,
    *,
    datasets="all",
    data_types=None,
    faults=None,
    data_poly="config",
    outdir="Modeling",
    file_type="png",
    plot_data=True,
    antisymmetric=True,
    res_use_data_norm=True,
    cmap="RdBu_r",
    gps_title=True,
    sar_title=True,
    gps_figsize=None,
    sar_figsize="double",
    gps_scale=0.05,
    gps_legendscale=0.2,
    sar_cbaxis=(0.1, 0.15, 0.35, 0.04),
    remove_direction_labels=False,
    gps_kwargs=None,
    sar_kwargs=None,
    show=True,
) -> dict[str, list[Path] | list[str]]:
    """Build observed/synthetic fit plots for configured geodetic datasets.

    This is a thin product-level wrapper around existing CSI/ECAT plotting
    methods.  It does not change observations or solved model parameters.

    ``data_poly="config"`` follows the parsed per-dataset
    ``config.geodata['polys']`` settings.  Use ``"include"`` to force solved
    corrections into every selected prediction, or ``None`` for the explicit
    source/slip-only diagnostic view.

    ``gps_kwargs`` and ``sar_kwargs`` override display defaults only.  The
    product owns dataset identity, plotted data roles, output path, and
    ``show``.  Supplying those owned keys in a free-form dictionary raises a
    clear :class:`ValueError`.

    Returns
    -------
    dict
        Written paths grouped by data type, plus names skipped because their
        data type has no product implementation.
    """
    target_faults = _resolve_faults(inversion, faults)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gps_kwargs = dict(gps_kwargs or {})
    sar_kwargs = dict(sar_kwargs or {})
    written: dict[str, list[Path] | list[str]] = {
        "gps": [],
        "insar": [],
        "leveling": [],
        "crossfaultoffset": [],
        "skipped": [],
    }

    _prepare_fault_traces(target_faults, color="k", linewidth=2.0)
    for spec in _iter_geodata(inversion, datasets=datasets, data_types=data_types):
        data = spec.data
        vertical = spec.vertical
        resolved_poly = resolve_data_poly(spec.configured_poly, requested=data_poly)
        dtype = str(getattr(data, "dtype", ""))
        name = str(getattr(data, "name", "dataset"))
        if dtype == "gps":
            kwargs = {"vertical": vertical, "poly": resolved_poly}
            if vertical is None:
                kwargs.pop("vertical")
            data.buildsynth(target_faults, **kwargs)
            if not plot_data:
                continue
            box = [data.lon.min(), data.lon.max(), data.lat.min(), data.lat.max()]
            current_gps_kwargs = _merge_product_plot_kwargs(
                {
                    "drawCoastlines": True,
                    "scale": gps_scale,
                    "legendscale": gps_legendscale,
                    "color": ["#e33e1c", "#2e5b99"],
                    "seacolor": "lightblue",
                    "box": box,
                    "titleyoffset": 1.02,
                    "title": gps_title,
                    "figsize": gps_figsize,
                    "remove_direction_labels": remove_direction_labels,
                },
                gps_kwargs,
                locked=("faults", "data"),
                context="plot_data_fits_product(gps)",
            )
            data.plot(faults=target_faults, data=["data", "synth"], **current_gps_kwargs)
            path = outdir / f"gps_{name}"
            data.fig.savefig(str(path), ftype=file_type, dpi=600, bbox_inches="tight", mapaxis=None, saveFig=["map"])
            written["gps"].append(path.with_suffix(f".{file_type}"))
        elif dtype == "insar":
            data.buildsynth(target_faults, vertical=True, poly=resolved_poly)
            if not plot_data:
                continue
            datamin, datamax = float(np.nanmin(data.vel)), float(np.nanmax(data.vel))
            absmax = max(abs(datamin), abs(datamax))
            data_norm = [-absmax, absmax] if antisymmetric else [datamin, datamax]
            path = outdir / f"{name}_fit_comparison.{file_type}"
            current_sar_kwargs = _merge_product_plot_kwargs(
                {
                    "cmap": cmap,
                    "vmin": data_norm[0],
                    "vmax": data_norm[1],
                    "share_colorbar": res_use_data_norm,
                    "cbaxis": sar_cbaxis,
                    "figsize": sar_figsize,
                },
                sar_kwargs,
                locked=("faults", "save_path", "show"),
                context="plot_data_fits_product(insar)",
            )
            data.plot_fit_comparison(
                faults=target_faults,
                save_path=path,
                show=show,
                **current_sar_kwargs,
            )
            written["insar"].append(path)
        elif dtype == "leveling":
            data.buildsynth(target_faults, vertical=True, poly=resolved_poly)
            if plot_data:
                for item in ("data", "synth"):
                    data.write2file(f"{name}_{item}.txt", outDir=str(outdir), data=item)
                _plot_leveling_fit(data, save_dir=outdir, file_type=file_type, show=show)
                written["leveling"].append(outdir / f"{name}_leveling_fit.{file_type}")
        elif dtype == "crossfaultoffset":
            data.buildsynth(target_faults, poly=resolved_poly)
            if plot_data:
                for item in ("data", "synth"):
                    data.write2file(f"{name}_{item}.txt", outDir=str(outdir), data=item)
                _plot_crossfaultoffset_fit(data, save_dir=outdir, file_type=file_type, show=show)
                written["crossfaultoffset"].append(outdir / f"{name}_crossfault_fit.{file_type}")
        else:
            written["skipped"].append(name)
    return written


def _normalize_fault_field(field: str) -> str:
    key = str(field).lower().replace("-", "_")
    aliases = {
        "slip": "total",
        "total_slip": "total",
        "total": "total",
        "strike": "strikeslip",
        "strikeslip": "strikeslip",
        "strike_slip": "strikeslip",
        "ss": "strikeslip",
        "dip": "dipslip",
        "dipslip": "dipslip",
        "dip_slip": "dipslip",
        "ds": "dipslip",
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unknown fault field '{field}'. Use total, strike, or dip.") from exc


def plot_fault_fields_product(
    inversion: Any,
    *,
    faults=None,
    fields=("total",),
    field_plot_kwargs=None,
    outdir="output",
    file_type="pdf",
    slip_cmap="cmc.roma_r",
    show=True,
    savefig=True,
    **plot_kwargs,
) -> dict[str, Any]:
    """Plot standard slip fields on one or more faults.

    Display kwargs resolve as product defaults, then common ``plot_kwargs``,
    then ``field_plot_kwargs[field]``.  Fault selection, normalized slip field,
    output lifecycle, and file type remain product-owned.
    """
    outdir = Path(outdir)
    if savefig:
        outdir.mkdir(parents=True, exist_ok=True)
    field_plot_kwargs = dict(field_plot_kwargs or {})
    results = {}
    for field in fields:
        slip = _normalize_fault_field(field)
        current_plot_kwargs = _merge_product_plot_kwargs(
            {"cmap": slip_cmap},
            plot_kwargs,
            field_plot_kwargs.get(str(field), {}),
            locked=("faults", "slip", "show", "savefig", "outdir", "ftype"),
            context=f"plot_fault_fields_product({field!r})",
        )
        suffix = current_plot_kwargs.pop("suffix", f"_{slip}")
        results[slip] = inversion.plot_multifaults_slip(
            faults=faults,
            slip=slip,
            show=show,
            savefig=savefig,
            outdir=str(outdir),
            ftype=file_type,
            suffix=suffix,
            **current_plot_kwargs,
        )
    return results


def plot_interseismic_summary_product(
    inversion: Any,
    *,
    faults="all",
    fields=("tectonic_loading_rate", "backslip_rate", "coupling_ratio"),
    field_plot_kwargs=None,
    euler_params1=None,
    euler_params2=None,
    solution=None,
    slip_component="strikeslip",
    model=None,
    store=True,
    outdir="output/interseismic",
    file_type="png",
    show=True,
    savefig=True,
    dpi=300,
    **plot_kwargs,
) -> dict[str, dict[str, Any]]:
    """Plot a standard bundle of Euler/block interseismic fields.

    Each fault is calculated once and the same result is reused for every
    requested field.  Per-field dictionaries affect display only; they cannot
    replace ``field``, ``result``, ``show``, or ``savefig``.
    """
    outdir = Path(outdir)
    if savefig:
        outdir.mkdir(parents=True, exist_ok=True)
    field_plot_kwargs = dict(field_plot_kwargs or {})
    results = {}
    for fault in _resolve_faults(inversion, faults):
        fault_name = str(getattr(fault, "name", fault))
        result = inversion.calculate_interseismic_fields(
            fault_name,
            euler_params1=euler_params1,
            euler_params2=euler_params2,
            solution=solution,
            slip_component=slip_component,
            model=model,
            store=store,
        )
        results[fault_name] = {}
        for field in fields:
            current_plot_kwargs = _merge_product_plot_kwargs(
                {},
                plot_kwargs,
                field_plot_kwargs.get(str(field), {}),
                locked=("field", "result", "show", "savefig"),
                context=f"plot_interseismic_summary_product({field!r})",
            )
            ret = inversion.plot_interseismic_field(
                fault_name,
                field=field,
                result=result,
                show=show,
                savefig=False,
                **current_plot_kwargs,
            )
            if savefig:
                path = outdir / f"{fault_name}_{str(field).replace('_', '-')}.{file_type}"
                _save_last_fault_plot(fault, path, dpi=dpi, bbox_inches="tight")
            results[fault_name][str(field)] = ret
    return results


def plot_deep_slip_loading_summary_product(
    inversion: Any,
    *,
    shallow_fault,
    deep_faults=None,
    fields=("deep_loading_proxy_rate", "shallow_slip_rate", "coupling_to_deep"),
    field_plot_kwargs=None,
    result=None,
    mapping=None,
    field_mapping=None,
    field_shallow_selector="all",
    shallow_selector=None,
    deep_selectors=None,
    solution=None,
    component="strikeslip",
    zero_tolerance=1.0e-12,
    model=None,
    store=True,
    mapping_kwargs=None,
    outdir="output/deep_slip_loading",
    file_type="png",
    show=True,
    savefig=True,
    dpi=300,
    **plot_kwargs,
) -> dict[str, Any]:
    """Plot a standard bundle of deep-slip proxy fields.

    Calculate the shared mapping result once unless ``result`` is supplied,
    then delegate each requested field to the existing plotting method.
    Scientific identity and output lifecycle arguments remain product-owned.
    """
    outdir = Path(outdir)
    if savefig:
        outdir.mkdir(parents=True, exist_ok=True)
    field_plot_kwargs = dict(field_plot_kwargs or {})
    if result is None:
        result = inversion.calculate_deep_slip_loading_fields(
            shallow_fault=shallow_fault,
            deep_faults=deep_faults,
            mapping=mapping,
            field_mapping=field_mapping,
            field_shallow_selector=field_shallow_selector,
            shallow_selector=shallow_selector,
            deep_selectors=deep_selectors,
            solution=solution,
            component=component,
            zero_tolerance=zero_tolerance,
            model=model,
            store=store,
            **dict(mapping_kwargs or {}),
        )
    fault_name = str(result["shallow_fault"])
    results = {}
    for field in fields:
        current_plot_kwargs = _merge_product_plot_kwargs(
            {},
            plot_kwargs,
            field_plot_kwargs.get(str(field), {}),
            locked=(
                "field",
                "shallow_fault",
                "deep_faults",
                "result",
                "mapping",
                "show",
                "savefig",
            ),
            context=f"plot_deep_slip_loading_summary_product({field!r})",
        )
        ret = inversion.plot_deep_slip_loading_field(
            field=field,
            shallow_fault=shallow_fault,
            deep_faults=deep_faults,
            result=result,
            mapping=mapping,
            show=show,
            savefig=False,
            **current_plot_kwargs,
        )
        if savefig:
            path = outdir / f"{fault_name}_{str(field).replace('_', '-')}.{file_type}"
            shallow = get_fault_by_name(inversion, result["shallow_fault"])
            _save_last_fault_plot(shallow, path, dpi=dpi, bbox_inches="tight")
        results[str(field)] = ret
    return results
