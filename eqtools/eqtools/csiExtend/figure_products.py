"""High-level figure products built from existing ECAT/CSI plot methods."""

from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np

from .data_plot_utils import _plot_crossfaultoffset_fit, _plot_leveling_fit
from .data_prediction import (
    get_geodata_prediction_specs,
    rebuild_diagnostic_synthetics,
)
from .interseismic_fields import get_fault_by_name, get_faults_from_inversion
from ecat_viz import PlotStyle, normalize_image_format


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


def _resolve_faults(
    inversion: Any,
    faults: Sequence[Any] | str | Any | None = None,
) -> list[Any]:
    selector = getattr(inversion, "_select_faults", None)
    if callable(selector):
        return list(selector(faults))
    all_faults = list(
        inversion._get_faults()
        if hasattr(inversion, "_get_faults")
        else get_faults_from_inversion(inversion)
    )
    if faults is None or (
        isinstance(faults, str) and faults.strip().lower() == "all"
    ):
        return all_faults
    if isinstance(faults, str):
        return [get_fault_by_name(inversion, faults)]
    if not isinstance(faults, Iterable):
        return [faults]
    faults = list(faults)
    if not faults:
        return all_faults
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


def _save_geodetic_map(
    plotter: Any,
    stem: Path,
    *,
    file_type: str,
    dpi=600,
    bbox_inches="tight",
) -> Path:
    """Save a CSI geodetic map and return its real output path.

    ``csi.geodeticplot.savefig`` accepts a filename prefix and appends
    ``_map.<format>`` when only the map panel is requested.
    """
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    plotter.savefig(
        str(stem),
        ftype=file_type,
        dpi=dpi,
        bbox_inches=bbox_inches,
        mapaxis=None,
        saveFig=["map"],
    )
    return stem.parent / f"{stem.name}_map.{file_type}"


def _gps_comparison_kwargs(defaults=None, kwargs=None):
    """Separate display preferences from the product's scientific ownership.

    show_vertical is a visibility preference; the active-U flag still comes
    from inversion configuration. Neither output identity nor prediction roles
    can be replaced through display kwargs. Explicit style fields override only
    their corresponding defaults, so a legend font override retains PDF/font
    settings supplied by the result entry point.
    """
    options = dict(kwargs or {})
    show_vertical = options.pop("show_vertical", True)
    if not isinstance(show_vertical, (bool, np.bool_)):
        raise ValueError("GPS show_vertical must be a boolean")
    if "style_kwargs" in options and defaults and "style_kwargs" in defaults:
        options["style_kwargs"] = {
            **dict(defaults["style_kwargs"]),
            **dict(options["style_kwargs"] or {}),
        }
    options = _merge_product_plot_kwargs(
        defaults, options,
        locked=("vertical", "faults", "data", "show", "save_path", "ax"),
        context="GPS comparison product",
    )
    legacy_keys = set(options) & {"scale", "legendscale", "box", "verticalsize", "verticalnorm", "drawCoastlines", "Map", "Fault"}
    if legacy_keys:
        raise ValueError("GPS comparison does not accept legacy map options: " + ", ".join(sorted(legacy_keys)))
    return options, bool(show_vertical)


def plot_gps_comparison_product(data, *, vertical, faults, save_path, show,
                                defaults=None, kwargs=None):
    """Render already prepared GPS fields, without rebuilding predictions.

    Shared by Figure Products and the independent geometry SMC result path.
    Configuration owns the active U flag; display options may only hide it.
    """
    method = getattr(data, "plot_fit_comparison", None)
    if not callable(method):
        raise RuntimeError("GPS comparison requires the matching CSI update")
    options, show_vertical = _gps_comparison_kwargs(defaults, kwargs)
    options.setdefault("close", not show)
    return method(vertical=bool(vertical) and show_vertical, faults=faults,
                  save_path=str(save_path), show=show, **options)


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
    gps_plot_mode="legacy",
    sar_kwargs=None,
    opticorr_kwargs=None,
    raster_render_mode="points",
    raster_cell_edge_width=0.25,
    gps_fault_color="k",
    sar_fault_color="k",
    fault_linewidth=2.0,
    pdf_fonttype=None,
    gps_fontsize=None,
    sar_fontsize=None,
    show=True,
) -> dict[str, list[Path] | list[str]]:
    """Build observed/synthetic fit plots for configured geodetic datasets.

    This is a thin product-level wrapper around existing CSI/ECAT plotting
    methods.  It does not change observations or solved model parameters.

    ``data_poly="config"`` follows the parsed per-dataset
    ``config.geodata['polys']`` settings.  Use ``"include"`` to force solved
    corrections into every selected prediction, or ``None`` for the explicit
    source/slip-only diagnostic view.

    ``gps_kwargs``, ``sar_kwargs``, and ``opticorr_kwargs`` override display
    defaults only. ``gps_plot_mode="legacy"`` preserves CSI map output;
    ``"comparison"`` draws one EN/U comparison and uses the configured U flag.
    Its ``show_vertical=False`` only hides U; it cannot activate an unused
    component. ``gps_scale`` and ``gps_legendscale`` apply only to legacy maps.
    Comparison display conversion and paper scaling are explicit in
    ``gps_kwargs`` (``value_scale``, ``value_unit``, ``arrow_scale``).
    ``antisymmetric=True`` uses zero-centred automatic limits;
    ``False`` uses each raster field's finite data range. Explicit ``vmin``
    and ``vmax`` in the type-specific kwargs remain authoritative.
    ``raster_render_mode`` controls only the spatial carrier of
    InSAR/optical figures: ``"points"`` preserves sample-center plots,
    ``"cells"`` requires corner geometry, and ``"auto"`` uses cells when
    available. The
    product owns dataset identity, plotted data roles, output path, and
    ``show``.  Supplying those owned keys in a free-form dictionary raises a
    clear :class:`ValueError`.

    Returns
    -------
    dict
        Written paths grouped by data type, plus names skipped because their
        data type has no product implementation.
    """
    file_type = normalize_image_format(file_type)
    if gps_plot_mode not in ("legacy", "comparison"):
        raise ValueError("gps_plot_mode must be 'legacy' or 'comparison'")
    if gps_plot_mode == "comparison":
        _gps_comparison_kwargs(kwargs=gps_kwargs)
    target_faults = _resolve_faults(inversion, faults)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gps_kwargs = dict(gps_kwargs or {})
    sar_kwargs = dict(sar_kwargs or {})
    opticorr_kwargs = dict(opticorr_kwargs or {})
    written: dict[str, list[Path] | list[str]] = {
        "gps": [],
        "insar": [],
        "opticorr": [],
        "leveling": [],
        "crossfaultoffset": [],
        "skipped": [],
    }

    grouped_specs: dict[str, list[Any]] = {
        "gps": [],
        "insar": [],
        "opticorr": [],
        "leveling": [],
        "crossfaultoffset": [],
        "other": [],
    }
    for spec in _iter_geodata(inversion, datasets=datasets, data_types=data_types):
        dtype = str(getattr(spec.data, "dtype", "")).lower()
        grouped_specs[dtype if dtype in grouped_specs else "other"].append(spec)

    if gps_plot_mode == "comparison" and plot_data:
        for spec in grouped_specs["gps"]:
            if not callable(getattr(spec.data, "plot_fit_comparison", None)):
                raise RuntimeError("GPS comparison requires the matching CSI update")

    publish_predictions = getattr(inversion, "publish_fit_predictions", None)
    selected_specs = [
        spec
        for dtype, specs in grouped_specs.items()
        if dtype != "other"
        for spec in specs
    ]
    if callable(publish_predictions):
        prediction_faults = (
            faults
            if faults is None
            or (isinstance(faults, str) and faults.strip().lower() == "all")
            else target_faults
        )
        publish_predictions(
            data_poly=data_poly,
            faults=prediction_faults,
            data_objects=[spec.data for spec in selected_specs],
        )
    else:
        rebuild_diagnostic_synthetics(
            selected_specs,
            target_faults,
            requested_poly=data_poly,
        )

    gps_style = (
        PlotStyle("science", usetex=False, pdf_fonttype=pdf_fonttype, fontsize=gps_fontsize)
        if gps_plot_mode == "legacy" and (pdf_fonttype is not None or gps_fontsize is not None)
        else nullcontext()
    )
    _prepare_fault_traces(
        target_faults,
        color=gps_fault_color,
        linewidth=fault_linewidth,
    )
    with gps_style:
        for spec in grouped_specs["gps"]:
            data = spec.data
            if not plot_data:
                continue
            name = str(getattr(data, "name", "dataset"))
            if gps_plot_mode == "comparison":
                path = outdir / f"gps_{name}_fit_comparison.{file_type}"
                defaults = {
                    "title": gps_title,
                    "remove_direction_labels": remove_direction_labels,
                    "fault_color": gps_fault_color,
                }
                if gps_figsize is not None:
                    defaults["figsize"] = gps_figsize
                if fault_linewidth is not None:
                    defaults["fault_linewidth"] = fault_linewidth
                style_kwargs = {}
                if pdf_fonttype is not None:
                    style_kwargs["pdf_fonttype"] = pdf_fonttype
                if gps_fontsize is not None:
                    style_kwargs["fontsize"] = gps_fontsize
                if style_kwargs:
                    defaults["style_kwargs"] = style_kwargs
                plot_gps_comparison_product(
                    data, vertical=spec.vertical, faults=target_faults,
                    save_path=path, show=show, defaults=defaults, kwargs=gps_kwargs,
                )
                written["gps"].append(path)
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
                locked=("faults", "data", "show"),
                context="plot_data_fits_product(gps)",
            )
            data.plot(
                faults=target_faults,
                data=["data", "synth"],
                show=show,
                **current_gps_kwargs,
            )
            path = _save_geodetic_map(
                data.fig,
                outdir / f"gps_{name}",
                file_type=file_type,
            )
            written["gps"].append(path)

    sar_style = (
        PlotStyle("science", usetex=False, pdf_fonttype=pdf_fonttype, fontsize=sar_fontsize)
        if pdf_fonttype is not None or sar_fontsize is not None
        else nullcontext()
    )
    _prepare_fault_traces(target_faults, color=sar_fault_color)
    with sar_style:
        for spec in grouped_specs["insar"]:
            data = spec.data
            if not plot_data:
                continue
            name = str(getattr(data, "name", "dataset"))
            path = outdir / f"{name}_fit_comparison.{file_type}"
            current_sar_kwargs = _merge_product_plot_kwargs(
                {
                    "cmap": cmap,
                    "antisymmetric": antisymmetric,
                    "share_colorbar": res_use_data_norm,
                    "cbaxis": sar_cbaxis,
                    "figsize": sar_figsize,
                    "render_mode": raster_render_mode,
                    "cell_edge_width": raster_cell_edge_width,
                },
                sar_kwargs,
                locked=(
                    "faults", "save_path", "show",
                    "render_mode", "cell_edge_width",
                ),
                context="plot_data_fits_product(insar)",
            )
            data.plot_fit_comparison(
                faults=target_faults,
                save_path=path,
                show=show,
                **current_sar_kwargs,
            )
            written["insar"].append(path)

        for spec in grouped_specs["opticorr"]:
            data = spec.data
            if not plot_data:
                continue
            name = str(getattr(data, "name", "dataset"))
            path = outdir / f"{name}_fit_comparison.{file_type}"
            current_opticorr_kwargs = _merge_product_plot_kwargs(
                {
                    "cmap": cmap,
                    "antisymmetric": antisymmetric,
                    "share_colorbar": res_use_data_norm,
                    "cbaxis": sar_cbaxis,
                    "figsize": sar_figsize,
                    "render_mode": raster_render_mode,
                    "cell_edge_width": raster_cell_edge_width,
                },
                opticorr_kwargs,
                locked=(
                    "faults", "save_path", "show",
                    "render_mode", "cell_edge_width",
                ),
                context="plot_data_fits_product(opticorr)",
            )
            data.plot_fit_comparison(
                faults=target_faults,
                save_path=path,
                show=show,
                **current_opticorr_kwargs,
            )
            written["opticorr"].append(path)

    for spec in grouped_specs["leveling"]:
        data = spec.data
        name = str(getattr(data, "name", "dataset"))
        if plot_data:
            for item in ("data", "synth"):
                data.write2file(f"{name}_{item}.txt", outDir=str(outdir), data=item)
            _plot_leveling_fit(data, save_dir=outdir, file_type=file_type, show=show)
            written["leveling"].append(outdir / f"{name}_leveling_fit.{file_type}")

    for spec in grouped_specs["crossfaultoffset"]:
        data = spec.data
        name = str(getattr(data, "name", "dataset"))
        if plot_data:
            for item in ("data", "synth"):
                data.write2file(f"{name}_{item}.txt", outDir=str(outdir), data=item)
            _plot_crossfaultoffset_fit(data, save_dir=outdir, file_type=file_type, show=show)
            written["crossfaultoffset"].append(outdir / f"{name}_crossfault_fit.{file_type}")

    for spec in grouped_specs["other"]:
        written["skipped"].append(str(getattr(spec.data, "name", "dataset")))
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
    file_type = normalize_image_format(file_type)
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
    file_type = normalize_image_format(file_type)
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
    file_type = normalize_image_format(file_type)
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
