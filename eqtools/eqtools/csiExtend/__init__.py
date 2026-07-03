from importlib import import_module


_LAZY_EXPORTS = {
    "BayesianAdaptiveTriangularPatches": (
        ".BayesianAdaptiveTriangularPatches",
        "BayesianAdaptiveTriangularPatches",
    ),
    "BayesianTriFaultBase": (".bayesian_perturbation_base", "BayesianTriFaultBase"),
    "PerturbationBase": (".bayesian_perturbation_base", "PerturbationBase"),
    "PerturbationRegistry": (".bayesian_perturbation_base", "PerturbationRegistry"),
    "SharedFaultInfo": (".bayesian_perturbation_base", "SharedFaultInfo"),
    "track_mesh_update": (".bayesian_perturbation_base", "track_mesh_update"),
    "NonlinearGeometrySMCInversion": (
        ".nonlinear_geometry_smc",
        "NonlinearGeometrySMCInversion",
    ),
    "SurfaceForwardResult": (".surface_forward", "SurfaceForwardResult"),
    "compute_multifault_surface_displacement": (
        ".surface_forward",
        "compute_multifault_surface_displacement",
    ),
    "get_fault_summary": (".fault_summary", "get_fault_summary"),
    "get_faults_summary": (".fault_summary", "get_faults_summary"),
    "print_fault_summary": (".fault_summary", "print_fault_summary"),
    "print_faults_summary": (".fault_summary", "print_faults_summary"),
    "project_enu_to_los": (".surface_forward", "project_enu_to_los"),
    "save_surface_forward_h5": (".surface_forward", "save_surface_forward_h5"),
    "save_surface_forward_txt": (".surface_forward", "save_surface_forward_txt"),
    "save_raster_like_geotiff": (".surface_forward", "save_raster_like_geotiff"),
    "save_lonlat_regular_geotiff": (
        ".surface_forward",
        "save_lonlat_regular_geotiff",
    ),
    "show_fault_summary": (".fault_summary", "show_fault_summary"),
    "show_faults_summary": (".fault_summary", "show_faults_summary"),
    "summarize_fault": (".fault_summary", "summarize_fault"),
    "summarize_faults": (".fault_summary", "summarize_faults"),
    "InterseismicKinematicsMixin": (
        ".interseismic_mixin",
        "InterseismicKinematicsMixin",
    ),
    "calculate_interseismic_fields": (
        ".interseismic_fields",
        "calculate_interseismic_fields",
    ),
    "calculate_tectonic_loading_rate": (
        ".interseismic_fields",
        "calculate_tectonic_loading_rate",
    ),
    "DeepSlipLoadingMixin": (
        ".deep_slip_loading_mixin",
        "DeepSlipLoadingMixin",
    ),
    "build_deep_slip_proxy_constraints": (
        ".deep_slip_loading",
        "build_deep_slip_proxy_constraints",
    ),
    "map_shallow_patches_to_deep_top_trace": (
        ".deep_slip_loading",
        "map_shallow_patches_to_deep_top_trace",
    ),
    "calculate_deep_slip_loading_fields": (
        ".deep_slip_loading_fields",
        "calculate_deep_slip_loading_fields",
    ),
    "DataCorrectionReportMixin": (
        ".data_correction_report_mixin",
        "DataCorrectionReportMixin",
    ),
    "DataCorrectionConstraintMixin": (
        ".data_correction_constraints",
        "DataCorrectionConstraintMixin",
    ),
    "DataCorrectionParameterRef": (
        ".data_correction_constraints",
        "DataCorrectionParameterRef",
    ),
    "build_data_correction_equality_matrix": (
        ".data_correction_constraints",
        "build_data_correction_equality_matrix",
    ),
    "resolve_data_correction_parameters": (
        ".data_correction_constraints",
        "resolve_data_correction_parameters",
    ),
    "euler_vector_to_pole": (
        ".data_correction_parameters",
        "euler_vector_to_pole",
    ),
    "euler_pole_to_vector": (
        ".data_correction_parameters",
        "euler_pole_to_vector",
    ),
    "interpret_data_correction_parameters": (
        ".data_correction_parameters",
        "interpret_data_correction_parameters",
    ),
    "normalize_deep_slip_loading_field": (
        ".deep_slip_loading_fields",
        "normalize_deep_slip_loading_field",
    ),
    "extract_inverted_slip": (".interseismic_fields", "extract_inverted_slip"),
    "normalize_interseismic_field": (
        ".interseismic_fields",
        "normalize_interseismic_field",
    ),
    "get_edge_patch_indices": (".patch_indices", "get_edge_patch_indices"),
    "get_patch_centers": (".patch_indices", "get_patch_centers"),
    "get_patches_by_depth": (".patch_indices", "get_patches_by_depth"),
    "get_patches_in_box": (".patch_indices", "get_patches_in_box"),
    "get_patches_in_trace_range": (
        ".patch_indices",
        "get_patches_in_trace_range",
    ),
    "normalize_patch_indices": (".patch_indices", "normalize_patch_indices"),
    "select_patch_indices": (".patch_indices", "select_patch_indices"),
    "buffer_trace": (".trace_ops", "buffer_trace"),
    "clean_trace": (".trace_ops", "clean_trace"),
    "cumulative_distance": (".trace_ops", "cumulative_distance"),
    "extend_trace": (".trace_ops", "extend_trace"),
    "orient_trace": (".trace_ops", "orient_trace"),
    "resample_trace": (".trace_ops", "resample_trace"),
    "reverse_trace": (".trace_ops", "reverse_trace"),
    "simplify_trace": (".trace_ops", "simplify_trace"),
    "smooth_trace": (".trace_ops", "smooth_trace"),
    "trace_length": (".trace_ops", "trace_length"),
    "trim_trace": (".trace_ops", "trim_trace"),
}


__all__ = tuple(_LAZY_EXPORTS)


def __getattr__(name):
    try:
        module_name, attr_name = _LAZY_EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from exc

    value = getattr(import_module(module_name, __name__), attr_name)
    globals()[name] = value
    return value
