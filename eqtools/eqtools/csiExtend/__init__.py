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
    "export_interseismic_loading_pair_table": (
        ".interseismic_loading_pairs",
        "export_interseismic_loading_pair_table",
    ),
    "resolve_interseismic_loading_pairs": (
        ".interseismic_loading_pairs",
        "resolve_interseismic_loading_pairs",
    ),
    "summarize_interseismic_pair_diagnostics": (
        ".interseismic_loading_pairs",
        "summarize_interseismic_pair_diagnostics",
    ),
    "export_interseismic_motion_sense_table": (
        ".interseismic_overrides",
        "export_interseismic_motion_sense_table",
    ),
    "resolve_interseismic_motion_sense": (
        ".interseismic_overrides",
        "resolve_interseismic_motion_sense",
    ),
    "summarize_interseismic_motion_sense_diagnostics": (
        ".interseismic_overrides",
        "summarize_interseismic_motion_sense_diagnostics",
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
        '.reporting.data_correction_report_mixin',
        "DataCorrectionReportMixin",
    ),
    "FigureProductMixin": (
        ".reporting.plot_product_mixin",
        "FigureProductMixin",
    ),
    "plot_data_fits_product": (
        ".reporting.figure_products",
        "plot_data_fits_product",
    ),
    "plot_fault_fields_product": (
        ".reporting.figure_products",
        "plot_fault_fields_product",
    ),
    "plot_interseismic_summary_product": (
        ".reporting.figure_products",
        "plot_interseismic_summary_product",
    ),
    "plot_deep_slip_loading_summary_product": (
        ".reporting.figure_products",
        "plot_deep_slip_loading_summary_product",
    ),
    "data_fit_vectors": (
        ".reporting.fit_statistics",
        "data_fit_vectors",
    ),
    "fit_metrics_from_vectors": (
        ".reporting.fit_statistics",
        "fit_metrics_from_vectors",
    ),
    "solver_fit_metrics": (
        ".reporting.fit_statistics",
        "solver_fit_metrics",
    ),
    "format_fit_statistics_report": (
        ".reporting.fit_statistics",
        "format_fit_statistics_report",
    ),
    "write_fit_statistics_report_files": (
        ".reporting.fit_statistics",
        "write_fit_statistics_report_files",
    ),
    "DataCorrectionConstraintMixin": (
        '.constraints.data_correction_constraints',
        "DataCorrectionConstraintMixin",
    ),
    "DataCorrectionParameterRef": (
        '.constraints.data_correction_constraints',
        "DataCorrectionParameterRef",
    ),
    "build_data_correction_equality_matrix": (
        '.constraints.data_correction_constraints',
        "build_data_correction_equality_matrix",
    ),
    "resolve_data_correction_parameters": (
        '.constraints.data_correction_constraints',
        "resolve_data_correction_parameters",
    ),
    "euler_vector_to_pole": (
        '.observations.data_correction_parameters',
        "euler_vector_to_pole",
    ),
    "euler_pole_to_vector": (
        '.observations.data_correction_parameters',
        "euler_pole_to_vector",
    ),
    "interpret_data_correction_parameters": (
        '.observations.data_correction_parameters',
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
    "get_edge_patch_indices": (".geometry.patch_indices", "get_edge_patch_indices"),
    "get_patch_centers": (".geometry.patch_indices", "get_patch_centers"),
    "get_patches_by_depth": (".geometry.patch_indices", "get_patches_by_depth"),
    "get_patches_in_box": (".geometry.patch_indices", "get_patches_in_box"),
    "get_patches_in_trace_segment": (
        ".geometry.patch_indices",
        "get_patches_in_trace_segment",
    ),
    "get_patches_in_trace_range": (
        ".geometry.patch_indices",
        "get_patches_in_trace_range",
    ),
    "normalize_patch_indices": (".geometry.patch_indices", "normalize_patch_indices"),
    "resolve_trace_marker": (".geometry.patch_indices", "resolve_trace_marker"),
    "sample_trace_markers": (".geometry.patch_indices", "sample_trace_markers"),
    "select_patch_indices": (".geometry.patch_indices", "select_patch_indices"),
    "trace_range_selector_from_markers": (
        ".geometry.patch_indices",
        "trace_range_selector_from_markers",
    ),
    "set_fault_loading_override_selector": (
        ".interseismic_config_tools",
        "set_fault_loading_override_selector",
    ),
    "set_fault_motion_sense_override_selector": (
        ".interseismic_config_tools",
        "set_fault_motion_sense_override_selector",
    ),
    "update_fault_loading_override_from_trace_segment": (
        ".interseismic_config_tools",
        "update_fault_loading_override_from_trace_segment",
    ),
    "update_fault_motion_sense_override_from_trace_segment": (
        ".interseismic_config_tools",
        "update_fault_motion_sense_override_from_trace_segment",
    ),
    "set_fault_loading_region_selector": (
        ".interseismic_config_tools",
        "set_fault_loading_region_selector",
    ),
    "update_fault_loading_region_from_trace_segment": (
        ".interseismic_config_tools",
        "update_fault_loading_region_from_trace_segment",
    ),
    "buffer_trace": (".geometry.trace_ops", "buffer_trace"),
    "clean_trace": (".geometry.trace_ops", "clean_trace"),
    "cumulative_distance": (".geometry.trace_ops", "cumulative_distance"),
    "extend_trace": (".geometry.trace_ops", "extend_trace"),
    "orient_trace": (".geometry.trace_ops", "orient_trace"),
    "point_at_trace_distance": (".geometry.trace_ops", "point_at_trace_distance"),
    "project_points_to_trace": (".geometry.trace_ops", "project_points_to_trace"),
    "resample_trace": (".geometry.trace_ops", "resample_trace"),
    "reverse_trace": (".geometry.trace_ops", "reverse_trace"),
    "sample_trace_distances": (".geometry.trace_ops", "sample_trace_distances"),
    "simplify_trace": (".geometry.trace_ops", "simplify_trace"),
    "smooth_trace": (".geometry.trace_ops", "smooth_trace"),
    "trace_coordinate_intersections": (
        ".geometry.trace_ops",
        "trace_coordinate_intersections",
    ),
    "trace_length": (".geometry.trace_ops", "trace_length"),
    "trim_trace": (".geometry.trace_ops", "trim_trace"),
    "AlongTopLocation": (
        ".geometry.dip_transition_analysis",
        "AlongTopLocation",
    ),
    "DipTransitionSuggestion": (
        ".geometry.dip_transition_analysis",
        "DipTransitionSuggestion",
    ),
    "TopCurvatureAnalysis": (
        ".geometry.dip_transition_analysis",
        "TopCurvatureAnalysis",
    ),
    "analyze_top_curvature": (
        ".geometry.dip_transition_analysis",
        "analyze_top_curvature",
    ),
    "TraceMarker": (".geometry.trace_markers", "TraceMarker"),
    "resolve_trace_markers": (".geometry.trace_markers", "resolve_trace_markers"),
    "TraceOperation": (".geometry.trace_processing", "TraceOperation"),
    "TracePath": (".geometry.trace_processing", "TracePath"),
    "TraceProjection": (".geometry.trace_processing", "TraceProjection"),
    "process_trace": (".geometry.trace_processing", "process_trace"),
    "TraceSegment": (".geometry.trace_io", "TraceSegment"),
    "read_trace": (".geometry.trace_io", "read_trace"),
    "read_trace_segments": (".geometry.trace_io", "read_trace_segments"),
    "write_trace": (".geometry.trace_io", "write_trace"),
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
