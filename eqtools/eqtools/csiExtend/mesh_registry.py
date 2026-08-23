"""
Mesh method registry — static metadata for mesh generation methods.

Provides a central lookup for parameter classification (spec, replay,
geometry-mutating) and Bayesian forbidden rules.  Used by
``PerturbationBase.record_mesh_call()`` and Bayesian config normalization.

Design: simple module-level dict, no classes, no decorators, no MRO traversal.
See ``docs/MESH_REGISTRY_DESIGN.md`` for rationale.
"""

_MESH_METHODS = {}


def register(method_name, meta_info):
    _MESH_METHODS[method_name] = meta_info


def get_meta(method_name):
    return _MESH_METHODS.get(method_name)


def get_replayable_keys(method_name):
    meta = _MESH_METHODS.get(method_name)
    if meta is None:
        return None
    return (
        meta['spec_keys']
        + meta.get('replay_keys', [])
        + meta.get('geometry_mutating_keys', [])
    )


def get_bayesian_forbidden(method_name):
    meta = _MESH_METHODS.get(method_name)
    if meta is None:
        return {}
    return meta.get('bayesian_forbidden', {})


def get_bayesian_replay_contract(method_name):
    """Return the sampling-time state contract for a mesh method, if any.

    The contract is descriptive metadata used by Bayesian preflight.  It does
    not execute a mesh method or change its ordinary interactive behaviour.
    """
    meta = _MESH_METHODS.get(method_name)
    if meta is None:
        return None
    return meta.get('bayesian_replay')


def is_registered(method_name):
    return method_name in _MESH_METHODS


# ---------------------------------------------------------------------------
# Static registrations
# ---------------------------------------------------------------------------

register('generate_and_deform_mesh', {
    'spec_keys': ['top_size', 'bottom_size', 'num_segments', 'disct_z',
                  'rotation_angle', 'bias', 'min_dz', 'projection',
                  'field_size_dict', 'mesh_func', 'tolerance'],
    'replay_keys': ['remap', 'use_current_mesh'],
    'geometry_mutating_keys': ['bottom_norm_offset'],
    'bayesian_forbidden': {
        'remap': True,
        'use_current_mesh': True,
        'bottom_norm_offset': 'not_none',
    },
    # Bayesian proposals must reuse the mapping prepared before target
    # construction.  These parameters define the structured grid interpreted
    # by the stored per-vertex parametric coordinates and therefore must stay
    # consistent during fixed-topology replay.
    'bayesian_replay': {
        'state': 'prepared_parametric_mapping',
        'mapping_parameter_keys': [
            'num_segments', 'disct_z', 'bias', 'min_dz', 'projection',
        ],
    },
})

register('generate_simple_mesh', {
    'spec_keys': ['disct_z', 'bias', 'min_dz', 'use_depth_only'],
    'bayesian_forbidden': {},
})

register('generate_simple_multilayer_mesh', {
    'spec_keys': ['disct_z', 'bias'],
    'bayesian_forbidden': {},
})

register('rebuild_simple_mesh', {
    'spec_keys': ['disct_z', 'bias', 'min_dz', 'segs', 'top_tolerance',
                  'bottom_tolerance', 'lonlat', 'buffer_depth', 'sort_axis',
                  'sort_order', 'use_trace', 'discretized'],
    'bayesian_forbidden': {},
})

register('generate_mesh', {
    'spec_keys': ['top_size', 'bottom_size', 'field_size_dict', 'segments_dict',
                  'mesh_algorithm', 'optimize_method', 'mesh_func'],
    'geometry_mutating_keys': ['smooth_coords', 'smooth_window', 'smooth_method'],
    'bayesian_forbidden': {'smooth_coords': 'truthy'},
})

register('generate_multilayer_mesh', {
    'spec_keys': ['mesh_func', 'top_size', 'bottom_size', 'field_size_dict',
                  'mesh_algorithm', 'optimize_method', 'nodes_on_layers',
                  'occ_method', 'sparse_points', 'sparse_factor', 'lonlat'],
    'geometry_mutating_keys': ['smooth_coords', 'smooth_window', 'smooth_method'],
    'bayesian_forbidden': {'smooth_coords': 'truthy'},
})

register('generate_layered_mesh', {
    'spec_keys': ['num_layers', 'layer_depths', 'use_profile_depths',
                  'nodes_on_layers', 'mesh_func', 'top_size', 'bottom_size',
                  'field_size_dict', 'mesh_algorithm', 'optimize_method',
                  'occ_method', 'sparse_points', 'sparse_factor'],
    'geometry_mutating_keys': ['smooth_layers', 'smooth_window', 'smooth_method'],
    'bayesian_forbidden': {'smooth_layers': 'truthy'},
})
