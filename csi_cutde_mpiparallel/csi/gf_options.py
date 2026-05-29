"""
Centralized Green's function options registry.

Provides a single entry point for resolving method-specific configuration:

    from csi.gf_options import resolve_gf_options, describe_gf_options

    # From Python API
    opts = resolve_gf_options('edcmp', EdcmpOptions(engine='ctypes'))
    opts = resolve_gf_options('edcmp', {'engine': 'ctypes', 'n_jobs': 8})
    opts = resolve_gf_options('okada', None)  # → None (no config needed)

    # Discover available options
    describe_gf_options('edcmp')
    describe_gf_options()  # all methods
"""

import warnings

from .edgrn_edcmp.edcmp_backends import EdcmpOptions
from .psgrn_pscmp.pscmp_options import PscmpOptions

VALID_GF_METHODS = frozenset({
    'okada', 'ok92', 'meade',
    'edks',
    'pscmp', 'psgrn',
    'edcmp', 'edgrn',
    'cutde',
    'empty',
    'homogeneous', 'volume',
})

GF_OPTIONS_REGISTRY = {
    'edcmp': EdcmpOptions,
    'edgrn': EdcmpOptions,
    'pscmp': PscmpOptions,
    'psgrn': PscmpOptions,
    'okada': None, 'ok92': None, 'meade': None,
    'edks': None, 'cutde': None,
    'homogeneous': None, 'empty': None, 'volume': None,
}


def resolve_gf_options(method, options=None):
    """Resolve *options* into the correct Options object for *method*.

    Parameters
    ----------
    method : str
        GF method name (e.g. 'edcmp', 'pscmp', 'okada').
    options : dict | EdcmpOptions | PscmpOptions | None
        - ``None`` → default Options (or None for methods without config).
        - ``dict`` → converted via ``Options.from_kwargs(**options)``.
        - Options instance → returned as-is (type-checked).

    Returns
    -------
    EdcmpOptions | PscmpOptions | None

    Raises
    ------
    ValueError
        If *method* is unknown.
    TypeError
        If *options* type doesn't match the expected Options class.
    """
    method_lower = method.lower()
    if method_lower not in GF_OPTIONS_REGISTRY:
        raise ValueError(
            f"Unknown GF method '{method}'. "
            f"Valid methods: {sorted(VALID_GF_METHODS)}"
        )

    cls = GF_OPTIONS_REGISTRY[method_lower]

    if cls is None:
        if options:
            warnings.warn(
                f"Method '{method}' does not accept options; "
                f"the provided options will be ignored.",
                UserWarning, stacklevel=2,
            )
        return None

    if options is None:
        return cls()

    if isinstance(options, cls):
        return options

    if isinstance(options, dict):
        return cls.from_kwargs(**options)

    raise TypeError(
        f"options for method '{method}' must be {cls.__name__}, dict, or None, "
        f"got {type(options).__name__}"
    )


def describe_gf_options(method=None, format='text'):
    """Print available options for Green's function methods.

    Parameters
    ----------
    method : str, optional
        If given, print options for that method only.
        If None, print options for all configurable methods.
    format : str
        ``'text'`` (default) for a human-readable table,
        ``'yaml'`` for a YAML example with inline comments.
    """
    if format not in ('text', 'yaml'):
        raise ValueError(f"format must be 'text' or 'yaml', got '{format}'")

    if method is not None:
        method_lower = method.lower()
        cls = GF_OPTIONS_REGISTRY.get(method_lower)
        if cls is None:
            print(f"Method '{method}' has no configurable options (uses defaults).")
        elif format == 'yaml':
            print(f"# {cls.__name__} - use under 'options:' in your YAML config")
            print(cls.describe_yaml())
        else:
            cls.describe_options()
        return

    seen = set()
    for name, cls in GF_OPTIONS_REGISTRY.items():
        if cls is not None and cls not in seen:
            seen.add(cls)
            if format == 'yaml':
                print(f"# {cls.__name__} - use under 'options:' in your YAML config")
                print(cls.describe_yaml())
            else:
                cls.describe_options()
                print()

    print(f"Available GF methods: {sorted(VALID_GF_METHODS)}")
    simple = sorted(k for k, v in GF_OPTIONS_REGISTRY.items() if v is None)
    print(f"Methods without configurable options: {simple}")
