import contextlib
import glob
import importlib
import logging
import os
import sys
import warnings
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .edcmp_coord import csi_obs_to_edcmp, csi_source_to_edcmp, edcmp_disp_to_csi

logger = logging.getLogger(__name__)

VALID_EDCMP_ENGINES = ("auto", "exe", "ctypes")
DEFAULT_EDCMP_FALLBACK_ENGINES = ("exe",)

_MODEL_CACHE = {}
_DLL_DIR_HANDLES = []
_SHARED_MEMORY_METADATA = None


def _init_shared_memory_worker(metadata):
    """ProcessPoolExecutor initializer: set shared memory metadata in worker."""
    global _SHARED_MEMORY_METADATA
    _SHARED_MEMORY_METADATA = metadata


@dataclass
class EdcmpOptions:
    """EDCMP Green's function configuration.

    Pass as ``options=`` to ``buildGFs(method='edcmp', options=...)``.
    Use ``EdcmpOptions.describe_options()`` to see all fields.
    """
    engine: str = "auto"
    fallback_engines: Optional[List[str]] = None
    module_dir: Optional[str] = None
    allow_triangle: bool = True
    triangle_rect_dx_km: float = 0.1
    triangle_rect_dy_km: float = 0.1
    grn_dir: str = "edgrnfcts"
    output_dir: str = "edcmpgrns"
    workdir: str = "edcmp_ecat"
    layered_model: bool = True
    n_jobs: Optional[int] = None
    cleanup_inp: bool = False
    force_recompute: bool = True

    _FIELD_DESCRIPTIONS = {
        'engine':              ('str',              'auto',      'Computation engine: "auto", "exe", or "ctypes"'),
        'fallback_engines':    ('list[str] | None', 'None',      'Fallback engine chain when engine="auto" (e.g. ["exe"])'),
        'module_dir':          ('str | None',       'None',      'Directory containing edcmp4py_ctypes module'),
        'allow_triangle':      ('bool',             'True',      'Allow triangle patches (decomposed to rectangles)'),
        'triangle_rect_dx_km': ('float',            '0.1',       'Rectangle decomposition dx in km (for triangle patches)'),
        'triangle_rect_dy_km': ('float',            '0.1',       'Rectangle decomposition dy in km (for triangle patches)'),
        'grn_dir':             ('str',              'edgrnfcts', "EDGRN Green's function directory (relative to workdir)"),
        'output_dir':          ('str',              'edcmpgrns', 'EDCMP output directory (relative to workdir)'),
        'workdir':             ('str',              'edcmp_ecat','Working directory for all intermediate files'),
        'layered_model':       ('bool',             'True',      'Use layered earth model (requires EDGRN tables)'),
        'n_jobs':              ('int | None',       'None',      'Number of parallel workers (None = auto-detect)'),
        'cleanup_inp':         ('bool',             'False',     'Remove intermediate .inp files after computation'),
        'force_recompute':     ('bool',             'True',      'Recompute even if output files exist'),
    }

    def __post_init__(self):
        self.engine = normalize_edcmp_engine(self.engine)
        if self.fallback_engines is not None:
            self.fallback_engines = normalize_edcmp_fallback_engines(
                self.fallback_engines
            )
        if self.triangle_rect_dx_km <= 0:
            raise ValueError(
                f"triangle_rect_dx_km must be positive, got {self.triangle_rect_dx_km}"
            )
        if self.triangle_rect_dy_km <= 0:
            raise ValueError(
                f"triangle_rect_dy_km must be positive, got {self.triangle_rect_dy_km}"
            )
        if self.n_jobs is not None and int(self.n_jobs) < 1:
            raise ValueError(f"n_jobs must be >= 1, got {self.n_jobs}")

    @classmethod
    def from_kwargs(cls, **kwargs):
        """Build from a dict.  Only canonical field names are accepted."""
        valid = set(cls.__dataclass_fields__)
        unknown = set(kwargs) - valid
        if unknown:
            warnings.warn(
                f"Unknown EdcmpOptions keys ignored: {sorted(unknown)}. "
                f"Valid keys: {sorted(valid)}",
                UserWarning, stacklevel=2,
            )
        return cls(**{k: v for k, v in kwargs.items() if k in valid})

    @classmethod
    def to_commented_map(cls, instance=None):
        """Return a ruamel.yaml CommentedMap with inline comments."""
        from ruamel.yaml.comments import CommentedMap
        obj = instance or cls()
        cm = CommentedMap()
        for name in cls.__dataclass_fields__:
            cm[name] = getattr(obj, name)
            desc = cls._FIELD_DESCRIPTIONS.get(name)
            if desc:
                cm.yaml_add_eol_comment(desc[2], name)
        return cm

    @classmethod
    def describe_yaml(cls):
        """Return a YAML string showing all options with inline comments."""
        from ruamel.yaml import YAML
        from io import StringIO
        y = YAML()
        y.indent(mapping=2, sequence=4, offset=2)
        buf = StringIO()
        y.dump(cls.to_commented_map(), buf)
        return buf.getvalue()

    @classmethod
    def describe_options(cls):
        """Print a human-readable summary of all available EDCMP options."""
        lines = ["EdcmpOptions — EDCMP Green's function configuration", "=" * 55]
        for name in cls.__dataclass_fields__:
            desc = cls._FIELD_DESCRIPTIONS.get(name, ('', '', ''))
            lines.append(f"  {name:.<30s} type={desc[0]}, default={desc[1]}")
            if desc[2]:
                lines.append(f"    {desc[2]}")
        lines.append("")
        lines.append("Pass as: fault.buildGFs(data, method='edcmp', options=EdcmpOptions(...))")
        lines.append("")
        lines.append("YAML example:")
        lines.append("  options:")
        for line in cls.describe_yaml().splitlines():
            lines.append(f"    {line}")
        print("\n".join(lines))


@contextlib.contextmanager
def _optional_sys_path(module_dir):
    if not module_dir:
        yield
        return

    module_dir = os.path.abspath(module_dir)
    inserted = False
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
        inserted = True
    try:
        yield
    finally:
        if inserted:
            try:
                sys.path.remove(module_dir)
            except ValueError:
                pass


def normalize_edcmp_engine(engine):
    engine = (engine or "auto").lower()
    if engine not in VALID_EDCMP_ENGINES:
        raise ValueError(
            f"Invalid EDCMP engine '{engine}'. Allowed values: {VALID_EDCMP_ENGINES}"
        )
    return engine


def normalize_edcmp_fallback_engines(fallback_engines):
    if fallback_engines is None:
        return list(DEFAULT_EDCMP_FALLBACK_ENGINES)

    if isinstance(fallback_engines, str):
        fallback_engines = [item.strip() for item in fallback_engines.split(",") if item.strip()]
    elif isinstance(fallback_engines, (tuple, list)):
        fallback_engines = list(fallback_engines)
    else:
        raise ValueError(
            "edcmp_fallback_engines must be None, a comma-separated string, or a list/tuple"
        )

    normalized = []
    for engine in fallback_engines:
        name = normalize_edcmp_engine(engine)
        if name == "auto":
            raise ValueError("edcmp_fallback_engines cannot contain 'auto'")
        if name not in normalized:
            normalized.append(name)
    return normalized


def _backend_module_dir(module_dir=None):
    return module_dir or os.environ.get("EDCMP4PY_MODULE_DIR")


def _register_windows_dll_dir(path):
    if os.name != "nt" or not path or not os.path.isdir(path):
        return

    path = os.path.abspath(path)
    for handle in _DLL_DIR_HANDLES:
        if getattr(handle, "_codex_path", None) == path:
            return

    try:
        handle = os.add_dll_directory(path)
    except (AttributeError, FileNotFoundError, OSError):
        return

    handle._codex_path = path
    _DLL_DIR_HANDLES.append(handle)


def _ensure_windows_ctypes_runtime_dirs(module_dir=None):
    if os.name != "nt":
        return

    candidate_dirs = []
    module_dir = _backend_module_dir(module_dir)
    if module_dir:
        candidate_dirs.extend(
            [
                module_dir,
                os.path.join(module_dir, "bin"),
            ]
        )

    conda_roots = []
    for root in (sys.prefix, getattr(sys, "base_prefix", None), os.environ.get("CONDA_PREFIX")):
        if root and root not in conda_roots:
            conda_roots.append(root)

    for root in conda_roots:
        candidate_dirs.extend(
            [
                os.path.join(root, "Library", "mingw-w64", "bin"),
                os.path.join(root, "Library", "bin"),
            ]
        )
        candidate_dirs.extend(
            glob.glob(os.path.join(root, "envs", "*", "Library", "mingw-w64", "bin"))
        )
        candidate_dirs.extend(glob.glob(os.path.join(root, "pkgs", "*", "Library", "mingw-w64", "bin")))

    seen = set()
    for path in candidate_dirs:
        if not path:
            continue
        path = os.path.abspath(path)
        if path in seen:
            continue
        seen.add(path)
        _register_windows_dll_dir(path)


def _get_csi_bin_dir():
    """Return the path to csi/bin/ directory, or None if not found."""
    import importlib.util

    csi_module = sys.modules.get("csi")
    if csi_module is not None and getattr(csi_module, "__path__", None):
        csi_dir = csi_module.__path__[0]
    else:
        spec = importlib.util.find_spec("csi")
        if spec is None or not spec.submodule_search_locations:
            return None
        csi_dir = spec.submodule_search_locations[0]

    bin_dir = os.path.join(csi_dir, "bin")
    if os.path.isdir(bin_dir):
        return bin_dir
    return None


def _import_edcmp4py_module(engine, module_dir=None):
    module_dir = _backend_module_dir(module_dir)
    module_name = {
        "ctypes": "edcmp4py_ctypes",
    }.get(engine)
    if module_name is None:
        raise ValueError(f"Engine '{engine}' is not an in-memory edcmp4py engine")

    if engine == "ctypes":
        _ensure_windows_ctypes_runtime_dirs(module_dir=module_dir)

    # 1. User-specified directory (module_dir or EDCMP4PY_MODULE_DIR)
    if module_dir:
        with _optional_sys_path(module_dir):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                pass

    # 2. csi/bin/ directory (packaged default location)
    csi_bin_dir = _get_csi_bin_dir()
    if csi_bin_dir:
        if engine == "ctypes":
            # Register the platform-specific subdirectory for DLL loading
            if sys.platform.startswith("win"):
                _register_windows_dll_dir(os.path.join(csi_bin_dir, "windows"))
            _ensure_windows_ctypes_runtime_dirs(module_dir=csi_bin_dir)
        with _optional_sys_path(csi_bin_dir):
            try:
                return importlib.import_module(module_name)
            except ImportError:
                pass

    # 3. Global sys.path fallback
    return importlib.import_module(module_name)


def _engine_is_available(engine, module_dir=None):
    engine = normalize_edcmp_engine(engine)

    if engine == "exe":
        from .EDGRNcmp import get_edcmp_bin

        try:
            bin_dir = get_edcmp_bin()
        except Exception:
            return False
        exe_name = "edcmp.exe" if sys.platform.startswith("win") else "edcmp"
        return os.path.isfile(os.path.join(bin_dir, exe_name))

    if engine == "auto":
        return True

    try:
        _import_edcmp4py_module(engine, module_dir=module_dir)
    except Exception:
        return False
    return True


def resolve_edcmp_engine(engine="auto", fallback_engines=None, module_dir=None):
    engine = normalize_edcmp_engine(engine)

    if engine != "auto":
        if not _engine_is_available(engine, module_dir=module_dir):
            raise RuntimeError(
                f"Requested EDCMP engine '{engine}' is not available. "
                f"Check installation and EDCMP4PY_MODULE_DIR."
            )
        return engine

    chain = ["ctypes"]
    for fallback in normalize_edcmp_fallback_engines(fallback_engines):
        if fallback not in chain:
            chain.append(fallback)

    for candidate in chain:
        if _engine_is_available(candidate, module_dir=module_dir):
            return candidate

    raise RuntimeError(
        "No usable EDCMP engine is available. Tried: "
        + ", ".join(chain)
        + ". Check edcmp.exe packaging or EDCMP4PY_MODULE_DIR."
    )


_JUNCTION_CLEANUP = []
_CLEANUP_REGISTERED = False


def _register_junction_cleanup():
    global _CLEANUP_REGISTERED
    if _CLEANUP_REGISTERED:
        return
    _CLEANUP_REGISTERED = True
    import atexit

    def _cleanup():
        import shutil as _shutil
        for p in _JUNCTION_CLEANUP:
            try:
                if _is_junction(p):
                    os.rmdir(p)
                elif os.path.isdir(p):
                    _shutil.rmtree(p, ignore_errors=True)
            except OSError:
                pass
    atexit.register(_cleanup)


def _is_junction(path):
    if os.name != "nt":
        return False
    import ctypes
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    return attrs != -1 and (attrs & 0x400) != 0


def _try_short_path_dir(grn_dir_abs):
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        n = ctypes.windll.kernel32.GetShortPathNameW(grn_dir_abs, buf, len(buf))
        if n and buf.value and buf.value.isascii():
            return buf.value
    except Exception:
        pass
    return None


def _try_junction(grn_dir_abs):
    try:
        import _winapi
        import hashlib
        import tempfile
        dir_hash = hashlib.sha256(
            os.path.normcase(grn_dir_abs).encode("utf-8")
        ).hexdigest()[:16]
        jct_root = os.path.join(tempfile.gettempdir(), "edcmp_grn_jct")
        os.makedirs(jct_root, exist_ok=True)
        jct_path = os.path.join(jct_root, dir_hash)
        if os.path.isdir(jct_path):
            if _is_junction(jct_path):
                return jct_path
            os.rmdir(jct_path)
        _winapi.CreateJunction(grn_dir_abs, jct_path)
        _JUNCTION_CLEANUP.append(jct_path)
        _register_junction_cleanup()
        logger.debug("Created junction %s -> %s", jct_path, grn_dir_abs)
        return jct_path
    except Exception as exc:
        logger.debug("Junction creation failed: %s", exc)
        return None


def _try_temp_copy_dir(grn_dir_abs):
    import hashlib
    import shutil
    import tempfile
    dir_hash = hashlib.sha256(
        os.path.normcase(grn_dir_abs).encode("utf-8")
    ).hexdigest()[:16]
    tmp_dir = os.path.join(tempfile.gettempdir(), "edcmp_grn_cache", dir_hash)
    os.makedirs(tmp_dir, exist_ok=True)
    for fname in ("edgrnhs.ss", "edgrnhs.ds", "edgrnhs.cl"):
        src = os.path.join(grn_dir_abs, fname)
        dst = os.path.join(tmp_dir, fname)
        if not os.path.isfile(src):
            continue
        if not os.path.isfile(dst) or os.path.getmtime(src) > os.path.getmtime(dst):
            dst_tmp = f"{dst}.{os.getpid()}.tmp"
            shutil.copy2(src, dst_tmp)
            os.replace(dst_tmp, dst)
    _JUNCTION_CLEANUP.append(tmp_dir)
    _register_junction_cleanup()
    return tmp_dir


def _ascii_safe_grn_dir(grn_dir_abs):
    """Return an ASCII-safe directory path for Fortran I/O on Windows.

    Strategy: (1) already ASCII → pass through, (2) 8.3 short path,
    (3) NTFS junction (zero-copy), (4) atomic file copy to temp dir.
    """
    if grn_dir_abs.isascii():
        return grn_dir_abs
    short = _try_short_path_dir(grn_dir_abs)
    if short:
        return short
    jct = _try_junction(grn_dir_abs)
    if jct:
        return jct
    return _try_temp_copy_dir(grn_dir_abs)


def _resolve_grn_dir(grn_dir, workdir):
    if grn_dir is None:
        raise ValueError("grn_dir must not be None for in-memory EDCMP engines")

    workdir = os.path.abspath(workdir or ".")
    if os.path.isabs(grn_dir):
        grn_dir_abs = grn_dir
    else:
        grn_dir_abs = os.path.join(workdir, grn_dir)

    if not os.path.isdir(grn_dir_abs):
        raise FileNotFoundError(f"Green's-function directory '{grn_dir_abs}' does not exist")

    if os.name == "nt":
        grn_dir_abs = _ascii_safe_grn_dir(grn_dir_abs)

    return grn_dir_abs


def _load_inmemory_backend(engine, grn_dir, workdir, module_dir=None, use_shared_memory=False):
    """
    Load EDCMP backend model.

    Parameters
    ----------
    engine : str
        Engine type
    grn_dir : str
        Green's function directory
    workdir : str
        Working directory
    module_dir : str, optional
        Module directory
    use_shared_memory : bool
        If True, try to load from shared memory first

    Returns
    -------
    model : object
        Loaded EDCMP model
    """
    engine = normalize_edcmp_engine(engine)
    if engine != "ctypes":
        raise ValueError(f"Engine '{engine}' is not an in-memory engine")

    module_dir = _backend_module_dir(module_dir)
    grn_dir_abs = _resolve_grn_dir(grn_dir, workdir)
    cache_key = (engine, os.path.abspath(module_dir) if module_dir else "", grn_dir_abs)

    # Check shared memory first (worker processes)
    if use_shared_memory and _SHARED_MEMORY_METADATA is not None:
        try:
            from .shared_memory_backend import attach_shared_greens
            model = attach_shared_greens(_SHARED_MEMORY_METADATA, module_dir=module_dir)
            logger.debug(f"Loaded {engine} model from shared memory")
            return model
        except Exception as e:
            logger.warning(f"Failed to load from shared memory: {e}, falling back to normal loading")

    # Check process-local cache
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    # Load from files
    grnss = os.path.join(grn_dir_abs, "edgrnhs.ss")
    grnds = os.path.join(grn_dir_abs, "edgrnhs.ds")
    grncl = os.path.join(grn_dir_abs, "edgrnhs.cl")
    for path in (grnss, grnds, grncl):
        if not os.path.isfile(path):
            raise FileNotFoundError(f"Missing EDGRN output file: {path}")

    module = _import_edcmp4py_module(engine, module_dir=module_dir)

    model = module.EdcmpLayeredCtypes()
    info = model.load_greenfunctions(grnss, grnds, grncl)
    ierr = info[-1]

    if ierr != 0:
        raise RuntimeError(
            f"Failed to load EDGRN tables for engine '{engine}' from '{grn_dir_abs}' (ierr={ierr})"
        )

    _MODEL_CACHE[cache_key] = model
    return model


def _as_contiguous_f64(value):
    return np.ascontiguousarray(np.atleast_1d(np.asarray(value, dtype=np.float64)))


def _source_arrays(source_params):
    xs, ys, zs, width, length, strike, dip, mean_x, mean_y = source_params

    xs = _as_contiguous_f64(xs)
    ys = _as_contiguous_f64(ys)
    zs = _as_contiguous_f64(zs)
    if xs.shape != ys.shape or xs.shape != zs.shape:
        raise ValueError("source_params xs/ys/zs must have matching shapes")

    nsrc = xs.size

    def _expand(value):
        arr = np.asarray(value, dtype=np.float64)
        if arr.ndim == 0:
            return np.full(nsrc, float(arr), dtype=np.float64)
        arr = _as_contiguous_f64(arr)
        if arr.size != nsrc:
            raise ValueError("Per-source parameter arrays must match xs/ys/zs length")
        return arr

    width = _expand(width)
    length = _expand(length)
    strike = _expand(strike)
    dip = _expand(dip)

    return xs, ys, zs, width, length, strike, dip, float(mean_x), float(mean_y)


def _receiver_arrays(data, mean_x, mean_y):
    return csi_obs_to_edcmp(data.x, data.y, mean_x, mean_y)


def _check_nwarn(nwarn, context=""):
    if nwarn < 0:
        raise RuntimeError(
            f"EDCMP Fortran error (nwarn={nwarn}){': ' + context if context else ''}. "
            f"nwarn=-999 means Green's functions not loaded."
        )


def _compute_single_disp(model, engine, slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake, xrec, yrec):
    nrec = int(len(xrec))
    if engine == "ctypes" and hasattr(model, "compute_single_disp"):
        disp, nwarn = model.compute_single_disp(
            slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake, nrec, xrec, yrec
        )
        _check_nwarn(nwarn, "compute_single_disp")
    else:
        result = model.compute_single(
            slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake, nrec, xrec, yrec
        )
        disp, nwarn = result[0], result[-1]
        _check_nwarn(nwarn, "compute_single")

    disp = np.asarray(disp, dtype=np.float64)
    return edcmp_disp_to_csi(disp)


def _compute_bundle_disp(model, engine, slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake, xrec, yrec):
    slip = _as_contiguous_f64(slip)
    nsrc = slip.size
    if nsrc == 0:
        return np.zeros((len(xrec), 3), dtype=np.float64)

    arrays = {"xs0": xs0, "ys0": ys0, "zs0": zs0, "flen": flen, "fwid": fwid, "fstrike": fstrike, "fdip": fdip, "frake": frake}
    for name, arr in arrays.items():
        if np.asarray(arr).size != nsrc:
            raise ValueError(f"{name} must have length {nsrc}, got {np.asarray(arr).size}")
    xs0, ys0, zs0 = _as_contiguous_f64(xs0), _as_contiguous_f64(ys0), _as_contiguous_f64(zs0)
    flen, fwid = _as_contiguous_f64(flen), _as_contiguous_f64(fwid)
    fstrike, fdip, frake = _as_contiguous_f64(fstrike), _as_contiguous_f64(fdip), _as_contiguous_f64(frake)

    nrec = int(len(xrec))
    if engine == "ctypes" and hasattr(model, "compute_bundle_sum_disp"):
        disp, nwarn = model.compute_bundle_sum_disp(
            slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake, nrec, xrec, yrec
        )
        _check_nwarn(nwarn, "compute_bundle_sum_disp")
        disp = np.asarray(disp, dtype=np.float64)
        return edcmp_disp_to_csi(disp)

    if engine == "ctypes" and hasattr(model, "compute_bundle_sum") and nsrc > 1:
        result = model.compute_bundle_sum(
            slip, xs0, ys0, zs0, flen, fwid, fstrike, fdip, frake, nrec, xrec, yrec
        )
        disp, nwarn = result[0], result[-1]
        _check_nwarn(nwarn, "compute_bundle_sum")
        disp = np.asarray(disp, dtype=np.float64)
        return edcmp_disp_to_csi(disp)

    disp = np.zeros((nrec, 3), dtype=np.float64)
    for i in range(nsrc):
        disp += _compute_single_disp(
            model,
            engine,
            slip[i],
            xs0[i],
            ys0[i],
            zs0[i],
            flen[i],
            fwid[i],
            fstrike[i],
            fdip[i],
            frake[i],
            xrec,
            yrec,
        )
    return disp


def compute_inmemory_edcmp_greens(
    data,
    source_params,
    slip,
    engine,
    grn_dir,
    workdir=".",
    module_dir=None,
    use_shared_memory=False,
):
    model = _load_inmemory_backend(
        engine,
        grn_dir=grn_dir,
        workdir=workdir,
        module_dir=module_dir,
        use_shared_memory=use_shared_memory
    )

    xs, ys, zs, width, length, strike, dip, mean_x, mean_y = _source_arrays(source_params)
    xrec, yrec = _receiver_arrays(data, mean_x, mean_y)

    ss = np.zeros((len(xrec), 3), dtype=np.float64)
    ds = np.zeros((len(xrec), 3), dtype=np.float64)
    ts = np.zeros((len(xrec), 3), dtype=np.float64)

    if len(slip) >= 3 and slip[2]:
        logger.warning(
            "EDCMP engine '%s' does not compute tensile-slip Green's functions; returning zeros for tensile component.",
            engine,
        )

    xs0, ys0 = csi_source_to_edcmp(xs, ys)
    zs0 = zs

    if slip[0]:
        ss = _compute_bundle_disp(
            model,
            engine,
            np.ones(xs.size, dtype=np.float64),
            xs0,
            ys0,
            zs0,
            length,
            width,
            strike,
            dip,
            np.zeros(xs.size, dtype=np.float64),
            xrec,
            yrec,
        )

    if slip[1]:
        ds = _compute_bundle_disp(
            model,
            engine,
            np.ones(xs.size, dtype=np.float64),
            xs0,
            ys0,
            zs0,
            length,
            width,
            strike,
            dip,
            np.full(xs.size, 90.0, dtype=np.float64),
            xrec,
            yrec,
        )

    return ss, ds, ts


def compute_inmemory_edcmp_forward(
    data,
    source_params,
    slip,
    engine,
    grn_dir,
    workdir=".",
    module_dir=None,
):
    model = _load_inmemory_backend(engine, grn_dir=grn_dir, workdir=workdir, module_dir=module_dir)
    xs, ys, zs, width, length, strike, dip, mean_x, mean_y = _source_arrays(source_params)
    xrec, yrec = _receiver_arrays(data, mean_x, mean_y)

    disp = np.zeros((len(xrec), 3), dtype=np.float64)

    ss, ds, ts = np.asarray(slip, dtype=np.float64)
    if ts != 0.0:
        logger.warning(
            "EDCMP engine '%s' forward calculation ignores tensile slip; only strike-slip and dip-slip are used.",
            engine,
        )

    slip_total = float(np.hypot(ss, ds))
    if slip_total == 0.0:
        return disp

    rake = float(np.degrees(np.arctan2(ds, ss)))

    xs0, ys0 = csi_source_to_edcmp(xs, ys)

    disp = _compute_bundle_disp(
        model,
        engine,
        np.full(xs.size, slip_total, dtype=np.float64),
        xs0,
        ys0,
        zs,
        length,
        width,
        strike,
        dip,
        np.full(xs.size, rake, dtype=np.float64),
        xrec,
        yrec,
    )

    return disp
