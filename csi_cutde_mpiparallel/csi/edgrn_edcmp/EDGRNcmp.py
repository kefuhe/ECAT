import os
import sys
import subprocess
import shutil
import numpy as np
from .edcmp_config import EDCMPConfig, EDCMPParameters, RectangularSource
from .edcmp_coord import csi_obs_to_edcmp, csi_source_to_edcmp, edcmp_disp_to_csi
from .tri2rectpoints import patch_local2d_inv, patch_local2d, triangle_to_rectangles

def get_edcmp_bin():
    """
    Return the absolute path to the EDCMP binary directory under the csi package,
    regardless of the current module location.

    Returns
    -------
    bin_dir : str
        Absolute path to the EDCMP binary directory for the current platform.
    """
    import importlib.util
    # Find the csi package location
    csi_module = sys.modules.get("csi")
    if csi_module is not None and getattr(csi_module, "__path__", None):
        csi_dir = csi_module.__path__[0]
    else:
        spec = importlib.util.find_spec("csi")
        if spec is None or not spec.submodule_search_locations:
            raise ImportError("Cannot find csi package. Please ensure csi is installed.")
        csi_dir = spec.submodule_search_locations[0]
    if sys.platform.startswith('win'):
        bin_dir = os.path.join(csi_dir, 'bin', 'windows')
    elif sys.platform.startswith('linux'):
        bin_dir = os.path.join(csi_dir, 'bin', 'ubuntu20.04')
    else:
        raise RuntimeError('Unsupported platform: ' + sys.platform)
    return bin_dir

def read_disp_bin(filename, nrec, ncol):
    """
    Read EDCMP binary displacement output and convert to CSI convention.

    Returns
    -------
    disp_csi : np.ndarray, shape (nrec, 3)
        Displacement in CSI convention (dx=East, dy=North, dz=Up).
    """
    with open(filename, 'rb') as f:
        # Read nrec
        f.read(4)
        nrec_bin = np.fromfile(f, dtype=np.int32, count=1)[0]
        f.read(4)
        # Read xrec
        f.read(4)
        xrec = np.fromfile(f, dtype=np.float64, count=nrec_bin)
        f.read(4)
        # Read yrec
        f.read(4)
        yrec = np.fromfile(f, dtype=np.float64, count=nrec_bin)
        f.read(4)
        # Read disp
        f.read(4)
        disp = np.fromfile(f, dtype=np.float64, count=nrec_bin*ncol)
        f.read(4)
        disp = disp.reshape((nrec_bin, ncol), order='F')

    return edcmp_disp_to_csi(disp)


def _prepare_grn_dir_for_edcmp(workdir, grn_dir, grn_files):
    src_dir = grn_dir if os.path.isabs(grn_dir) else os.path.join(workdir, grn_dir)
    if not os.path.isdir(src_dir) or not os.listdir(src_dir):
        raise FileNotFoundError(
            f"Green's function directory '{src_dir}' does not exist or is empty. "
            "Please prepare EDGRN outputs in this directory before running EDCMP."
        )

    if os.path.isabs(grn_dir):
        runtime_name = os.path.basename(os.path.normpath(src_dir)) or "edgrnfcts"
        runtime_dir = os.path.join(workdir, runtime_name)
        os.makedirs(runtime_dir, exist_ok=True)
        for fname in grn_files:
            src_file = os.path.join(src_dir, fname)
            dst_file = os.path.join(runtime_dir, fname)
            if not os.path.isfile(src_file):
                raise FileNotFoundError(f"Missing EDGRN output file: {src_file}")
            if not os.path.isfile(dst_file):
                try:
                    os.link(src_file, dst_file)
                except OSError:
                    shutil.copy2(src_file, dst_file)
        cfg_dir = os.path.join('.', runtime_name)
    else:
        runtime_dir = src_dir
        cfg_dir = os.path.join('.', grn_dir)

    return src_dir, runtime_dir, cfg_dir


def _build_edcmp_sources(xs, ys, zs, width, length, strike, dip, slip_total, rake):
    """
    Build a list of RectangularSource objects with CSI→EDCMP coordinate swap.

    Parameters
    ----------
    xs, ys : float or array-like
        Source positions in CSI convention (metres, already offset from origin).
    zs : float or array-like
        Source depth(s) in metres.
    width, length, strike, dip : float or array-like
        Geometric parameters.
    slip_total : float
        Slip amplitude.
    rake : float
        Rake angle in degrees.

    Returns
    -------
    sources : list of RectangularSource
    """
    xs_edcmp, ys_edcmp = csi_source_to_edcmp(xs, ys)

    if isinstance(xs, (list, tuple, np.ndarray)):
        n = len(xs)
        width_arr = width if isinstance(width, (list, tuple, np.ndarray)) else [width] * n
        length_arr = length if isinstance(length, (list, tuple, np.ndarray)) else [length] * n
        strike_arr = strike if isinstance(strike, (list, tuple, np.ndarray)) else [strike] * n
        dip_arr = dip if isinstance(dip, (list, tuple, np.ndarray)) else [dip] * n
        zs_arr = zs if isinstance(zs, (list, tuple, np.ndarray)) else [zs] * n
        return [
            RectangularSource(
                source_id=i + 1,
                slip=slip_total,
                xs=xs_edcmp[i],
                ys=ys_edcmp[i],
                zs=zs_arr[i],
                width=width_arr[i],
                length=length_arr[i],
                strike=strike_arr[i],
                dip=dip_arr[i],
                rake=rake,
            )
            for i in range(n)
        ]
    else:
        return [RectangularSource(
            source_id=1, slip=slip_total,
            xs=xs_edcmp, ys=ys_edcmp, zs=zs,
            width=width, length=length,
            strike=strike, dip=dip, rake=rake
        )]


def _run_edcmp_exe(params, workdir, inp_basename, BIN_EDCMP_PATH):
    """
    Write EDCMP input file, run the exe, and check for errors.

    Parameters
    ----------
    params : EDCMPParameters
        Fully configured parameters (sources, observation points, etc.).
    workdir : str
        Absolute path to working directory.
    inp_basename : str
        Basename for the .inp file (written under workdir).
    BIN_EDCMP_PATH : str
        Path to directory containing the edcmp executable.
    """
    inp_file = os.path.join(workdir, inp_basename)
    config = EDCMPConfig(params)
    config.write_config_file(inp_file, verbose=False)

    exe_path = os.path.join(BIN_EDCMP_PATH, 'edcmp' + ('.exe' if sys.platform.startswith('win') else ''))
    cmd = [exe_path, os.path.join('.', inp_basename)]
    with subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            shell=(sys.platform == "win32")
        ) as proc:
        out, err = proc.communicate()
        if proc.returncode != 0:
            print(f"[EDCMP ERROR] Command failed: {' '.join(cmd)}")
            print(f"[EDCMP ERROR] Return code: {proc.returncode}")
            print(f"[EDCMP ERROR] Working directory: {workdir}")
            print(f"[EDCMP ERROR] stdout:\n{out.decode(errors='ignore')}")
            print(f"[EDCMP ERROR] stderr:\n{err.decode(errors='ignore')}")
            raise RuntimeError(f"EDCMP failed with return code {proc.returncode}")


def _prepare_exe_environment(BIN_EDCMP, workdir, output_dir, grn_dir):
    """
    Common setup for exe-mode functions: resolve binary path, prepare directories.

    Returns
    -------
    BIN_EDCMP_PATH, workdir, out_dir, grn_dir_abs, grn_runtime_dir, grn_dir_cfg : str
    """
    BIN_EDCMP_PATH = get_edcmp_bin() if BIN_EDCMP is None else os.environ.get(BIN_EDCMP, None)
    if not BIN_EDCMP_PATH:
        raise RuntimeError(
            f"Environment variable '{BIN_EDCMP}' is not set or empty. "
            "Please set it to your EDCMP binary directory."
        )

    grn_files = ('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    workdir = os.path.abspath(workdir)
    out_dir = os.path.join(workdir, output_dir)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    grn_dir_abs, grn_runtime_dir, grn_dir_cfg = _prepare_grn_dir_for_edcmp(workdir, grn_dir, grn_files)

    return BIN_EDCMP_PATH, workdir, out_dir, grn_dir_cfg, grn_files


def _make_base_params(output_dir_cfg, grn_dir_cfg, grn_files, layered_model, data, mean_x, mean_y):
    """
    Create EDCMPParameters with observation points set.

    Returns
    -------
    params : EDCMPParameters
    """
    obs_xrec, obs_yrec = csi_obs_to_edcmp(data.x, data.y, mean_x, mean_y)
    obs_coords = np.column_stack([obs_xrec, obs_yrec])

    params = EDCMPParameters(
        output_dir=output_dir_cfg + os.sep,
        output_flags=(1, 0, 0, 0),
        output_files=('edcmp.disp', 'edcmp.strn', 'edcmp.strss', 'edcmp.tilt'),
        layered_model=layered_model,
        grn_dir=grn_dir_cfg + os.sep,
        grn_files=grn_files
    )
    params.set_irregular_observation_points([tuple(pt) for pt in obs_coords])
    return params


def edcmpslip2dis(
    data, source_params, slip,
    BIN_EDCMP=None,
    grn_dir='edgrnfcts',
    output_dir='edcmpgrns',
    filename_suffix='',
    workdir='edcmp_ecat',
    layered_model=True,
    force_recompute=True,
    faultname='',
    dataname=''
):
    """
    Generate EDCMP input file and call EDCMP to compute Green's functions for a single rectangular fault.

    Parameters
    ----------
    data : object
        Observation points object, must have .x, .y attributes (unit: kilo-meters)
    source_params : tuple
        (xs, ys, zs, width, length, strike, dip, mean_x, mean_y)
    slip : list or tuple
        Slip vector: [strike-slip, dip-slip, tensile-slip].
    BIN_EDCMP : str or None
        Environment variable name for EDCMP binary directory (default: None, use built-in bin)
    grn_dir : str
        Directory for Green's functions, relative to workdir or absolute
    output_dir : str
        Directory for EDCMP outputs, relative to workdir or absolute
    filename_suffix : str
        Suffix for output files to avoid name conflicts
    workdir : str
        Working directory for all intermediate and output files
    layered_model : bool
        Use layered model for Green's functions
    force_recompute : bool
        If True, recompute Green's functions even if they already exist.
    faultname : str
        Name of the fault.
    dataname : str
        Name of the data.

    Returns
    -------
    ss, ds, ts : np.ndarray
        Green's functions for strike-slip, dip-slip, tensile-slip (Nd, 3).
    """
    BIN_EDCMP_PATH, workdir, out_dir, grn_dir_cfg, grn_files = _prepare_exe_environment(
        BIN_EDCMP, workdir, output_dir, grn_dir
    )

    xs, ys, zs, width, length, strike, dip, mean_x, mean_y = source_params
    output_dir_cfg = out_dir if os.path.isabs(output_dir) else os.path.join('.', output_dir)

    params = _make_base_params(output_dir_cfg, grn_dir_cfg, grn_files, layered_model, data, mean_x, mean_y)

    # Output file paths
    ds_data_basename = f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.disp'
    ds_file = os.path.join(out_dir, ds_data_basename)
    ss_data_basename = f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.disp'
    ss_file = os.path.join(out_dir, ss_data_basename)
    Nd = len(data.x)
    ds = np.zeros((Nd, 3))
    ss = np.zeros((Nd, 3))

    # Check cache
    if (not force_recompute) and os.path.exists(ds_file) and os.path.exists(ss_file):
        ds = read_disp_bin(ds_file, Nd, 3)
        ss = read_disp_bin(ss_file, Nd, 3)
        ts = np.zeros_like(ss)
        return ss, ds, ts

    # Compute strike-slip Green's function
    if slip[0] == 1.0:
        params.sources = _build_edcmp_sources(xs, ys, zs, width, length, strike, dip, 1.0, 0.0)
        params.output_files = (ss_data_basename,
                            f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.strn',
                            f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.strss',
                            f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.tilt')
        _run_edcmp_exe(params, workdir, f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.inp', BIN_EDCMP_PATH)

    # Compute dip-slip Green's function
    if slip[1] == 1.0:
        params.sources = _build_edcmp_sources(xs, ys, zs, width, length, strike, dip, 1.0, 90.0)
        params.output_files = (ds_data_basename,
                            f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.strn',
                            f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.strss',
                            f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.tilt')
        _run_edcmp_exe(params, workdir, f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.inp', BIN_EDCMP_PATH)

    # Read outputs
    if os.path.exists(ss_file):
        ss = read_disp_bin(ss_file, Nd, 3)
    if os.path.exists(ds_file):
        ds = read_disp_bin(ds_file, Nd, 3)
    ts = np.zeros_like(ss)
    return ss, ds, ts

def edcmpslip2dis_forward(
    data, source_params, slip,
    BIN_EDCMP=None,
    grn_dir='edgrnfcts',
    output_dir='edcmpgrns',
    filename_suffix='',
    workdir='edcmp_ecat',
    layered_model=True,
    force_recompute=True,
    faultname='',
    dataname=''
):
    """
    Forward calculation of surface displacement using EDCMP for a single rectangular fault.
    Accepts a physical slip vector [ss, ds, ts], automatically computes rake and slip amplitude,
    and outputs only one displacement file.

    Parameters
    ----------
    data : object
        Observation points object, must have .x, .y attributes (unit: kilo-meters)
    source_params : tuple or list
        (xs, ys, zs, width, length, strike, dip, mean_x, mean_y)
    slip : list or tuple
        Physical slip vector: [strike-slip, dip-slip, tensile-slip].
    BIN_EDCMP : str or None
        Environment variable name for EDCMP binary directory (default: None, use built-in bin)
    grn_dir : str
        Directory for Green's functions, relative to workdir or absolute
    output_dir : str
        Directory for EDCMP outputs, relative to workdir or absolute
    filename_suffix : str
        Suffix for output files to avoid name conflicts
    workdir : str
        Working directory for all intermediate and output files
    layered_model : bool
        Use layered model for Green's functions
    force_recompute : bool
        If True, recompute Green's functions even if they already exist.
    faultname : str
        Name of the fault.
    dataname : str
        Name of the data.

    Returns
    -------
    disp : np.ndarray
        Displacement field at each observation point, shape (Nd, 3).
    """
    BIN_EDCMP_PATH, workdir, out_dir, grn_dir_cfg, grn_files = _prepare_exe_environment(
        BIN_EDCMP, workdir, output_dir, grn_dir
    )

    xs, ys, zs, width, length, strike, dip, mean_x, mean_y = source_params
    output_dir_cfg = out_dir if os.path.isabs(output_dir) else os.path.join('.', output_dir)

    params = _make_base_params(output_dir_cfg, grn_dir_cfg, grn_files, layered_model, data, mean_x, mean_y)

    # Compute slip amplitude and rake from input slip vector
    ss_slip, ds_slip, ts_slip = slip
    slip_total = np.sqrt(ss_slip**2 + ds_slip**2)
    if slip_total > 0:
        rake = np.degrees(np.arctan2(ds_slip, ss_slip))
    else:
        rake = 0.0

    # Output file
    disp_data_basename = f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.disp'
    disp_file = os.path.join(out_dir, disp_data_basename)
    Nd = len(data.x)
    disp = np.zeros((Nd, 3))

    # Check cache
    if (not force_recompute) and os.path.exists(disp_file):
        disp = read_disp_bin(disp_file, Nd, 3)
        return disp

    # Build sources and run
    params.sources = _build_edcmp_sources(xs, ys, zs, width, length, strike, dip, slip_total, rake)
    params.output_files = (disp_data_basename,
                          f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.strn',
                          f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.strss',
                          f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.tilt')

    _run_edcmp_exe(params, workdir, f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.inp', BIN_EDCMP_PATH)

    # Read output
    if os.path.exists(disp_file):
        disp = read_disp_bin(disp_file, Nd, 3)
    return disp

# Example usage
if __name__ == "__main__":
    class DummyObs:
        def __init__(self, x, y):
            self.x = np.array(x)
            self.y = np.array(y)
    obs = DummyObs(x=[0, 1000, 2000], y=[0, 1000, 2000])
    source_params = dict(
        slip=2.5, xs=0.0, ys=0.0, zs=0.2,
        length=11e3, width=10e3, strike=174.0, dip=88.0, rake=178.0
    )
    disp = edcmpslip2dis(obs, source_params)
    print(disp)
