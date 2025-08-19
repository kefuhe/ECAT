import os
import sys
import subprocess
import numpy as np
import pandas as pd
from .edcmp_config import EDCMPConfig, EDCMPParameters, RectangularSource

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

def edcmpslip2dis(
    data, source_params, slip,
    BIN_EDCMP=None,
    grn_dir='edgrnfcts',
    output_dir='edcmpgrns',
    filename_suffix='',
    workdir='edcmp_ecat',
    layered_model=True
):
    """
    Generate EDCMP input file and call EDCMP to compute Green's functions for a single rectangular fault.
    Parameters
    ----------
    data : object
        Observation points object, must have .x, .y attributes (unit: meters)
    source_params : dict
        Parameters required by RectangularSource
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
    Returns
    -------
    ss, ds, ts : np.ndarray
        Green's functions for strike-slip, dip-slip, tensile-slip (Nd, 3).
    """
    BIN_EDCMP_PATH = get_edcmp_bin() if BIN_EDCMP is None else os.environ.get(BIN_EDCMP, None)
    # print('The EDCMP binary directory is:', BIN_EDCMP_PATH)
    if not BIN_EDCMP_PATH:
        raise RuntimeError(f"Environment variable '{BIN_EDCMP}' is not set or empty. Please set it to your EDCMP binary directory.")

    # Prepare working directories
    workdir = os.path.abspath(workdir)
    grn_dir_abs = os.path.join(workdir, grn_dir)
    out_dir = os.path.join(workdir, output_dir)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(grn_dir_abs, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Check Green's function directory
    if not os.path.isdir(grn_dir_abs) or not os.listdir(grn_dir_abs):
        raise FileNotFoundError(
            f"Green's function directory '{grn_dir_abs}' does not exist or is empty. "
            "Please prepare EDGRN outputs in this directory before running EDCMP."
        )

    # Prepare observation points
    layered_model=layered_model
    grn_files=('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    output_flags = (1, 0, 0, 0)
    output_files = ('edcmp.disp', 'edcmp.strn', 'edcmp.strss', 'edcmp.tilt')

    xs, ys, zs, width, length, strike, dip, mean_x, mean_y = source_params

    obs_coords = np.column_stack([(data.y-mean_y)*1000.0, (data.x-mean_x)*1000.0])
    params = EDCMPParameters(
        output_dir=os.path.join('.', output_dir) + os.sep,
        output_flags=output_flags,
        output_files=output_files,
        layered_model=layered_model,
        grn_dir=os.path.join('.', grn_dir) + os.sep,
        grn_files=grn_files
    )
    params.set_irregular_observation_points([tuple(pt) for pt in obs_coords])

    # Compute strike-slip Green's function
    if slip[0] == 1.0:
        slip_total = 1.0
        rake = 0.0
        # swap x and y for edcmp
        if isinstance(xs, (list, tuple, np.ndarray)):
            params.sources = [RectangularSource(source_id=i+1, slip=slip_total,
                                                xs=ys[i], ys=xs[i], zs=zs[i],
                                                width=width, length=length,
                                                strike=strike, dip=dip, rake=rake) for i in range(len(xs))]
        else:
            params.sources = [RectangularSource(source_id=1, slip=slip_total,
                                                xs=ys, ys=xs, zs=zs,
                                                width=width, length=length,
                                                strike=strike, dip=dip, rake=rake)]
        params.output_files = (f'edcmp_ss_{filename_suffix}.disp',
                            f'edcmp_ss_{filename_suffix}.strn',
                            f'edcmp_ss_{filename_suffix}.strss',
                            f'edcmp_ss_{filename_suffix}.tilt')
        # Write input file
        inp_ss = os.path.join(os.path.basename(workdir), f'edcmp_ss_{filename_suffix}.inp')
        config = EDCMPConfig(params)
        config.write_config_file(inp_ss, verbose=False)
        # Call edcmp
        exe_path = os.path.join(BIN_EDCMP_PATH, 'edcmp' + ('.exe' if sys.platform.startswith('win') else ''))
        cmd = [exe_path, os.path.join('.', f'edcmp_ss_{filename_suffix}.inp')]
        # with subprocess.Popen(
        #     cmd,
        #     cwd=workdir,
        #     stdout=subprocess.DEVNULL,
        #     stderr=subprocess.DEVNULL,
        #     shell=(sys.platform == "win32")
        # ) as proc:
        #     proc.wait()
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

    # Compute dip-slip Green's function
    if slip[1] == 1.0:
        slip_total = 1.0
        rake = 90.0
        # Swap xs and ys for edcmp
        if isinstance(xs, (list, tuple, np.ndarray)):
            params.sources = [RectangularSource(source_id=i+1, slip=slip_total,
                                                xs=ys[i], ys=xs[i], zs=zs[i],
                                                width=width, length=length,
                                                strike=strike, dip=dip, rake=rake) for i in range(len(xs))]
        else:
            params.sources = [RectangularSource(source_id=1, slip=slip_total,
                                                xs=ys, ys=xs, zs=zs,
                                                width=width, length=length,
                                                strike=strike, dip=dip, rake=rake)]
        params.output_files = (f'edcmp_ds_{filename_suffix}.disp',
                            f'edcmp_ds_{filename_suffix}.strn',
                            f'edcmp_ds_{filename_suffix}.strss',
                            f'edcmp_ds_{filename_suffix}.tilt')
        # Write input file
        inp_ds = os.path.join(os.path.basename(workdir), f'edcmp_ds_{filename_suffix}.inp')
        config = EDCMPConfig(params)
        config.write_config_file(inp_ds, verbose=False)
        # Call edcmp
        exe_path = os.path.join(BIN_EDCMP_PATH, 'edcmp' + ('.exe' if sys.platform.startswith('win') else ''))
        cmd = [exe_path, os.path.join('.', f'edcmp_ds_{filename_suffix}.inp')]
        with subprocess.Popen(
            cmd,
            cwd=workdir,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
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


    # Read outputs
    ds_file = os.path.join(out_dir, f'edcmp_ds_{filename_suffix}.disp')
    ss_file = os.path.join(out_dir, f'edcmp_ss_{filename_suffix}.disp')
    Nd = len(data.x)
    ds = np.zeros((Nd, 3))
    ss = np.zeros((Nd, 3))
    if os.path.exists(ds_file):
        data_ds = pd.read_csv(ds_file, sep=r'\s+', comment='#', names=['X_m', 'Y_m', 'Ux_m', 'Uy_m', 'Uz_m'])
        ds = data_ds[['Ux_m', 'Uy_m', 'Uz_m']].copy()
        ds = ds.rename({'Uy_m': 'dx', 'Ux_m': 'dy', 'Uz_m': 'dz'}, axis=1)
        ds['dz'] *= -1
        ds = ds[['dx', 'dy', 'dz']].values
    if os.path.exists(ss_file):
        data_ss = pd.read_csv(ss_file, sep=r'\s+', comment='#', names=['X_m', 'Y_m', 'Ux_m', 'Uy_m', 'Uz_m'])
        ss = data_ss[['Ux_m', 'Uy_m', 'Uz_m']].copy()
        ss = ss.rename({'Uy_m': 'dx', 'Ux_m': 'dy', 'Uz_m': 'dz'}, axis=1)
        ss['dz'] *= -1
        ss = ss[['dx', 'dy', 'dz']].values

    ts = np.zeros_like(ss)
    return ss, ds, ts

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