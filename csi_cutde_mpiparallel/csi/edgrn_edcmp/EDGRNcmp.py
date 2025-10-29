import os
import sys
import subprocess
import numpy as np
import pandas as pd
from .edcmp_config import EDCMPConfig, EDCMPParameters, RectangularSource
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

        disp_arr = np.empty_like(disp)
        disp_arr[:, 0] = disp[:, 1]
        disp_arr[:, 1] = disp[:, 0]
        disp_arr[:, 2] = -disp[:, 2]
    return disp_arr

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

    # Read outputs
    ds_data_basename = f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.disp'
    ds_file = os.path.join(out_dir, ds_data_basename)
    ss_data_basename = f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.disp'
    ss_file = os.path.join(out_dir, ss_data_basename)
    Nd = len(data.x)
    ds = np.zeros((Nd, 3))
    ss = np.zeros((Nd, 3))
    if (not force_recompute) and os.path.exists(ds_file) and os.path.exists(ss_file):
        # # Skip the comment line and read (2, 3, 4)
        # ds_arr = np.loadtxt(ds_file, comments='#', usecols=(2, 3, 4))
        # ds = np.empty_like(ds_arr)
        # ds[:, 0] = ds_arr[:, 1]  # dx = Uy_m
        # ds[:, 1] = ds_arr[:, 0]  # dy = Ux_m
        # ds[:, 2] = -ds_arr[:, 2] # dz = -Uz_m
    
        # ss_arr = np.loadtxt(ss_file, comments='#', usecols=(2, 3, 4))
        # ss = np.empty_like(ss_arr)
        # ss[:, 0] = ss_arr[:, 1]
        # ss[:, 1] = ss_arr[:, 0]
        # ss[:, 2] = -ss_arr[:, 2]
    
        # ts = np.zeros_like(ss)

        ds = read_disp_bin(ds_file, Nd, 3)
        ss = read_disp_bin(ss_file, Nd, 3)
        ts = np.zeros_like(ss)  # Tensile slip Green's function is not computed
        return ss, ds, ts

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
        params.output_files = (ss_data_basename,
                            f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.strn',
                            f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.strss',
                            f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.tilt')
        # Write input file
        inp_ss_basename = f'edcmp_ss_{faultname}_{dataname}_{filename_suffix}.inp'
        inp_ss = os.path.join(os.path.basename(workdir), inp_ss_basename)
        config = EDCMPConfig(params)
        config.write_config_file(inp_ss, verbose=False)
        # Call edcmp
        exe_path = os.path.join(BIN_EDCMP_PATH, 'edcmp' + ('.exe' if sys.platform.startswith('win') else ''))
        cmd = [exe_path, os.path.join('.', inp_ss_basename)]
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
        params.output_files = (ds_data_basename,
                            f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.strn',
                            f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.strss',
                            f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.tilt')
        # Write input file
        inp_ds_basename = f'edcmp_ds_{faultname}_{dataname}_{filename_suffix}.inp'
        inp_ds = os.path.join(os.path.basename(workdir), inp_ds_basename)
        config = EDCMPConfig(params)
        config.write_config_file(inp_ds, verbose=False)
        # Call edcmp
        exe_path = os.path.join(BIN_EDCMP_PATH, 'edcmp' + ('.exe' if sys.platform.startswith('win') else ''))
        cmd = [exe_path, os.path.join('.', inp_ds_basename)]
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
    if os.path.exists(ds_file) and os.path.exists(ss_file):
        # Skip the comment line and read (2, 3, 4)
        # ds_arr = np.loadtxt(ds_file, comments='#', usecols=(2, 3, 4))
        # ds = np.empty_like(ds_arr)
        # ds[:, 0] = ds_arr[:, 1]  # dx = Uy_m
        # ds[:, 1] = ds_arr[:, 0]  # dy = Ux_m
        # ds[:, 2] = -ds_arr[:, 2] # dz = -Uz_m
    
        # ss_arr = np.loadtxt(ss_file, comments='#', usecols=(2, 3, 4))
        # ss = np.empty_like(ss_arr)
        # ss[:, 0] = ss_arr[:, 1]
        # ss[:, 1] = ss_arr[:, 0]
        # ss[:, 2] = -ss_arr[:, 2]
    
        # ts = np.zeros_like(ss)
        ds = read_disp_bin(ds_file, Nd, 3)
        ss = read_disp_bin(ss_file, Nd, 3)
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
    import os
    import sys
    import numpy as np
    from .edcmp_config import EDCMPConfig, EDCMPParameters, RectangularSource

    BIN_EDCMP_PATH = get_edcmp_bin() if BIN_EDCMP is None else os.environ.get(BIN_EDCMP, None)
    if not BIN_EDCMP_PATH:
        raise RuntimeError(f"Environment variable '{BIN_EDCMP}' is not set or empty. Please set it to your EDCMP binary directory.")

    workdir = os.path.abspath(workdir)
    grn_dir_abs = os.path.join(workdir, grn_dir)
    out_dir = os.path.join(workdir, output_dir)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(grn_dir_abs, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.isdir(grn_dir_abs) or not os.listdir(grn_dir_abs):
        raise FileNotFoundError(
            f"Green's function directory '{grn_dir_abs}' does not exist or is empty. "
            "Please prepare EDGRN outputs in this directory before running EDCMP."
        )

    xs, ys, zs, width, length, strike, dip, mean_x, mean_y = source_params

    # Prepare observation points (EDCMP expects y, x order, in meters relative to mean)
    obs_coords = np.column_stack([(data.y-mean_y)*1000.0, (data.x-mean_x)*1000.0])

    params = EDCMPParameters(
        output_dir=os.path.join('.', output_dir) + os.sep,
        output_flags=(1, 0, 0, 0),
        output_files=('edcmp.disp', 'edcmp.strn', 'edcmp.strss', 'edcmp.tilt'),
        layered_model=layered_model,
        grn_dir=os.path.join('.', grn_dir) + os.sep,
        grn_files=('edgrnhs.ss', 'edgrnhs.ds', 'edgrnhs.cl')
    )
    params.set_irregular_observation_points([tuple(pt) for pt in obs_coords])

    # Compute slip amplitude and rake from input slip vector
    ss, ds, ts = slip
    slip_total = np.sqrt(ss**2 + ds**2)
    if slip_total > 0:
        rake = np.degrees(np.arctan2(ds, ss))
    else:
        rake = 0.0

    # Only one source, only one output file
    disp_data_basename = f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.disp'
    disp_file = os.path.join(out_dir, disp_data_basename)

    params.sources = [RectangularSource(
        source_id=1, slip=slip_total,
        xs=ys, ys=xs, zs=zs,
        width=width, length=length,
        strike=strike, dip=dip, rake=rake
    )]
    params.output_files = (disp_data_basename,
                          f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.strn',
                          f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.strss',
                          f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.tilt')

    inp_basename = f'edcmp_disp_{faultname}_{dataname}_{filename_suffix}.inp'
    inp_file = os.path.join(os.path.basename(workdir), inp_basename)
    config = EDCMPConfig(params)
    config.write_config_file(inp_file, verbose=False)

    exe_path = os.path.join(BIN_EDCMP_PATH, 'edcmp' + ('.exe' if sys.platform.startswith('win') else ''))
    cmd = [exe_path, os.path.join('.', inp_basename)]
    Nd = len(data.x)
    disp = np.zeros((Nd, 3))

    # If not force_recompute and output exists, read directly
    if (not force_recompute) and os.path.exists(disp_file):
        disp = read_disp_bin(disp_file, Nd, 3)
        return disp

    # Run EDCMP
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