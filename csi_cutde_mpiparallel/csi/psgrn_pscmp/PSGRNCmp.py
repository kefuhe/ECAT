import os
import sys
import subprocess
import pickle
import numpy as np 
import pandas as pd
from glob import glob
# import internal libs
from .pscmp_config import PSCMPConfig, PSCMPParameters, FaultSource

def get_pscmp_bin():
    """
    Return the absolute path to the PSCMP binary directory under the csi package,
    regardless of the current module location.

    Returns
    -------
    bin_dir : str
        Absolute path to the PSCMP binary directory for the current platform.
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

# -----------------------------------------------------------------------------------------
def pscmpslip2dis(
    data, p, slip,
    BIN_PSCMP=None, # 'PSCMP_BIN'
    psgrndir='psgrnfcts',
    output_dir='pscmpgrns',
    filename_suffix='',
    workdir='pscmp_ecat'
):
    """
    Generate PSCMP input file and compute Green's functions for a single patch using PSCMPConfig.
    All input/output is isolated per patch. Both psgrndir and output_dir are created under workdir.
    All system commands are executed in workdir for safety.
    Parameters
    ----------
    data : object
        Observation data, must have .lat and .lon attributes (array-like).
    p : tuple
        Fault geometry: (lon, lat, depth, width, length, strike, dip).
    slip : list or tuple
        Slip vector: [strike-slip, dip-slip, tensile-slip].
    BIN_PSCMP : str
        Environment variable name for PSCMP binary directory.
        In Linux: 
            export PSCMP_BIN=/home/yourname/pscmp_bin, where yourname maybe ecat_bins
        In Windows: 
            set PSCMP_BIN=C:\yourname\pscmp_bin, where yourname maybe ecat_bins
    psgrndir : str
        Directory for Green's functions, relative to workdir or absolute.
    output_dir : str
        Directory for PSCMP outputs, relative to workdir or absolute.
    filename_suffix : str
        Suffix for output files to avoid name conflicts.
    workdir : str
        Working directory for all intermediate and output files.
    Returns
    -------
    ss, ds, ts : np.ndarray
        Green's functions for strike-slip, dip-slip, tensile-slip (Nd, 3).
    """
    # print('The bin package in:', get_pscmp_bin())
    BIN_PSCMP_PATH = get_pscmp_bin() if BIN_PSCMP is None else os.environ.get(BIN_PSCMP, None)
    # BIN_PSCMP_PATH = os.environ.get(BIN_PSCMP, None)
    if not BIN_PSCMP_PATH:
        raise RuntimeError(f"Environment variable '{BIN_PSCMP}' is not set or empty. Please set it to your PSCMP binary directory.")

    # Prepare working directories
    workdir = os.path.abspath(workdir)
    psgrn_dir = os.path.join(workdir, psgrndir)
    out_dir = os.path.join(workdir, output_dir)
    os.makedirs(workdir, exist_ok=True)
    os.makedirs(psgrn_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # Use relative paths for PSCMP input
    rel_psgrn_dir = os.path.join('.', psgrndir)
    rel_out_dir = os.path.join('.', output_dir)

    # Check Green's function directory
    if not os.path.isdir(psgrn_dir) or not os.listdir(psgrn_dir):
        raise FileNotFoundError(
            f"Green's function directory '{psgrn_dir}' does not exist or is empty. "
            "Please prepare PSGRN outputs in this directory before running PSCMP."
        )

    # Prepare observation points DataFrame
    obs_df = pd.DataFrame({'lat': data.lat, 'lon': data.lon})

    # Prepare FaultSource
    lon, lat, depth, width, length, strike, dip = p
    if not isinstance(lon, (list, tuple, np.ndarray)):
        fault = FaultSource(
            fault_id=1,
            o_lat=lat,
            o_lon=lon,
            o_depth=depth,
            length=length,
            width=width,
            strike=strike,
            dip=dip,
            np_st=1,
            np_di=1,
            start_time=0.0
        )
        fault_sources = [fault]
    else:
        # Unpack source parameters
        fault_sources = [FaultSource(
            fault_id=i,
            o_lat=lat[i],
            o_lon=lon[i],
            o_depth=depth[i],
            length=length,
            width=width,
            strike=strike,
            dip=dip,
            np_st=1,
            np_di=1,
            start_time=0.0
        ) for i in range(len(lon))]

    # Prepare PSCMP parameters
    params = PSCMPParameters(
        iposrec=0,
        output_dir=rel_out_dir + os.sep,
        grn_dir=rel_psgrn_dir + os.sep,
        fault_sources=fault_sources
    )
    params.load_observation_points_from_dataframe(obs_df)

    # Compute strike-slip Green's function
    # Prepare input file paths
    inp_ss = os.path.join(os.path.basename(workdir), f'pscmp_ss_{filename_suffix}.inp')
    if slip[0] == 1.0:
        for ifault in fault_sources:
            ifault.patches = [{
                'pos_s': 0.0,
                'pos_d': 0.0,
                'slip_strike': 1.0,
                'slip_downdip': 0.0,
            'opening': 0.0
            }]
        params.snapshots = [{
            'time': 0.00,
            'filename': f'snapshot_coseism_ss1_{filename_suffix}.dat',
            'comment': '0 co-seismic'
        }]
        config = PSCMPConfig(params)
        config.write_config_file(inp_ss, verbose=False)
        # Run PSCMP in workdir (cross-platform)
        exe_path = os.path.join(BIN_PSCMP_PATH, 'fomosto_pscmp2008a' + ('.exe' if sys.platform.startswith('win') else ''))
        cmd = [exe_path, os.path.join('.', f'pscmp_ss_{filename_suffix}.inp')]
        with subprocess.Popen(
                cmd,
                cwd=workdir,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=(sys.platform == "win32")
            ) as proc:
            out, err = proc.communicate()
            if proc.returncode != 0:
                print(f"[PSCMP ERROR] Command failed: {' '.join(cmd)}")
                print(f"[PSCMP ERROR] Return code: {proc.returncode}")
                print(f"[PSCMP ERROR] Working directory: {workdir}")
                print(f"[PSCMP ERROR] stdout:\n{out.decode(errors='ignore')}")
                print(f"[PSCMP ERROR] stderr:\n{err.decode(errors='ignore')}")

    # Compute dip-slip Green's function
    # Prepare input file path
    inp_ds = os.path.join(os.path.basename(workdir), f'pscmp_ds_{filename_suffix}.inp')
    if slip[1] == 1.0:
        for ifault in fault_sources:
            ifault.patches = [{
                'pos_s': 0.0,
                'pos_d': 0.0,
                'slip_strike': 0.0,
                'slip_downdip': -1.0,
            'opening': 0.0
            }]
        params.snapshots = [{
            'time': 0.00,
            'filename': f'snapshot_coseism_ds1_{filename_suffix}.dat',
            'comment': '0 co-seismic'
        }]
        config = PSCMPConfig(params)
        config.write_config_file(inp_ds, verbose=False)
        # Run PSCMP in workdir (cross-platform)
        exe_path = os.path.join(BIN_PSCMP_PATH, 'fomosto_pscmp2008a' + ('.exe' if sys.platform.startswith('win') else ''))
        cmd = [exe_path, os.path.join('.', f'pscmp_ds_{filename_suffix}.inp')]
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
                print(f"[PSCMP ERROR] Command failed: {' '.join(cmd)}")
                print(f"[PSCMP ERROR] Return code: {proc.returncode}")
                print(f"[PSCMP ERROR] Working directory: {workdir}")
                print(f"[PSCMP ERROR] stdout:\n{out.decode(errors='ignore')}")
                print(f"[PSCMP ERROR] stderr:\n{err.decode(errors='ignore')}")

    # Read outputs
    ds_file = os.path.join(out_dir, f'snapshot_coseism_ds1_{filename_suffix}.dat')
    ss_file = os.path.join(out_dir, f'snapshot_coseism_ss1_{filename_suffix}.dat')
    Nd = len(data.lat)
    ds = np.zeros((Nd, 3))
    ss = np.zeros((Nd, 3))
    if os.path.exists(ds_file):
        data_ds = pd.read_csv(ds_file, sep=r'\s+')
        ds = data_ds[['Ux', 'Uy', 'Uz']].copy()
        ds = ds.rename({'Uy': 'dx', 'Ux': 'dy', 'Uz': 'dz'}, axis=1)
        ds['dz'] *= -1
        ds = ds[['dx', 'dy', 'dz']].values
    if os.path.exists(ss_file):
        data_ss = pd.read_csv(ss_file, sep=r'\s+')
        ss = data_ss[['Ux', 'Uy', 'Uz']].copy()
        ss = ss.rename({'Uy': 'dx', 'Ux': 'dy', 'Uz': 'dz'}, axis=1)
        ss['dz'] *= -1
        ss = ss[['dx', 'dy', 'dz']].values

    ts = np.zeros_like(ss)
    return ss, ds, ts
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------
def changePoints2pscmpfile(data, BIN_PSCMP='PSCMP_BIN', pscmpinp='./pscmp_user.inp'):
    '''
    用于更新pscmp.inp文件中需求的观测点位信息
    '''
    # Get executables
    # Environment variables need to be set in advance
    BIN_PSCMP = os.environ[BIN_PSCMP]
    pscmp_template = os.path.join(BIN_PSCMP, 'pscmp_template.inp')

    import math
    lon, lat = data.lon, data.lat
    npnts = data.lon.shape[0]
    output_format = '({0:.3f},{1:.3f})'
    pntlines = ''
    for i, (ilat, ilon) in enumerate(zip(lat, lon)):
        if i%3 == 0:
            pntlines += '   ' + output_format.format(ilat, ilon)
        elif i%3 == 1:
            pntlines += ', ' + output_format.format(ilat, ilon)
        else:
            pntlines += ', ' + output_format.format(ilat, ilon) + '\n'
    if i%3 != 2:
        pntlines += '\n'
    
    assert os.path.exists(pscmp_template), 'Not find the pscmp input file: {}'.format(pscmp_template)
    os.system('cp "{0}" {1}'.format(pscmp_template, pscmpinp))
    with open(pscmpinp, 'rt') as fin:
        lines = fin.readlines()
        nlines = int(lines[71].strip())
        lines[71] = '  {0:d}\n'.format(npnts)
        lines[72] = pntlines
    
    nlines = math.ceil(nlines/3.)
    lines = lines[:73] + lines[72+nlines:]
    
    with open(pscmpinp, 'wt') as fout:
        for line in lines:
            print(line, end='', file=fout)

    return
# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------

# -----------------------------------------------------------------------------------------