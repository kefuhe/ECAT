import os
import sys
import subprocess
import importlib.util

def run_csi_bin(bin_name_win, bin_name_linux):
    # Dynamically find csi package
    spec = importlib.util.find_spec("csi")
    if spec is None or not spec.submodule_search_locations:
        raise ImportError("Cannot find csi package. Please ensure csi is installed.")
    csi_dir = spec.submodule_search_locations[0]
    if sys.platform.startswith('win'):
        bin_dir = os.path.join(csi_dir, 'bin', 'windows')
        exe_name = bin_name_win
    elif sys.platform.startswith('linux'):
        bin_dir = os.path.join(csi_dir, 'bin', 'ubuntu20.04')
        exe_name = bin_name_linux
    else:
        raise RuntimeError('Unsupported platform: ' + sys.platform)
    exe_path = os.path.join(bin_dir, exe_name)
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"{exe_name} executable not found: {exe_path}")
    args = sys.argv[1:]
    cmd = [exe_path] + args
    result = subprocess.run(cmd)
    sys.exit(result.returncode)