import os
import sys
import subprocess

def get_psgrn_bin():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(base_dir)  # Adjust to the parent directory if needed
    if sys.platform.startswith('win'):
        bin_dir = os.path.join(base_dir, 'bin', 'windows')
        exe_name = 'fomosto_psgrn2008a.exe'
    elif sys.platform.startswith('linux'):
        bin_dir = os.path.join(base_dir, 'bin', 'ubuntu20.04')
        exe_name = 'fomosto_psgrn2008a'
    else:
        raise RuntimeError('Unsupported platform: ' + sys.platform)
    exe_path = os.path.join(bin_dir, exe_name)
    if not os.path.exists(exe_path):
        raise FileNotFoundError(f"psgrn executable not found: {exe_path}")
    return exe_path

def main():
    exe_path = get_psgrn_bin()
    # 传递所有命令行参数给 Fortran 可执行文件
    args = sys.argv[1:]
    cmd = [exe_path] + args
    # 直接继承当前终端的输入输出
    result = subprocess.run(cmd)
    sys.exit(result.returncode)

if __name__ == "__main__":
    main()