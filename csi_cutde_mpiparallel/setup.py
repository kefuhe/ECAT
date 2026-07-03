from setuptools import setup, find_packages
from setuptools.command.install import install

setup(
    name="csi",
    version="1.0.0",
    description="CSI stands for Classic Slip Inversion",
    author="jolivet",
    author_email="romain.jolivet@ens.fr",
    url="http://www.geologie.ens.fr/~jolivet/csi/index.html",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "csi": [
            "bin/edcmp4py_ctypes.py",
            "bin/windows/*",
            "bin/ubuntu20.04/*",
            "edgrn_edcmp/build/*.f90",
            "edgrn_edcmp/build/*.f",
            "edgrn_edcmp/build/build_edcmp4py_ctypes.py",
            "sbarbot_src/*.f90",
            "sbarbot_src/build_sbarbot.py",
            "sbarbot_src/sbarbot_python.py",
        ],
    },
    entry_points={
        "console_scripts": [
            "ecat-psgrn=csi.cli_tools.psgrn_cli:main",
        ],
    },
    install_requires=[],
)