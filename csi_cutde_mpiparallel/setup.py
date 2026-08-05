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
    python_requires=">=3.10,<3.13",
    install_requires=[
        # Imported by csi.__init__; keep this list in sync with
        # scripts/generate_requirements.py.
        "cartopy>=0.21,<0.23",
        "hdbscan>=0.8.33,<0.9",
        "matplotlib>=3.6,<3.9",
        "numba>=0.58,<0.60",
        "numpy>=1.23,<2",
        "okada4py>=12,<13",
        "pandas>=2.0,<2.3",
        "pyproj>=3.5,<3.7",
        "pyshp>=2.3,<3",
        "scikit-image>=0.22,<0.24",
        "scikit-learn>=1.2,<1.5",
        "scienceplots>=2.1,<3",
        "scipy>=1.10,<1.12",
        "shapely>=2.0,<2.1",
        "tqdm>=4.65,<5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
