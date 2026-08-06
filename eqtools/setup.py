from setuptools import find_packages, setup
from setuptools.command.install import install
import os
import shutil
import site
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CustomInstallCommand(install):
    """Customized setuptools install command - removes old data directory in the installation directory."""
    def run(self):
        # Define the data directory in the installation directory
        install_dir = site.getsitepackages()[0]
        data_dir = os.path.join(install_dir, 'eqtools', 'earthquake_clients', 'data')
        
        # Remove the old data directory if it exists
        if os.path.exists(data_dir):
            logger.info(f"Removing old data directory: {data_dir}")
            shutil.rmtree(data_dir)
        else:
            logger.info(f"No old data directory found at: {data_dir}")
        
        # Proceed with the standard installation
        install.run(self)

setup(
    name='eqtools',
    version='2.0.0',
    author='Kefeng He',
    author_email='kefenghe@whu.edu.cn',
    url='https://github.com/kefuhe/eqtools',
    description='Earthquake Cycle Analysis Toolkit (ECAT)',
    python_requires='>=3.10,<3.13',
    install_requires=[
        # Direct runtime dependencies imported by eqtools. A dependency that
        # is also imported by CSI is intentionally declared in both packages
        # so each standalone checkout remains installable on its own.
        'affine>=2.4,<3',
        'cartopy>=0.21,<0.23',
        'cmcrameri>=1.8,<2',
        'clarabel>=0.11.1,<0.12',
        'cutde>=23.6.25,<24',
        'cvxopt>=1.3,<1.4',
        'gdal>=3.6,<4',
        'geopandas>=0.14,<1',
        'gmsh>=4.11,<5',
        'h5py>=3.8,<4',
        'joblib>=1.3,<2',
        'matplotlib>=3.6,<3.9',
        'meshio>=5.3,<6',
        'mpi4py>=3.1,<4',
        'netCDF4>=1.6,<2',
        'numba>=0.58,<0.60',
        'numpy>=1.23,<2',
        'pandas>=2.0,<2.3',
        'psutil>=5.9,<7',
        'PyYAML>=6,<7',
        'pyproj>=3.5,<3.7',
        'pytz>=2023.3',
        'rasterio>=1.3,<2',
        'requests>=2.28,<3',
        'rtree>=1.0,<2',
        'ruamel.yaml>=0.18,<0.19',
        'scienceplots>=2.1,<3',
        'scikit-learn>=1.2,<1.5',
        'scipy>=1.10,<1.12',
        'seaborn>=0.12,<0.14',
        'shapely>=2.0,<2.1',
        'tabulate>=0.9,<1',
        'tqdm>=4.65,<5',
        'xarray>=2023.1,<2026',
    ],
    extras_require={
        "docs": [
            "mkdocs-material==9.7.7",
        ],
        "geoexport": [
            "h5netcdf>=1.2",
        ],
        "viewer": [
            "dash>=2.17,<4",
            "h5netcdf>=1.2",
            "plotly>=5.24,<7",
        ],
        "interaction": [
            "bokeh>=3.6,<4",
            "datashader>=0.19,<0.20",
            "h5netcdf>=1.2",
        ],
    },
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'eqtools': ['cpt/*'],  # , 'examples/*'
        'eqtools.cli_tools': ['templates/adapter_downsampling/*'],
        'eqtools.Tectonic_Utils': ['README.md', 'cover_picture.png'],
        'eqtools.earthquake_clients': [
            'data/*',
            'data/Faults/*',
            'data/Blocks/*',
            'data/GNSS/*',
        ],
        'eqtools.viztools': ['styles/*.mplstyle'],
    },
    classifiers=[
        # Choose classifiers from https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    entry_points={
        "console_scripts": [
            "ecat-generate-config=eqtools.cli_tools.generate_config:main",
            "ecat-generate-interseismic=eqtools.cli_tools.generate_interseismic_config:main",
            "ecat-generate-boundary=eqtools.cli_tools.generate_bounds_config:main",
            "ecat-generate-nonlinear=eqtools.cli_tools.generate_nonlinear_config:main",
            "ecat-generate-nonlinear-geometry=eqtools.cli_tools.generate_nonlinear_geometry_config:main",
            "ecat-generate-downsample=eqtools.cli_tools.generate_downsample_config:main",
            "ecat-downsample=eqtools.cli_tools.process_data_downsampling:main",
            "ecat-psgrn=eqtools.cli_tools.psgrn_cli:main",
            "ecat-pscmp=eqtools.cli_tools.pscmp_cli:main",
            "ecat-edgrn=eqtools.cli_tools.edgrn_cli:main",
            "ecat-edcmp=eqtools.cli_tools.edcmp_cli:main",
            "ecat-generate-psgrn-template=eqtools.cli_tools.psgrn_template_cli:main",
            "ecat-generate-pscmp-template=eqtools.cli_tools.pscmp_template_cli:main",
            "ecat-generate-edgrn-template=eqtools.cli_tools.edgrn_template_cli:main",
            "ecat-generate-edcmp-template=eqtools.cli_tools.edcmp_template_cli:main",
            "ecat-list-fault-perturb-methods=eqtools.cli_tools.list_fault_perturb_methods:main",
            "ecat-fault-trace-tool=eqtools.cli_tools.fault_trace_tool:main",
            "ecat-export-google-earth=eqtools.cli_tools.export_google_earth:main",
            "ecat-map=eqtools.map_viewer.cli:main",
            "ecat-trace-edit=eqtools.map_viewer.interactive.cli:main",
        ],
    },
    cmdclass={
        'install': CustomInstallCommand,
    },
)
