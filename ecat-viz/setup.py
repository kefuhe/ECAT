from setuptools import find_packages, setup

setup(
    name='ecat-viz',
    version='0.1.2',
    description='General scientific plotting styles, rasters, axes and CPT colors',
    author='Kefeng He',
    python_requires='>=3.10,<3.13',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    include_package_data=True,
    package_data={'ecat_viz': ['styles/*.mplstyle', 'cpt/*']},
    install_requires=[
        'matplotlib>=3.6,<3.9',
        'numpy>=1.23,<2',
        'scienceplots>=2.1,<3',
    ],
    extras_require={
        'raster': ['rasterio>=1.3,<2', 'xarray>=2023.1,<2026', 'netCDF4>=1.6,<2'],
    },
    license='Mixed: MIT plotting utilities; GPL-3.0 CPT parser; per-resource notices',
    license_files=['LICENSE', 'COPYING-GPL-3.0', 'NOTICE'],
)
