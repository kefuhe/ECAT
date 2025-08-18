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
    version='1.1.5',
    author='Kefeng He',
    author_email='kefenghe@whu.edu.cn',
    url='https://github.com/kefuhe/eqtools',
    description='Earthquake Cycle Analysis Toolkit (ECAT)',
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'eqtools': ['cpt/*'],  # , 'examples/*'
        'eqtools.Tectonic_Utils': ['README.md', 'cover_picture.png'],
        'eqtools.earthquake_clients': ['data/*', 'data/Faults/*', 'data/Blocks/*'],
    },
    classifiers=[
        # Choose classifiers from https://pypi.org/classifiers/
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.10',
    ],
    entry_points={
        "console_scripts": [
            "ecat-generate-config=eqtools.cli_tools.generate_config:main",
            "ecat-generate-boundary=eqtools.cli_tools.generate_bounds_config:main",
            "ecat-generate-nonlinear=eqtools.cli_tools.generate_nonlinear_config:main",
            "ecat-generate-downsample=eqtools.cli_tools.generate_downsample_config:main",
            "ecat-psgrn=eqtools.cli_tools.psgrn_cli:main",
            "ecat-pscmp=eqtools.cli_tools.pscmp_cli:main",
            "ecat-edgrn=eqtools.cli_tools.edgrn_cli:main",
            "ecat-edcmp=eqtools.cli_tools.edcmp_cli:main",
            "ecat-generate-psgrn-template=eqtools.cli_tools.psgrn_template_cli:main",
            "ecat-generate-pscmp-template=eqtools.cli_tools.pscmp_template_cli:main",
            "ecat-generate-edgrn-template=eqtools.cli_tools.edgrn_template_cli:main",
            "ecat-generate-edcmp-template=eqtools.cli_tools.edcmp_template_cli:main",
        ],
    },
    cmdclass={
        'install': CustomInstallCommand,
    },
)