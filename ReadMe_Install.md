### Installation Steps

#### We recommend installing Anaconda

We recommend using Anaconda to manage your Python environment. Please follow the steps below to install Anaconda and set up the `myecat` environment:

**Download and install Anaconda**:

- Visit the [Anaconda website](https://www.anaconda.com/products/distribution) and download the installer for your operating system.
- Follow the installation guide to complete the Anaconda installation.

**Note**: If you have already installed Anaconda on your system, you can ignore the instructions above.

#### Create and activate the `myecat` environment

```bash
# For Windows
conda create --name myecat --file conda-requirements-win-64.txt
# For Linux
conda create --name myecat --file conda-requirements-linux-64.txt

# Activate Environment
conda activate myecat

# Install additional dependencies
conda install scikit-learn-intelex

# For Windows
pip install -r pip-requirements-win-64.txt
# For Linux
pip install -r pip-requirements-linux-64.txt
```

**Note:** Use the appropriate `conda-requirements` and `pip-requirements` files based on your operating system:

* For Windows: `conda-requirements-win-64.txt` and `pip-requirements-win-64.txt`
* For Linux: `conda-requirements-linux-64.txt` and `pip-requirements-linux-64.txt`

Explanation: Be fault, conda will download packages from the `defaults` channel. If a package is not found in the `defaults` channel, conda will attempt to download it from the `conda-forge` channel. Note that the default Anaconda installation may not include the `conda-forge` channel. You can add it using the following command:

```bash
conda config --add channels conda-forge
```

#### Install our own packages and other necessary dependencies

1. Install ***eqtools*** and **csi** in **myecat**

```bash
#For Windows
chmod +x install.sh
./install.sh

# For Linux
.\install.bat
```

2. Install modified [okada4py](https://github.com/kefuhe/okada4py) in ***myecat* (See readme.md of [okada4py](https://github.com/kefuhe/okada4py) for details)**

```bash
# If you have not installed build in your environment.
pip install build

# Build and install; Only support install in python 3.7 and above with this way
python -m build
# The exact name of the .whl file will depend on your package version and Python version
pip install dist/okada4py-12.0.2-py3-none-any.whl
```

#### Install oneapi and config environment

1. Download [oneapi package](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-download.html?operatingsystem=linux&mpi-linux=offline)
2. Install ***oneapi***

```bash
chmod +x l_mpi_oneapi_p_2021.11.0.49494_offline.sh
sudo ./l_mpi_oneapi_p_2021.11.0.49494_offline.sh
```

3. Config environment

```bash
# If there are no oneAPI-related environment loading parameters in `~/.bashrc`, please add them manually.
# Intel MPI ifort
source ~/intel/oneapi/setvars.sh intel64
```
