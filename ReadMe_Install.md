### Installation Steps

#### We recommend installing Anaconda

We recommend using Anaconda to manage your Python environment. Please follow the steps below to install Anaconda and set up the `myecat` environment:

**Download and install Anaconda**:

- Visit the [Anaconda website](https://www.anaconda.com/products/distribution) and download the installer for your operating system.
- Follow the installation guide to complete the Anaconda installation.

#### Create and activate the `myecat` environment

```bash
conda create --name myecat --channel conda-forge --file conda-requirements.txt
# Activate Environment
conda activate myecat
conda install scikit-learn-intelex
pip install -r pip-requirements.txt
```

#### Install our own packages and other necessary dependencies

1. Install ***eqtools*** and **csi** in **myecat**

```bash
chmod +x install.sh
./install.sh
```

2. Install [okada4py](https://github.com/jolivetr/okada4py) in ***myecat***

```bash
cd path_to_okada4py
export CC=gcc
python setup.py build
python setup.py install --user --prefix=
```

After installing `okada4py` in the `myecat` environment (activated using `conda activate myecat`), the package should be located in the following directories depending on the operating system:

- **Linux**: The package should be located in a directory similar to `./.local/lib/python3.10/site-packages/okada4py`.
- **Windows**: The package should be located in a directory similar to `python310\Lib\site-packages\okada4py`.

**Note**: When using the `--prefix` option, make sure there is no content after it, including spaces.

3. If you encounter the issue of not finding the `okada4py` package during installation, you can solve it by following these steps:

   1. Clone the project to `~/anaconda3/envs/cutde/lib/python3.10/site-packages`, so you would have a folder named `okada4py`.
   2. Run `python setup.py build` and `python setup.py install`, so you could get `okada4py-12.0.2-py3.12-linux-x86_64.egg` in "`~/.local/lib/python3.10/site-packages/`".
   3. Copy all files in `okada4py-12.0.2-py3.12-linux-x86_64.egg` to the folder `okada4py`.
   4. Run `test.py`, and you can get two figures and no error toasted.

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
