### Installation Steps

#### Install external dependencies

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

After installing `okada4py`, the package should be located in the following directories depending on the operating system:

- **Linux**: The package should be located in a directory similar to `./.local/lib/python3.10/site-packages/okada4py`.
- **Windows**: The package should be located in a directory similar to `python310\Lib\site-packages\okada4py`.

#### Install oneapi and config environment

1. Download [oneapi package](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-download.html?operatingsystem=linux&mpi-linux=offline)
2. Install ***oneapi***

```bash
chmod +x l_mpi_oneapi_p_2021.11.0.49494_offline.sh
sudo ./l_mpi_oneapi_p_2021.11.0.49494_offline.sh
```

3. Config environment

```bash
# 如果~/.bahsrc中没有oneapi相关环境加载参数
# Intel MPI ifort
source ~/intel/oneapi/setvars.sh intel64
```
