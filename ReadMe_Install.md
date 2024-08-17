### 最终安装步骤

#### 安装外部依赖

```bash
conda create --name myecat --channel conda-forge --file conda-requirements.txt
# Activate Environment
conda activate myecat
conda install scikit-learn-intelex
pip install -r pip-requirements.txt
```

#### 安装自有包及必要其他必要依赖

1. eqtools和csi安装

```bash
chmod +x install.sh
./install.sh
```

2. [okada4py](https://github.com/jolivetr/okada4py)安装

```bash
cd path_to_okada4py
export CC=gcc
python setup.py build
python setup.py install --user --prefix=
```

#### 安装oneapi并加载环境

1. 下载[oneapi安装包](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library-download.html?operatingsystem=linux&mpi-linux=offline)
2. 安装及配置环境

```bash
chmod +x l_mpi_oneapi_p_2021.11.0.49494_offline.sh
sudo ./l_mpi_oneapi_p_2021.11.0.49494_offline.sh
```

3. 配置环境

```bash
# 如果~/.bahsrc中没有oneapi相关环境加载参数
# Intel MPI ifort
source ~/intel/oneapi/setvars.sh intel64
```


