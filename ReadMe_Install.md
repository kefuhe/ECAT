### 所有依赖安装步骤

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

2. okada4py安装

```bash
cd path_to_okada4py
export CC=gcc
python setup.py build
python setup.py install --user --prefix=
```
