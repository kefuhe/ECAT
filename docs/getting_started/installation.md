# 安装与环境检查

ECAT 支持 64 位 Windows 和 Linux 上的 CPython 3.10、3.11 和 3.12。推荐环境只包含
ECAT 的直接运行依赖，不复刻维护者机器的完整 Conda/Pip 环境。

## 1. 创建环境

```bash
conda create -n ecat -c conda-forge --file requirements/ecat-requirements.txt
conda activate ecat
```

清单允许 Python 3.10--3.12。若需要指定某个支持版本，可在创建命令中额外加入
`python=3.11`（或 `python=3.10`、`python=3.12`）。

仓库只保留 `requirements/ecat-requirements.txt` 这一个用户环境清单。它使用
经过测试的兼容版本范围，不包含平台 build string、PyMC/PyTensor/Theano，
也不包含 Flask、Jupyter、构建工具等传递或开发依赖。

## 2. 安装必需的 okada4py

当前 `csi` 在导入时会加载 `okada4py`，因此它是标准 ECAT 安装的必需前置。
请从 [okada4py Releases](https://github.com/kefuhe/okada4py/releases) 下载与
当前 CPython 版本和本机平台匹配的预编译 wheel，然后在已激活的环境中执行：

```bash
python -m pip install path/to/okada4py-<version>-cp<major><minor>-cp<major><minor>-<platform>.whl
python -c "import okada4py; print(okada4py.__file__)"
```

`win_amd64` 对应 64 位 Windows，`linux_x86_64` 对应 64 位 Linux。若提示
wheel 不受支持，应更换与 Python 版本、ABI、操作系统和 CPU 架构都匹配的文件。
例如 CPython 3.12 需要 `cp312-cp312` wheel。若 Releases 中没有对应文件，可从公开
固定标签构建（需要 C++ 编译器）：

```bash
python -m pip install "git+https://github.com/kefuhe/okada4py.git@v12.0.2"
python -c "import okada4py; print(okada4py.__file__)"
```

## 3. 安装 ECAT

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT

# Linux
chmod +x install.sh
./install.sh

# Windows
.\install.bat
```

安装脚本会使用 `python -m pip`，先安装 `csi`，再安装 `eqtools`，并检查两者
能否导入。若 Python 不在 3.10--3.12 范围内或缺少 `okada4py`，脚本会立即给出明确提示。

开发模式可使用：

```bash
python -m pip install -e csi_cutde_mpiparallel
python -m pip install -e eqtools
```

## 4. 基础环境已包含网格与 SAR

网格生成以及常用 SAR/InSAR、GeoTIFF 读取功能属于基础环境。Gmsh、meshio、
GeoPandas、GDAL、Rasterio 和 Xarray 已包含在
`requirements/ecat-requirements.txt` 中，不需要再单独安装 mesh 或 SAR extra。

## 5. 按需安装可选功能

```bash
python -m pip install -e "eqtools[geoexport]"   # Google Earth 导出
python -m pip install -e "eqtools[viewer]"      # Dash/Plotly 地图查看器
python -m pip install -e "eqtools[interaction]" # Bokeh 交互断层迹线编辑
```

标准流程仍是：先用 SMC 非线性反演估计紧凑断层几何，再在固定几何上用
BLSE/VCE 线性反演求分布式滑动。它们的运行依赖已在基础环境中。
ECAT 不再安装或支持 `pymc`、`pytensor`、`theano`；当前 SMC 使用
`mpi4py` 和 `numba`。

## 6. 快速检查

```bash
python -c "import csi, eqtools, okada4py; print('ECAT imports succeeded')"
ecat-generate-downsample --help
ecat-generate-nonlinear --help
ecat-downsample --help
```

如需 MPI 并行采样，先检查 MPI 运行时：

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

## 依赖维护

不要再使用 `conda list`、`pip freeze` 或个人 `cutde` 环境导出 requirements；
这些方式会混入与 ECAT 无关的包和传递依赖。修改源码导入或包依赖声明后，运行：

```bash
python scripts/generate_requirements.py --check
```

该脚本从两个包的 `install_requires` 生成唯一环境清单，并报告源码导入与
安装元数据的差异。若需要精确复现实验，应在干净测试环境中单独生成平台锁文件，
不要把锁文件当作普通用户的安装说明。
