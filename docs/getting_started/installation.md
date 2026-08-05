# 安装与环境检查

本页区分两种安装场景：第一次部署完整 ECAT，以及在已有 ECAT 环境中增量更新
`eqtools` 或 CSI。完整环境从 ECAT 仓库根目录建立；包级更新进入相应子目录执行。

ECAT 支持 64 位 Windows 和 Linux 上的 CPython 3.10、3.11 和 3.12。推荐使用
Conda/conda-forge 建立基础环境，再用 `python -m pip` 安装本地源码。不要用
`conda list`、`pip freeze` 或个人开发环境导出文件替代 ECAT 的依赖清单。

## 1. 第一次安装完整 ECAT

先获取统一仓库，再从仓库根目录创建环境：

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT

conda create -n ecat -c conda-forge --file requirements/ecat-requirements.txt
conda activate ecat
```

`requirements/ecat-requirements.txt` 是 ECAT 唯一的用户环境清单，包含 CSI 与
eqtools 的直接运行依赖和兼容范围。它不包含操作系统 build string、维护者机器上的
无关应用、PyMC、PyTensor 或 Theano。

清单允许 Python 3.10--3.12。若要明确选择版本，可在创建环境时加入版本约束：

```bash
conda create -n ecat -c conda-forge python=3.11 --file requirements/ecat-requirements.txt
```

Python 3.10、3.11 和 3.12 均属于支持范围；实际安装还需要确保下一节的
`okada4py` wheel 与所选 CPython 和平台匹配。

## 2. 安装必需的 okada4py

CSI 在导入时会加载 `okada4py`，因此它是标准 ECAT 环境的必需前置。优先从
[okada4py Releases](https://github.com/kefuhe/okada4py/releases) 下载匹配当前
CPython、ABI、操作系统和 CPU 架构的 wheel：

```bash
python -m pip install path/to/okada4py-<version>-cp<major><minor>-cp<major><minor>-<platform>.whl
python -c "import okada4py; print(okada4py.__file__)"
```

例如，CPython 3.12 需要 `cp312-cp312` wheel；`win_amd64` 表示 64 位 Windows，
`linux_x86_64` 表示 64 位 Linux。若没有匹配的 release wheel，可从公开固定标签
构建，但本机需要可用的 C++ 编译器：

```bash
python -m pip install "git+https://github.com/kefuhe/okada4py.git@v12.0.2"
python -c "import okada4py; print(okada4py.__file__)"
```

## 3. 安装统一仓库中的 CSI 与 eqtools

仍在 ECAT 仓库根目录时，运行对应平台脚本：

```bash
# Linux
chmod +x install.sh
./install.sh

# Windows
.\install.bat
```

脚本使用当前解释器的 `python -m pip`，先安装 CSI，再安装 eqtools，并检查两者
能否导入。若 Python 不在 3.10--3.12 范围内，或缺少 `okada4py`，脚本会停止并
给出提示。

## 4. 在已有环境中增量更新

日常更新 eqtools 不需要重新创建整个 Conda 环境。拉取新代码后，进入 eqtools
项目目录进行 editable 安装：

```bash
# 当前位于 ECAT 仓库根目录
git pull
conda activate ecat
cd eqtools
python -m pip install -e .
```

这会根据 `eqtools/setup.py` 检查直接依赖，只补装缺失项或调整超出兼容范围的包，
不会把维护者环境中的所有包写入用户环境。

只有 CSI 源码或其依赖声明发生变化时，才需要单独更新 CSI：

```bash
# 当前位于 ECAT 仓库根目录
cd csi_cutde_mpiparallel
python -m pip install -e .
```

维护者在独立 eqtools 或 CSI 开发仓库中验证时，复用已经建立的 `ecat` 环境，并在
各自仓库根目录直接执行：

```bash
conda activate ecat
python -m pip install -e .
```

该命令是包级增量安装，不是空白环境的完整 ECAT 部署。第一次使用 ECAT 时仍应从
统一仓库和唯一环境清单开始。

## 5. 基础环境已包含网格、SAR 和反演依赖

网格生成以及常用 SAR/InSAR、GeoTIFF 读取属于基础功能。Gmsh、meshio、
GeoPandas、GDAL、Rasterio、Xarray、cutde、CVXOPT、Clarabel、mpi4py 和 Numba
均由基础依赖声明管理，不需要再安装 mesh 或 SAR extra。

标准路线同样属于基础功能：先用 Bayesian SMC 非线性反演估计断层顶边中点的
经纬度和深度等紧凑几何参数，再固定几何并用 BLSE/VCE 线性反演求分布式滑动。

## 6. 按需增加 eqtools 功能

以下 extra 是 eqtools 包的增量安装。ECAT 用户先从统一仓库根目录进入 `eqtools`；
独立 eqtools 开发仓库中已经位于包根目录，不需要再次 `cd`：

```bash
# 当前位于 ECAT 仓库根目录时执行这一行
cd eqtools

python -m pip install -e ".[geoexport]"   # Google Earth / KMZ 导出
python -m pip install -e ".[viewer]"      # Dash/Plotly 本地科研地图
python -m pip install -e ".[interaction]" # Bokeh 交互断层迹线编辑
```

具体用法分别见 [Google Earth 科研导出](../workflows/06_google_earth_export.md)、
[本地科研地图查看](../workflows/07_research_map_viewer.md) 和
[交互调整断层迹线](../workflows/02c_interactive_trace_editing.md)。

## 7. MPI、mpi4py 与 oneAPI

`mpi4py` 是 ECAT 基础依赖，因为当前 SMC 实现直接导入它；`mpiexec` 则由 MPI
运行时提供，用于启动多个 Python 进程。Conda/conda-forge 通常会为 `mpi4py`
解析匹配的 MPI runtime 和启动器，因此普通 Windows 或 Linux 用户不需要另外安装
完整 oneAPI。

只有集群明确规定使用 Intel MPI 时，才按集群规范加载 oneAPI/Intel MPI，并确保
当前 `mpi4py` 与该 MPI 实现绑定一致。不要混用系统中的另一套 `mpiexec` 和当前
环境中的 `mpi4py`。

运行大型 SMC 任务前先检查：

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

MPI 检查失败不代表 CVXOPT/Clarabel、BLSE/VCE、数据读取或网格功能不可用；应把
MPI runtime 问题与普通 Python 包导入问题分开诊断。

## 8. 快速检查

```bash
python -c "import csi, eqtools, okada4py; print('ECAT imports succeeded')"
ecat-generate-downsample --help
ecat-generate-nonlinear --help
ecat-downsample --help
```

## 9. 依赖维护边界

普通用户不需要运行依赖生成脚本。只有将独立验证过的 eqtools 或 CSI 更新集成到
ECAT 时，维护者才从 ECAT 仓库根目录执行：

```bash
python scripts/generate_requirements.py --check
```

该工具从两个包的 `install_requires` 聚合唯一环境清单。独立包仓库以自己的
`setup.py` 为依赖事实来源，不另外维护完整环境导出。同步和发布顺序见
[独立开发仓库与 ECAT 集成](../developer/release_sync.md)。

## 常见问题

- `No module named 'okada4py'`：安装与当前 CPython 和平台匹配的 wheel，再运行
  ECAT 安装脚本。
- Conda 无法求解：使用新的 Python 3.10、3.11 或 3.12 环境和 conda-forge；不要
  任意删除唯一清单中的依赖或复制旧平台 build string。
- 更新源码后行为没有变化：确认已激活正确环境，并在实际发生变化的包目录执行了
  `python -m pip install -e .`。
- extra 安装失败或装到了错误项目：确认当前目录是包含 eqtools `setup.py` 的目录，
  再使用 `.[viewer]`、`.[interaction]` 或 `.[geoexport]`。
- `mpiexec` 不可用：先确认当前环境中的 MPI runtime 和 `mpi4py` 匹配；不需要为了
  普通非 MPI 工作流安装完整 oneAPI。
