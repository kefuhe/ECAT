# 安装与环境检查

本页区分两种安装场景：第一次部署完整 ECAT，以及在已有 ECAT 环境中增量更新
`eqtools` 或 CSI。完整环境从 ECAT 仓库根目录建立；包级更新进入相应子目录执行。

ECAT 支持 64 位 Windows 和 Linux 上的 CPython 3.10、3.11 和 3.12。当前默认
推荐并重点验证 CPython 3.10；3.11 和 3.12 保留为支持的安装目标。推荐使用
Conda/conda-forge 建立基础环境，再用 `python -m pip` 安装本地源码。安装时直接
使用仓库提供的依赖清单。

## 1. 第一次安装完整 ECAT

先获取统一仓库，再从仓库根目录创建环境：

```bash
git clone https://github.com/kefuhe/ECAT.git
cd ECAT

conda create -n ecat -c conda-forge python=3.10 --file requirements/ecat-requirements.txt
conda activate ecat
```

上面是普通用户的默认命令，Conda 会按平台选择 BLAS 和 MPI 实现。如果已经确定要
使用 MKL、Intel MPI 或 Windows MS-MPI，应在这一步创建环境时选择，不要装完以后
再原地替换二进制运行库。默认命令不承诺 Intel MPI，也不要求 `I_MPI_*` 变量；
实际实现必须用安装后的 `MPI.get_vendor()` 结果判断。

### 1.1 安装前选择数值和 MPI 实现（可选）

下面不是安装后的修复命令，而是默认 `conda create` 的替代命令。只选一条；后续仍
执行同一个 `conda activate ecat`、安装 `okada4py` 和 ECAT 脚本。

| 目标 | 建议选择 | 说明 |
| --- | --- | --- |
| 最简单、优先兼容 | 默认命令 | 让 Conda 按平台解析实现，适合首次使用 |
| Linux/WSL 上使用 MKL | MKL + Open MPI | 保留 Linux 常用 Open MPI，只把 BLAS 明确为 MKL |
| Windows 上明确稳定组合 | MKL + MS-MPI | Windows 原生 MPI，配合 MKL |
| Linux/Windows 使用 Intel MPI | MKL + Intel MPI | 使用 Conda 内的 `impi_rt`，适合明确要比较 Intel 栈时 |

Linux/WSL，MKL + Open MPI：

```bash
conda create -n ecat --override-channels -c conda-forge --strict-channel-priority python=3.10 "libblas=*=*_mkl" openmpi --file requirements/ecat-requirements.txt
```

Windows，MKL + MS-MPI：

```powershell
conda create -n ecat --override-channels -c conda-forge --strict-channel-priority python=3.10 "libblas=*=*_mkl" msmpi --file requirements/ecat-requirements.txt
```

Linux/WSL 或 Windows，MKL + Intel MPI：

```bash
conda create -n ecat --override-channels -c conda-forge --strict-channel-priority python=3.10 "libblas=*=*_mkl" impi_rt --file requirements/ecat-requirements.txt
```

这三种组合均由 Conda 同时解析 NumPy/SciPy、BLAS、mpi4py 和 MPI runtime，不需要
预先安装完整 oneAPI。`impi_rt` 只提供 Intel MPI 运行时；只有需要 Intel 编译器、
性能分析器或用外部 Intel MPI 编译本地扩展时，才需要完整 oneAPI。MKL/Intel MPI
不保证对所有任务更快，选择后仍应按同一案例测速。更完整的层级和取舍见
[计算运行栈](../concepts/compute_runtime_stack.md)。

显式 provider 命令中的 `--override-channels` 和 `--strict-channel-priority` 只对本次
创建生效，用于避免 MKL/MPI 等编译型包被用户已有 channel 配置拆成不同二进制栈；
它们不会修改 `.condarc`。

### 1.2 创建环境时最常见的两个直接替代

如果 VPN/代理已经关闭，但 Linux/WSL 当前终端仍保留失效代理变量，可以直接把
默认创建命令替换为：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  conda create -n ecat -c conda-forge python=3.10 \
  --file requirements/ecat-requirements.txt
```

Windows PowerShell 中确认代理已关闭后，可对当前终端执行：

```powershell
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY -ErrorAction SilentlyContinue
conda create -n ecat -c conda-forge python=3.10 --file requirements/ecat-requirements.txt
```

如果已经完成 metadata 下载，但渠道混用或 solver 导致求解失败，直接使用只对本次
命令生效的隔离式版本：

```bash
conda create -n ecat --override-channels -c conda-forge \
  --strict-channel-priority --solver libmamba python=3.10 \
  --file requirements/ecat-requirements.txt
```

上述命令都不永久修改 `.condarc`。Windows 代理清理、证书问题、已有 pin，以及想
把代理处理与 MKL/MPI 选择组合使用时，再进入
[安装与运行故障排查](troubleshooting.md#2-conda-创建环境失败或很慢)。

`requirements/ecat-requirements.txt` 是 ECAT 唯一的用户环境清单，包含 CSI 与
eqtools 的直接运行依赖和兼容范围。

清单按“CSI 与 eqtools 共享”“仅 CSI”“仅 eqtools”分组。共享包会分别保留在两个
独立包的 `setup.py` 中，因为 CSI 和 eqtools 都直接使用它们；清单生成时只去重一次，
不能根据它在清单中的显示位置判断依赖归属。

依赖清单保留 Python 3.10--3.12 的兼容范围，但主安装命令明确选择经过最多实际
案例验证的 3.10。需要测试较新解释器时，可以改为：

```bash
conda create -n ecat -c conda-forge python=3.11 --file requirements/ecat-requirements.txt
# 或
conda create -n ecat -c conda-forge python=3.12 --file requirements/ecat-requirements.txt
```

这三个版本都必须使用与所选 CPython 和平台匹配的 `okada4py` wheel。3.11 或
3.12 遇到第三方 wheel 缺失时，优先退回推荐的 3.10 环境，不要删除 ECAT 的核心
依赖或绕过版本范围。

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
项目目录执行普通安装：

```bash
# 当前位于 ECAT 仓库根目录
git pull
conda activate ecat
cd eqtools
python -m pip install .
```

这会根据 `eqtools/setup.py` 检查直接依赖，只补装缺失项或调整超出兼容范围的包，
不需要重新安装完整 ECAT 环境。

只有需要直接编辑源码并让修改立即生效的维护者，才使用 editable 安装：

```bash
python -m pip install -e .
```

只有 CSI 源码或其依赖声明发生变化时，才需要单独更新 CSI：

```bash
# 当前位于 ECAT 仓库根目录
cd csi_cutde_mpiparallel
python -m pip install .
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

python -m pip install ".[geoexport]"   # Google Earth / KMZ 导出
python -m pip install ".[viewer]"      # Dash/Plotly 本地科研地图
python -m pip install ".[interaction]" # Bokeh 交互断层迹线编辑
```

具体用法分别见 [Google Earth 科研导出](../workflows/06_google_earth_export.md)、
[本地科研地图查看](../workflows/07_research_map_viewer.md) 和
[交互调整断层迹线](../workflows/02c_interactive_trace_editing.md)。

## 7. MPI、mpi4py 与 oneAPI

`mpi4py` 是 ECAT 基础依赖；`mpiexec` 由 Open MPI、MPICH、Intel MPI 或 MS-MPI
等 MPI 实现提供。Conda 通常会按平台解析一套匹配组合。普通用户不需要为了本机
多核或 MKL 加速而安装完整 oneAPI；Open MPI、MPICH 和 MS-MPI 同样能在本机启动
多进程，而 NumPy/SciPy 是否使用 MKL 与 MPI 实现是两件独立的事。

oneAPI 中与 ECAT 相关的 oneMKL、Intel MPI 和编译器作用不同。安装 oneAPI 本身
不会自动重新链接已有 NumPy/SciPy，也不会让 Open MPI 版 `mpi4py` 自动改用
Intel MPI。完整层级、各实现特点和常见配套见
[Python 数值计算、BLAS、MPI 与 oneAPI 的层级](../concepts/compute_runtime_stack.md)。

运行大型 SMC 任务前先检查：

```bash
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

正确的两进程结果应包含 `0 2` 和 `1 2`。如果出现两次 `0 1` 或两次 rank 0，通常
是 `mpiexec` 与 `mpi4py` 加载了不同 MPI 实现；按
[MPI 故障排查](troubleshooting.md#7-mpi-或-mpiexec-失败) 检查路径和运行库。
第一次运行和受控线程对照的跨 MPI 实现模板见
[并行运行基础](../concepts/parallel_process_rank_thread.md#1-首次运行的通用复制模板)。

MPI 检查失败不代表 CVXOPT/Clarabel、BLSE/VCE、数据读取或网格功能不可用；应把
MPI runtime 问题与普通 Python 包导入问题分开诊断。

## 8. 快速检查

```bash
python -c "import csi, eqtools, okada4py; print('ECAT imports succeeded')"
ecat-generate-downsample --help
ecat-generate-nonlinear --help
ecat-downsample --help
```

## 常见问题

如果遇到 Conda 无法求解、pip 准备调整 NumPy/SciPy、`okada4py` wheel 不匹配、
`mpiexec` 不可用、VSCode 无法解析导入，或 BLSE 长时间停在
`Initializing solver object`，进入[安装与运行故障排查](troubleshooting.md)。该页
给出问题原因、逐项检查顺序，以及不会永久修改用户环境的 MKL/OpenBLAS 线程测速
方法。
