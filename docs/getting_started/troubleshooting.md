# 安装与运行故障排查

本页只在默认安装或运行出现具体症状时使用。正常安装仍按
[安装与环境检查](installation.md) 的简短命令执行，不需要预先设置代理、固定
BLAS、指定 MPI 实现或修改线程数。

| 症状 | 先检查 |
| --- | --- |
| 下载失败、VPN 开关后仍连不上 | [网络、VPN 与残留代理](#网络vpn-与残留代理) |
| 已下载 metadata，但长时间停在 `Solving environment` | [渠道与求解器](#渠道与求解器) |
| BLSE 停在 `Initializing solver object` | [BLAS 后端与线程](#5-blse-初始化blas-后端与线程) |
| 两个 MPI 进程都显示 rank 0 | [MPI 启动器与运行库](#7-mpi-或-mpiexec-失败) |
| VSCode 标出导入波浪线 | [解释器和包来源](#1-先确认解释器和包来源) |

## 1. 先确认解释器和包来源

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
python -c "import csi, eqtools; print(csi.__file__); print(eqtools.__file__)"
python -m pip check
```

这些路径分别说明当前 Python 和实际导入的包来自哪里。默认推荐 CPython 3.10；
3.11 和 3.12 仍在支持范围内，但必须具有与平台和 CPython ABI 匹配的
`okada4py` 及其他编译型 wheel。

命令行可以运行、编辑器却标出导入波浪线时，通常是编辑器选用了另一个解释器；
让编辑器使用 `sys.executable` 显示的解释器即可。

## 2. Conda 创建环境失败或很慢

先使用安装页的默认命令：

```bash
conda create -n ecat -c conda-forge python=3.10 \
  --file requirements/ecat-requirements.txt
```

不要一开始就叠加下面所有参数。先根据停留阶段区分网络问题和依赖求解问题。

### 网络、VPN 与残留代理

Conda 可以从 `.condarc` 的 `proxy_servers` 或 `HTTP_PROXY`、`HTTPS_PROXY` 等环境
变量读取代理。VPN 已关闭不代表这些变量已经消失；如果它们仍指向不再监听的本地
端口，Conda 会继续尝试连接失效代理。

Linux/WSL 检查：

```bash
env | grep -iE '^(http|https|all)_proxy='
conda config --show proxy_servers
curl -I https://conda.anaconda.org/conda-forge/noarch/repodata.json
```

Windows PowerShell 检查：

```powershell
Get-ChildItem Env: | Where-Object Name -Match '^(HTTP|HTTPS|ALL)_PROXY$'
conda config --show proxy_servers
curl.exe -I https://conda.anaconda.org/conda-forge/noarch/repodata.json
```

如果代理正在使用且可以访问，就保留它。如果确认 VPN/代理已经关闭，但当前终端仍
存在失效变量，仅对这一次 Linux/WSL 命令忽略代理：

```bash
env -u http_proxy -u https_proxy -u all_proxy \
    -u HTTP_PROXY -u HTTPS_PROXY -u ALL_PROXY \
  conda create -n ecat -c conda-forge python=3.10 \
    --file requirements/ecat-requirements.txt
```

Windows PowerShell 可以在记录原值后，为当前终端临时清除并重新运行默认命令：

```powershell
Remove-Item Env:HTTP_PROXY, Env:HTTPS_PROXY, Env:ALL_PROXY -ErrorAction SilentlyContinue
conda create -n ecat -c conda-forge python=3.10 --file requirements/ecat-requirements.txt
```

需要继续使用代理时，应恢复原值或重新打开由代理软件正确初始化的终端。不要把
`ssl_verify: false` 当成 VPN/代理故障的常规解决方案；证书问题应配置可信证书。
Conda 的代理配置项见
[Conda configuration](https://docs.conda.io/projects/conda/en/stable/configuration.html)。

### 渠道与求解器

如果已经显示 `Collecting package metadata ... done`，随后停在
`Solving environment`，通常是在本地求解依赖，不是绘图或 ECAT 代码运行。先看
当前配置：

```bash
conda config --show-sources
conda config --show channels channel_priority solver pinned_packages
```

常见原因是多个渠道的编译型包、额外 pin 或较慢的 classic solver。只有确认属于
渠道/求解器问题时，才使用下面的隔离式备用命令：

```bash
conda create -n ecat \
  --override-channels -c conda-forge \
  --strict-channel-priority --solver libmamba \
  python=3.10 --file requirements/ecat-requirements.txt
```

该命令只影响本次创建，不永久修改用户 `.condarc`。`--override-channels` 避免本次
求解混入其他渠道；`--strict-channel-priority` 保持同一编译栈；
`--solver libmamba` 只是在当前未使用 libmamba 时提供更快求解器。conda-forge
不支持与 Anaconda `defaults` 混合的编译栈，详见
[conda-forge 渠道迁移说明](https://conda-forge.org/docs/user/transitioning_from_defaults/)。

如果网络和渠道问题同时存在，可以把上一节的 `env -u ...` 放在这条备用命令前；
只有确实同时命中两个症状时才组合使用。

## 3. pip 准备调整 NumPy、SciPy 或 Numba

当前稳定兼容范围是：

```text
numpy>=1.23,<2
scipy>=1.10,<1.12
numba>=0.58,<0.60
```

普通安装和更新使用：

```bash
python -m pip install .
```

如果 pip 计划把当前 NumPy、SciPy 或 Numba 调整到上述范围，说明当前包组合不符合
项目声明，而不是 `pip install .` 本身失效。优先使用新的 Python 3.10 ECAT
环境，不要把 `--no-deps` 作为普通安装方法绕过兼容检查。

## 4. 找不到 `okada4py`

CSI 在导入时会加载 `okada4py`。检查当前 Python 标签和系统架构：

```bash
python -c "import sys, platform; print(sys.version); print(platform.machine())"
```

然后安装与 CPython、ABI 和平台匹配的 wheel。例如 CPython 3.10 需要
`cp310-cp310` wheel。安装完成后检查实际来源：

```bash
python -c "import okada4py; print(okada4py.__file__)"
```

如果所选 3.11 或 3.12 没有匹配 wheel，优先使用推荐的 3.10 环境；不要安装与
当前解释器标签不一致的 wheel。

## 5. BLSE 初始化、BLAS 后端与线程

BLSE 初始化会组装数据协方差并执行逆矩阵、Cholesky 分解等稠密线性代数。数据量
较大时，`Initializing solver object` 后较长时间没有新日志不一定表示绘图阻塞或
程序死锁。线程过多也可能因调度、内存带宽和线程竞争而更慢。

先查看当前进程实际加载的数值后端：

```bash
python -c "import numpy, scipy; from threadpoolctl import threadpool_info; print(threadpool_info())"
```

重点查看：

- `internal_api`：常见为 `mkl` 或 `openblas`；
- `filepath`：实际加载的动态库路径；
- `num_threads`：当前线程数；
- `threading_layer`：常见为 `pthreads`、`openmp` 或 `intel`。

安装完整 oneAPI 不会自动把现有 NumPy/SciPy 改成 MKL。只有这里实际显示
`internal_api: mkl` 和 MKL 动态库路径，当前 Python 数值栈才在使用 MKL。不同
Python 扩展也可能各自带有 BLAS，因此输出中可能同时出现 MKL 和 OpenBLAS。

oneMKL 与 OpenBLAS 的作用、切换测试环境以及 oneAPI 组件关系见
[计算运行栈](../concepts/compute_runtime_stack.md)。

### 单进程临时测速

不要假设线程越多越快。单进程 BLSE 可以先用 8 线程作为保守起点，再对同一个
代表性案例测试 1、4、8、16 线程并记录总耗时。以下变量只对这一条命令启动的
进程生效。

Linux/WSL，OpenBLAS pthreads：

```bash
OPENBLAS_NUM_THREADS=1 python your_script.py
OPENBLAS_NUM_THREADS=4 python your_script.py
OPENBLAS_NUM_THREADS=8 python your_script.py
OPENBLAS_NUM_THREADS=16 python your_script.py
```

Linux/WSL，MKL：

```bash
MKL_NUM_THREADS=1 python your_script.py
MKL_NUM_THREADS=4 python your_script.py
MKL_NUM_THREADS=8 python your_script.py
MKL_NUM_THREADS=16 python your_script.py
```

Windows PowerShell，OpenBLAS：

```powershell
$env:OPENBLAS_NUM_THREADS = "8"
python your_script.py
Remove-Item Env:OPENBLAS_NUM_THREADS -ErrorAction SilentlyContinue
```

Windows PowerShell，MKL：

```powershell
$env:MKL_NUM_THREADS = "8"
python your_script.py
Remove-Item Env:MKL_NUM_THREADS -ErrorAction SilentlyContinue
```

如果原来已经设置过变量，应在测试前记录旧值并在测试后恢复。环境变量必须在启动
Python 前设置；NumPy/SciPy 已经加载后再修改可能不会生效。

OpenBLAS pthreads 构建读取 `OPENBLAS_NUM_THREADS`；OpenMP 构建通常读取
`OMP_NUM_THREADS`。后者还可能影响 CUTDE、Numba 或其他 OpenMP 代码，因此只在
`threading_layer` 明确显示 `openmp` 时进行单次临时测试。详见
[OpenBLAS 运行时变量](https://www.openmathlib.org/OpenBLAS/docs/runtime_variables/)
和 [oneMKL 线程说明](https://www.intel.com/content/www/us/en/docs/onemkl/developer-guide-linux/2023-1/setting-the-number-of-openmp-threads.html)。

## 6. MPI 进程与 BLAS 线程相互叠加

MPI 启动多个 Python 进程，BLAS 又可能在每个进程内部启动多个线程：

```text
总计算线程约等于 MPI 进程数 × 每个进程的 BLAS 线程数
```

例如 8 个 MPI 进程、每个进程 16 个 BLAS 线程，可能产生约 128 个计算线程。
大型 SMC 或 MPI 任务变慢时，应同时比较进程数和每进程 BLAS 线程数：

```bash
# OpenBLAS 环境
OPENBLAS_NUM_THREADS=1 mpiexec -n 8 python your_script.py
OPENBLAS_NUM_THREADS=4 mpiexec -n 8 python your_script.py

# MKL 环境
MKL_NUM_THREADS=1 mpiexec -n 8 python your_script.py
MKL_NUM_THREADS=4 mpiexec -n 8 python your_script.py
```

多进程任务先从每个 MPI 进程 1 个 BLAS 线程开始，再按实际物理核心数比较 2 或
4 线程；不要直接沿用单进程的 8 线程起点。Numba 的线程池由
`NUMBA_NUM_THREADS` 单独控制，不等同于 BLAS 线程。

在本地电脑运行 MPI 不需要 oneAPI。Open MPI、MPICH、Intel MPI 和 MS-MPI 都能
在单机启动多进程；Intel MPI 是否更快必须用同一案例、相同进程数和线程数实测。

## 7. MPI 或 `mpiexec` 失败

MPI 的完整层级是：MPI 标准 → Open MPI/MPICH/Intel MPI/MS-MPI 实现 → 动态库和
`mpiexec` → mpi4py Python 绑定。实现可以选择，但启动器和动态库必须属于同一
实现或明确兼容的 ABI，不能只替换其中一个。

先检查 Python 绑定实际加载的实现：

```bash
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
```

Linux/WSL 检查启动器搜索顺序：

```bash
echo "$CONDA_PREFIX"
command -v python
command -v mpiexec
type -a mpiexec
mpiexec --version
```

Windows PowerShell 检查：

```powershell
$env:CONDA_PREFIX
Get-Command python -All
Get-Command mpiexec -All
mpiexec --version
```

最后执行两进程自检：

```bash
mpiexec -n 2 python -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

正确输出应包含：

```text
0 2
1 2
```

如果得到两次 `0 1`、两次 rank 0，或出现 PMIx/PMI singleton 警告，说明两个
进程没有加入同一个 `MPI_COMM_WORLD`。最常见原因是系统、Conda、oneAPI 或
集群 module 提供了多套 MPI，`PATH` 中的 `mpiexec` 与 mpi4py 加载的动态库不匹配。

Linux/WSL 中，如果 Conda 环境本身提供了匹配启动器，可以先用绝对环境路径验证：

```bash
"$CONDA_PREFIX/bin/mpiexec" -n 2 "$CONDA_PREFIX/bin/python" -c \
  "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank(), MPI.COMM_WORLD.Get_size())"
```

若绝对路径正常而普通 `mpiexec` 失败，应调整 shell 初始化顺序，避免在所有终端中
全局加载另一套 MPI。不要卸载 oneAPI 来解决路径顺序；需要 Intel 编译器或
Intel MPI 时，使用单独终端或独立环境。

Windows 的 MS-MPI 或 Intel MPI 可以是系统级运行时，不一定安装在
`CONDA_PREFIX` 内；判断标准仍是 mpi4py 报告的实现与 `mpiexec`/DLL 配套，而不是
机械要求所有文件都位于同一目录。

如果明确要更换 MPI 实现，应在新环境中选择一整套配套，而不是在当前环境中只换
`mpiexec`。Open MPI、MPICH、Intel MPI 和 MS-MPI 的特点及安装组合见
[计算运行栈](../concepts/compute_runtime_stack.md)。

## 8. GDAL、PROJ 或 NumPy ABI 错误

典型信息包括：

```text
proj.db not found
numpy.ndarray size changed
binary incompatibility
```

这些问题通常来自编译型包和其运行时来自不同渠道，或 NumPy 升级后旧扩展没有
同步更新。优先在新的 Python 3.10 环境中从同一 Conda 渠道安装 GDAL、PROJ、
Rasterio、netCDF4、NumPy 和 SciPy，再执行普通 `python -m pip install .` 安装
项目源码。

## 9. 更新源码后行为没有变化

确认解释器和导入路径：

```bash
python -c "import sys, eqtools; print(sys.executable); print(eqtools.__file__)"
```

普通用户重新安装：

```bash
python -m pip install .
```

需要持续修改源码的维护者使用：

```bash
python -m pip install -e .
```

报告问题时，至少附上 Python 版本、`sys.executable`、包导入路径、`pip check`、
BLAS 检测结果、MPI vendor/launcher 结果和失败命令的完整错误信息。代理 URL 可能
包含用户名、密码或令牌，提交日志前必须隐藏敏感字段。
