# Python 数值计算、BLAS、MPI 与 oneAPI 的层级

ECAT 的安装包含 Python 包、数值运行库和并行运行时三个层次。它们经常由 Conda
一起安装，但不是同一种东西。理解这一层级有助于判断“包能导入但运行很慢”、
“`mpiexec` 启动后所有进程都是 rank 0”以及“安装了 oneAPI 但 NumPy 没变快”等
问题。

如果还不清楚进程、MPI rank、进程内线程、物理核心和 affinity 的区别，先读
[进程、MPI Rank、线程与 CPU 亲和性](parallel_process_rank_thread.md)。本页继续
解释这些计算单位由哪些 Python 包和底层运行库实现。

```text
ECAT / eqtools / CSI Python 代码
        │
        ├── NumPy / SciPy / CVXOPT ── BLAS/LAPACK 实现
        │                              ├── oneMKL
        │                              └── OpenBLAS
        │
        ├── mpi4py ── MPI 动态库 ── MPI 启动器 mpiexec
        │             ├── Open MPI
        │             ├── MPICH
        │             ├── Intel MPI
        │             └── MS-MPI
        │
        └── Numba / OpenMP / CUTDE 等其他并行运行时
```

## 1. oneAPI 不是一个单独的“加速开关”

Intel oneAPI 是一组工具和运行库。与 ECAT 最相关的是：

| oneAPI 组件 | 主要作用 | 会不会自动改变现有 Python 包 |
| --- | --- | --- |
| oneMKL | BLAS、LAPACK、FFT 等数值计算 | 不会；NumPy/SciPy 必须实际链接到 MKL |
| Intel MPI | 启动和通信多个 MPI 进程 | 不会；`mpi4py` 必须与 Intel MPI 兼容 |
| Intel C/C++/Fortran 编译器 | 编译本地 C、C++、Fortran 代码 | 不会；预编译 wheel 不会自动重新编译 |
| VTune、Advisor 等工具 | 性能分析和调优 | 只负责分析，不替换数值后端 |

因此，安装完整 oneAPI 或执行 `setvars.sh` 本身不会重新链接已经安装的
NumPy、SciPy、CVXOPT 或 mpi4py。`setvars.sh` 会修改 `PATH`、
`LD_LIBRARY_PATH` 和编译器相关变量；如果它把 Intel MPI 的 `mpiexec` 放到
Conda 环境前面，反而可能与 Conda 中的 Open MPI 版 mpi4py 冲突。

Intel 官方把 [oneMKL](https://www.intel.com/content/www/us/en/docs/oneapi/programming-guide/2025-0/intel-oneapi-math-kernel-library-onemkl.html)
描述为高度优化、支持多线程的数值库，把
[Intel MPI](https://www.intel.com/content/www/us/en/developer/tools/oneapi/mpi-library.html)
描述为基于 MPICH 体系、面向多种通信互连的 MPI 实现。两者可以独立使用：
NumPy/SciPy 使用 oneMKL 时，mpi4py 仍然可以使用 Open MPI。

## 2. BLAS/LAPACK、oneMKL 与 OpenBLAS

BLAS/LAPACK 提供矩阵乘法、分解、求解等底层运算接口。NumPy、SciPy 和部分
CVXOPT 路径调用这些接口；oneMKL 和 OpenBLAS 是两种实现。

| 实现 | 常见场景 | 特点 |
| --- | --- | --- |
| oneMKL | Windows Conda、Intel CPU 工作站 | 对许多稠密线性代数进行了 Intel 平台优化，带有自己的线程运行时 |
| OpenBLAS | Linux/WSL Conda 和许多 Python wheel | 开源、跨 CPU 厂商、部署方便，也支持多线程 |

conda-forge 的常见默认是 Windows 使用 MKL、Linux 使用 OpenBLAS，但最终必须以
当前环境实际加载结果为准。同一 Python 进程也可能同时出现多个后端，例如
NumPy 使用 MKL，而某个独立 wheel 自带 OpenBLAS。

检查实际后端和动态库路径：

```bash
python -c "import numpy, scipy; from threadpoolctl import threadpool_info; print(threadpool_info())"
```

重点查看 `internal_api`、`filepath`、`num_threads` 和 `threading_layer`。如果只需
查看 NumPy 构建信息，也可以运行：

```bash
python -c "import numpy; numpy.show_config()"
```

### 物理核心、逻辑 CPU 与 `num_threads`

性能设置中常见的三个数字含义不同：

| 名称 | 含义 |
| --- | --- |
| 物理核心数 | CPU 上实际的计算核心数量，适合作为“进程数 × 每进程线程数”的保守起点 |
| 逻辑 CPU 数 | 操作系统可调度的硬件线程数量；启用 SMT/超线程时通常大于物理核心数 |
| `threadpool_info()` 的 `num_threads` | 该数值库当前配置或允许使用的线程上限，不是程序此刻正在占用的 CPU 数 |

Linux/WSL 可用 `lscpu` 查看 `CPU(s)`、`Core(s) per socket` 和 `Socket(s)`；Windows
可在任务管理器的 CPU 性能页查看“内核”和“逻辑处理器”。操作系统、容器、作业
调度器或 CPU affinity 还可能只向当前进程开放其中一部分逻辑 CPU。

Windows、原生 Linux 与 WSL 怎样呈现 CPU、环境变量和 MPI affinity，以及新用户
应先用默认命令还是先设线程，见
[并行运行基础](parallel_process_rank_thread.md#7-windows原生-linux-与-wsl-的一般差异)。

因此，看到 OpenMP 或 BLAS “可见 32/64/104 个线程”，只说明运行时的默认上限或
当前进程可调度范围，并不表示这些线程都在工作，也不能证明这个数字就是最佳配置。
多个 `threadpool_info()` 条目也不能直接相加：NumPy、CVXOPT、CUTDE 或其他扩展
可能各自加载一个线程池，但它们未必在同一阶段同时满负荷运行。

常用变量控制的是不同层次：

| 变量 | 主要控制对象 | 说明 |
| --- | --- | --- |
| `MKL_NUM_THREADS` | oneMKL | NumPy/SciPy/CVXOPT 实际链接 MKL 时生效 |
| `OPENBLAS_NUM_THREADS` | OpenBLAS pthreads 构建 | 某些 wheel 可能单独携带 OpenBLAS |
| `OMP_NUM_THREADS` | OpenMP 运行时 | 也可能影响 CUTDE 或 OpenMP 版 BLAS；范围比单一 BLAS 更广 |
| `NUMBA_NUM_THREADS` | Numba parallel 线程池 | 只影响使用并行 Numba 内核的代码，不等同于 BLAS 线程 |

`I_MPI_PIN` 不属于数值线程变量；它由 Intel MPI 读取，只控制 Intel MPI 是否执行
进程 pinning。完整变量归属、默认来源和 shell 作用范围见
[前置环境变量怎样生效、由谁配置](parallel_process_rank_thread.md#6-前置环境变量怎样生效由谁配置)。

这些变量不应写成 ECAT 的永久默认值。先确认实际后端，再在单条运行命令中临时
设置并比较代表性案例；具体测速矩阵见
[安装与运行故障排查](../getting_started/troubleshooting.md#5-blse-初始化blas-后端与线程)。

安装 oneAPI 后，如果输出仍是 `openblas`，说明当前 NumPy/SciPy 没有使用
oneMKL；这不是安装失败，而是 Python 包仍链接到原来的 BLAS 实现。

### 创建环境前是否选择 MKL

MKL 在部分 Intel CPU 和稠密矩阵任务上可能更快，但不能只根据库名判断。矩阵
规模、线程数、内存带宽以及是否同时启动 MPI 都会改变结果。ECAT 不在
`setup.py` 中强制 MKL 或 OpenBLAS，也不自动永久 pin 用户的 BLAS。

如果第一次安装前已经决定使用 MKL，应直接选择
[安装页的 MKL/MPI 创建命令](../getting_started/installation.md#11-安装前选择数值和-mpi-实现可选)，
让 NumPy/SciPy、BLAS 和 MPI 一次求解完成，不要先创建默认环境再原地切换。

需要严格比较两个后端时，可另建两个完整测试环境：

```bash
# MKL 对照环境
conda create -n ecat-mkl --override-channels -c conda-forge \
  --strict-channel-priority python=3.10 "libblas=*=*_mkl" \
  --file requirements/ecat-requirements.txt

# OpenBLAS 对照环境
conda create -n ecat-openblas --override-channels -c conda-forge \
  --strict-channel-priority python=3.10 "libblas=*=*_openblas" \
  --file requirements/ecat-requirements.txt
```

只对同一个代表性案例比较总耗时。不要在已经稳定工作的科研环境中直接切换 BLAS
后再用旧结果判断，也不要默认把某台机器的 pin 写进公共依赖。

## 3. MPI 的层级：标准、实现、Python 绑定和启动器

[MPI Forum](https://www.mpi-forum.org/) 定义 MPI 标准；它不是一个可以直接安装
运行的程序。Open MPI、MPICH、Intel MPI 和 MS-MPI 是不同厂商或社区提供的 MPI
实现。`mpi4py` 是 MPI 的 Python 绑定，`mpiexec` 是某个 MPI 实现提供的进程
启动器。

本节列出可选实现，不表示 ECAT 默认采用 Intel MPI。普通安装由 Conda 按平台和
当前依赖解析出匹配组合；安装后以 `MPI.get_vendor()` 为准。通用运行模板不使用
任何 `I_MPI_*`、Open MPI 或 MS-MPI 专属参数。

```text
MPI 标准
   ↓ 由不同项目实现
Open MPI / MPICH / Intel MPI / MS-MPI
   ↓ 提供动态库和 mpiexec
mpi4py 调用动态库，mpiexec 启动多个 Python 进程
```

| MPI 实现 | 主要平台和特色 | 常见配套 |
| --- | --- | --- |
| Open MPI | Linux/macOS 与 HPC 集群常见；支持多种调度器、网络和 PMIx 运行时 | `mpi4py` + `openmpi` |
| MPICH | 开源、可移植、适合工作站和集群；许多商业/集群 MPI 由其派生或采用其 ABI | `mpi4py` + `mpich` |
| Intel MPI | Linux/Windows；属于 oneAPI，基于 MPICH 体系并面向 Intel/兼容 CPU、OFI 和集群互连调优 | `mpi4py` + `impi_rt`，或与系统 Intel MPI 匹配的 mpi4py |
| MS-MPI | Windows 原生 MPI，实现面向 Windows/HPC Pack | `mpi4py` + `msmpi` |

[mpi4py 官方安装说明](https://mpi4py.readthedocs.io/en/stable/install.html)
列出了 conda-forge 的四种配套：`mpi4py + openmpi`、`mpi4py + mpich`、
`mpi4py + impi_rt` 和 `mpi4py + msmpi`。ECAT 已把 `mpi4py` 放在基础依赖中；用户
只需在 `conda create` 时额外指定所选 runtime，Conda 就会挑选对应的 mpi4py
build。完整可复制命令见
[安装前选择数值和 MPI 实现](../getting_started/installation.md#11-安装前选择数值和-mpi-实现可选)。

不要先创建默认环境，再在同一个环境中依次安装不同 MPI runtime。ECAT 默认安装
让 Conda 按平台选择配套；只有明确需要另一实现时才在创建新环境时指定。

Intel MPI 参与
[MPICH ABI Compatibility Initiative](https://www.mpich.org/abi/)，因此某些针对
MPICH ABI 构建的程序可以使用 Intel MPI。ABI 兼容不表示 Intel MPI 可以启动
Open MPI 构建的 mpi4py；Open MPI 与 MPICH/Intel MPI 属于不同 ABI 家族。

## 4. 本地电脑并行是否需要 oneAPI

不需要。MPI 可以在同一台电脑上启动多个进程，也可以跨节点；Open MPI、MPICH、
Intel MPI 和 MS-MPI 都能进行本机多进程。BLAS 也可以在一个 Python 进程内部使用
多个 CPU 线程。

对 ECAT 常见任务，应区分：

| 工作负载 | 主要并行层 | oneAPI 的潜在作用 |
| --- | --- | --- |
| 单进程 BLSE/VCE 稠密线性代数 | BLAS/LAPACK 线程 | oneMKL 可能更快，但必须实测 |
| Bayesian SMC 多链/多粒子任务 | MPI 多进程，进程内还可能有 BLAS 线程 | Intel MPI 可能有优势，但本机不保证快于 Open MPI/MPICH |
| 已安装的 Python wheel | wheel 自己链接的运行库 | Intel 编译器不会自动优化已有 wheel |
| NumPy/SciPy CPU 运算 | 实际加载的 MKL/OpenBLAS | 安装 oneAPI 不等于自动使用 MKL |

在一台工作站上，MPI 实现之间的差异通常不应先于“进程数 × 每进程 BLAS 线程数”
进行优化。若 8 个 MPI 进程各启动 16 个 BLAS 线程，就可能产生约 128 个计算线程，
线程争用反而会降低速度。先让程序正确运行，再用同一案例分别比较实现和线程数。

## 5. 环境和依赖路径怎样决定实际加载内容

排查时需要同时看五条路径：

| 路径 | 检查内容 |
| --- | --- |
| `sys.executable` | 当前运行的是哪个 Python |
| `package.__file__` | 实际导入了哪份 csi/eqtools |
| `CONDA_PREFIX` | 当前 Conda 环境根目录 |
| `PATH` 中的 `mpiexec` | 实际使用哪个 MPI 启动器 |
| BLAS/MPI library version 或 `filepath` | Python 最终加载哪个动态库 |

Linux/WSL：

```bash
echo "$CONDA_PREFIX"
python -c "import sys, csi, eqtools; print(sys.executable); print(csi.__file__); print(eqtools.__file__)"
command -v python
command -v mpiexec
type -a mpiexec
mpiexec --version
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
```

Windows PowerShell：

```powershell
$env:CONDA_PREFIX
python -c "import sys, csi, eqtools; print(sys.executable); print(csi.__file__); print(eqtools.__file__)"
Get-Command python -All
Get-Command mpiexec -All
mpiexec --version
python -c "from mpi4py import MPI; print(MPI.get_vendor()); print(MPI.Get_library_version())"
```

在 Linux/WSL 中，`source oneapi/setvars.sh` 可能把 Intel MPI 的目录放到 `PATH`
前面。此时终端虽然显示已激活 Conda 环境，`mpiexec` 仍可能来自 oneAPI。最稳妥
的做法是不要在 shell 启动文件中全局加载完整 oneAPI；需要 Intel 编译器或
Intel MPI 时，在单独终端中按任务加载。

## 6. ECAT 的默认选择原则

- 默认用户按照安装页创建 Python 3.10 环境，不需要先理解或指定 BLAS/MPI 实现；
- Conda 负责解析同一渠道中的 Python 包和编译运行库，pip 负责安装 ECAT 源码；
- ECAT 不强制所有平台使用 MKL，也不要求安装完整 oneAPI；
- 想比较 MKL 时另建对照环境，不修改稳定环境；
- 想使用 Intel MPI 时，选择与 Intel MPI 配套的 mpi4py，而不是只替换 `mpiexec`；
- 出现安装、代理、速度或 MPI rank 问题时，进入
  [安装与运行故障排查](../getting_started/troubleshooting.md) 使用对应备用命令。
